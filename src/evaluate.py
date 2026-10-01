"""Evaluate AgriGuide.

  python -m src.evaluate           # retrieval metrics + threshold analysis (no Groq calls)
  python -m src.evaluate --full    # also run end-to-end: answers, refusals, keywords, faithfulness

Results are printed and saved to eval/results.md (paste the tables into your README)
and eval/results.json.
"""
import argparse
import json
import re
import sys
import time

from . import config
from .generator import build_context
from .retriever import HybridRetriever

EVAL_DIR = config.ROOT / "eval"
QUESTIONS_PATH = EVAL_DIR / "questions.json"
KS = (1, 3, 5)
DEPTH = 20
MODES = ["dense", "bm25", "hybrid", "hybrid+rerank"]
MODE_LABELS = {
    "dense": "Vector only (FAISS)",
    "bm25": "Keyword only (BM25)",
    "hybrid": "Hybrid (FAISS + BM25, RRF)",
    "hybrid+rerank": "Hybrid + rerank",
}


# ------------------------------------------------------------------ helpers
def load_questions():
    return json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))


def is_hit(chunk, expected):
    src = chunk["source"].lower()
    return any(e["source"].lower() in src and chunk["page"] in e["pages"] for e in expected)


def first_hit_rank(chunks, expected):
    for i, c in enumerate(chunks, start=1):
        if is_hit(c, expected):
            return i
    return None


def summarize(ranks):
    n = len(ranks)
    out = {f"hit@{k}": sum(1 for r in ranks if r and r <= k) / n for k in KS}
    out["mrr"] = sum(1 / r for r in ranks if r) / n
    return out


def with_retry(fn, tries=3):
    for i in range(tries):
        try:
            return fn()
        except Exception as exc:  # rate limits / transient API errors
            if i == tries - 1:
                raise
            print(f"   (retrying after error: {exc})")
            time.sleep(5 * (i + 1))


# ------------------------------------------------------- 1. retrieval eval
def retrieval_eval(retriever, reranker, questions):
    ranks = {m: [] for m in MODES}
    misses = []
    top_scores = {}

    for n, q in enumerate(questions, start=1):
        text = q["question"]
        print(f"  [{n}/{len(questions)}] {text[:70]}")
        hybrid = retriever.search(text, top_k=DEPTH, mode="hybrid")
        reranked = reranker.rerank(text, hybrid, top_k=DEPTH)
        top_scores[q["id"]] = reranked[0]["rerank_score"] if reranked else 0.0
        if not q["answerable"]:
            continue

        exp = q["expected"]
        ranks["dense"].append(first_hit_rank(retriever.search(text, top_k=DEPTH, mode="dense"), exp))
        ranks["bm25"].append(first_hit_rank(retriever.search(text, top_k=DEPTH, mode="bm25"), exp))
        ranks["hybrid"].append(first_hit_rank(hybrid, exp))
        r = first_hit_rank(reranked, exp)
        ranks["hybrid+rerank"].append(r)
        if r is None or r > 5:
            misses.append({"id": q["id"], "question": text, "rank": r,
                           "top3": [f"{c['source']} p.{c['page']}" for c in reranked[:3]]})

    metrics = {m: summarize(ranks[m]) for m in MODES}
    return metrics, misses, top_scores


def threshold_analysis(questions, top_scores):
    text = {q["id"]: q["question"] for q in questions}
    ans = {q["id"]: top_scores[q["id"]] for q in questions if q["answerable"]}
    ood = {q["id"]: top_scores[q["id"]] for q in questions
           if not q["answerable"] and q.get("kind") == "out_of_domain"}
    missing = {q["id"]: top_scores[q["id"]] for q in questions
               if not q["answerable"] and q.get("kind") == "in_domain_missing"}
    low_id, high_id = min(ans, key=ans.get), max(ood, key=ood.get)
    lowest_ans, highest_ood = ans[low_id], ood[high_id]

    if lowest_ans > highest_ood:
        suggested, separable = round(min(0.30, highest_ood + 0.25 * (lowest_ans - highest_ood)), 2), True
    else:
        # Scores overlap: keep the threshold below every real question so nothing valid is
        # refused; the LLM NOT_FOUND check handles the rest.
        suggested, separable = round(min(0.05, max(0.01, lowest_ans * 0.5)), 2), False

    return {
        "separable": separable,
        "lowest_answerable_id": low_id, "lowest_answerable_question": text[low_id],
        "lowest_answerable_top_score": lowest_ans,
        "highest_out_of_domain_id": high_id, "highest_out_of_domain_question": text[high_id],
        "highest_out_of_domain_top_score": highest_ood,
        "in_domain_missing_scores": missing,
        "falsely_refused_at_current_threshold": [
            {"id": i, "question": text[i], "score": sc} for i, sc in ans.items()
            if sc < config.MIN_RERANK_SCORE],
        "suggested_min_rerank_score": suggested,
    }


# ------------------------------------------------------- 2. end-to-end eval
def judge_faithful(bot, question, answer, top_chunks):
    """LLM-as-judge: is every claim in the answer supported by the retrieved context?"""
    clean = answer.replace(config.SAFETY_NOTE, "").strip()
    system = ("You are a strict fact-checker. You get CONTEXT passages, a QUESTION and an ANSWER. "
              "Reply with exactly one word: SUPPORTED if every fact, number and recommendation in the "
              "ANSWER is stated in the CONTEXT, otherwise UNSUPPORTED.")
    user = f"CONTEXT:\n{build_context(top_chunks)}\n\nQUESTION: {question}\n\nANSWER:\n{clean}"
    resp = bot.generator.client.chat.completions.create(
        model=config.GROQ_MODEL, temperature=0, max_tokens=800,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        extra_body={"reasoning_effort": config.REASONING_EFFORT},
    )
    verdict = (resp.choices[0].message.content or "").upper()
    if "UNSUPPORTED" in verdict:
        return False
    if "SUPPORTED" in verdict:
        return True
    return None


def full_eval(bot, questions, pause):
    rows = []
    for n, q in enumerate(questions, start=1):
        text = q["question"]
        print(f"  [{n}/{len(questions)}] {text[:70]}")
        res = with_retry(lambda: bot.ask(text))
        row = {"id": q["id"], "question": text, "answerable": q["answerable"],
               "found": res["found"], "answer": res["answer"],
               "sources": [f"{s['title']} p.{s['page']}" for s in res["sources"]]}

        if q["answerable"]:
            row["keywords_ok"] = res["found"] and all(re.search(k, res["answer"], re.I) for k in q["keywords"])
            row["has_citations"] = bool(res["sources"])
            row["faithful"] = None
            if res["found"]:
                cands = bot.retriever.search(text)
                top = bot.reranker.rerank(text, cands)
                row["faithful"] = with_retry(lambda: judge_faithful(bot, text, res["answer"], top))
        else:
            row["correctly_refused"] = not res["found"]
        rows.append(row)
        time.sleep(pause)

    ans = [r for r in rows if r["answerable"]]
    un = [r for r in rows if not r["answerable"]]
    judged = [r for r in ans if r["faithful"] is not None]
    summary = {
        "answerable_answered": sum(r["found"] for r in ans) / len(ans),
        "keyword_accuracy": sum(bool(r["keywords_ok"]) for r in ans) / len(ans),
        "citation_rate": sum(r["has_citations"] for r in ans if r["found"]) / max(1, sum(r["found"] for r in ans)),
        "faithfulness": (sum(1 for r in judged if r["faithful"]) / len(judged)) if judged else None,
        "correct_refusals": sum(r["correctly_refused"] for r in un) / len(un),
    }
    return summary, rows


# ------------------------------------------------------------------ report
def pct(x):
    return "n/a" if x is None else f"{x * 100:.0f}%"


def build_report(n_ans, n_un, metrics, misses, thr, e2e):
    L = ["# AgriGuide evaluation\n",
         f"{n_ans} answerable questions and {n_un} questions that should be refused, "
         f"written from the indexed PDFs (expected source + page for each).\n",
         "## Retrieval quality (higher is better)\n",
         "| Method | Hit@1 | Hit@3 | Hit@5 | MRR |", "|---|---|---|---|---|"]
    for m in MODES:
        s = metrics[m]
        L.append(f"| {MODE_LABELS[m]} | {pct(s['hit@1'])} | {pct(s['hit@3'])} | {pct(s['hit@5'])} | {s['mrr']:.3f} |")
    L += ["", "Hit@k = the correct source page appears in the top k chunks. "
          "MRR = mean reciprocal rank of the first correct chunk.\n"]

    L += ["## Refusal threshold\n",
          f"- Lowest top rerank score on an answerable question: **{thr['lowest_answerable_top_score']:.3f}** "
          f"({thr['lowest_answerable_id']}: {thr['lowest_answerable_question']})",
          f"- Highest top rerank score on an out-of-domain question: **{thr['highest_out_of_domain_top_score']:.3f}** "
          f"({thr['highest_out_of_domain_id']}: {thr['highest_out_of_domain_question']})"]
    for qid, sc in thr["in_domain_missing_scores"].items():
        L.append(f"- In-domain but missing from the documents ({qid}): **{sc:.3f}** "
                 "(a score threshold cannot catch this; the LLM `NOT_FOUND` check does)")
    bad = thr["falsely_refused_at_current_threshold"]
    if bad:
        L.append(f"- Real questions that the current threshold ({config.MIN_RERANK_SCORE}) would wrongly refuse:")
        for b in bad:
            L.append(f"  - {b['id']} (score {b['score']:.3f}): {b['question']}")
    note = "" if thr["separable"] else (" - scores overlap, so the threshold only filters junk "
                                        "retrieval and the LLM `NOT_FOUND` check does the rest")
    L.append(f"- Suggested `MIN_RERANK_SCORE`: **{thr['suggested_min_rerank_score']}** "
             f"(currently {config.MIN_RERANK_SCORE}){note}\n")

    if e2e:
        L += ["## End-to-end answers\n", "| Metric | Score |", "|---|---|",
              f"| Answerable questions answered | {pct(e2e['answerable_answered'])} |",
              f"| Answers containing the expected key facts | {pct(e2e['keyword_accuracy'])} |",
              f"| Answers with source citations | {pct(e2e['citation_rate'])} |",
              f"| Faithfulness (LLM judge: claims supported by context) | {pct(e2e['faithfulness'])} |",
              f"| Correct refusals on unanswerable questions | {pct(e2e['correct_refusals'])} |\n"]

    if misses:
        L += ["## Questions where the correct page was not in the top 5 (hybrid + rerank)\n"]
        for m in misses:
            L.append(f"- **{m['id']}** {m['question']} - correct rank: {m['rank']}; top results: {', '.join(m['top3'])}")
        L.append("")
    return "\n".join(L)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description="Evaluate AgriGuide")
    ap.add_argument("--full", action="store_true", help="also run end-to-end answers (uses Groq)")
    ap.add_argument("--pause", type=float, default=1.0, help="seconds between Groq calls in --full mode")
    args = ap.parse_args()

    questions = load_questions()
    n_ans = sum(q["answerable"] for q in questions)
    n_un = len(questions) - n_ans

    if args.full:
        from .pipeline import AgriGuide
        bot = AgriGuide()
        retriever, reranker = bot.retriever, bot.reranker
    else:
        from .reranker import Reranker
        bot, retriever, reranker = None, HybridRetriever(), Reranker()

    print(f"\n== Retrieval evaluation ({len(questions)} questions) ==")
    metrics, misses, top_scores = retrieval_eval(retriever, reranker, questions)
    thr = threshold_analysis(questions, top_scores)

    e2e, rows = None, None
    if args.full:
        print("\n== End-to-end evaluation (Groq) ==")
        e2e, rows = full_eval(bot, questions, args.pause)

    report = build_report(n_ans, n_un, metrics, misses, thr, e2e)
    print("\n" + report)
    EVAL_DIR.mkdir(exist_ok=True)
    (EVAL_DIR / "results.md").write_text(report, encoding="utf-8")
    (EVAL_DIR / "results.json").write_text(
        json.dumps({"retrieval": metrics, "threshold": thr, "top_scores": top_scores, "misses": misses,
                    "end_to_end": e2e, "rows": rows}, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\nSaved eval/results.md and eval/results.json")


if __name__ == "__main__":
    main()
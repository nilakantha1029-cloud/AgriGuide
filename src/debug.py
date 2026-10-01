"""Diagnose why a question fails.

Usage:  python -m src.debug "What fertilizer is recommended for wheat cultivation?"
        python -m src.debug "wheat fertilizer" --keyword wheat
"""
import argparse
from collections import Counter

from . import config
from .generator import Generator, build_context
from .reranker import Reranker
from .retriever import HybridRetriever


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("question")
    ap.add_argument("--keyword", help="Count chunks containing this word (default: longest word in question)")
    ap.add_argument("--category")
    args = ap.parse_args()

    r = HybridRetriever()
    print("\n=== 1. INDEX CONTENTS ===")
    per_file = Counter(c["file"] for c in r.chunks)
    for f, n in sorted(per_file.items()):
        print(f"  {n:6d} chunks  {f}")
    print(f"  TOTAL: {len(r.chunks)} chunks")

    kw = (args.keyword or max(args.question.replace("?", "").split(), key=len)).lower()
    hits = [c for c in r.chunks if kw in c["text"].lower()]
    print(f"\n=== 2. KEYWORD CHECK: chunks containing '{kw}': {len(hits)} ===")
    for c in hits[:5]:
        print(f"  - {c['source']} p.{c['page']}: {c['text'][:110].replace(chr(10), ' ')}...")
    if not hits:
        print("  -> The word never appears in your indexed text. Either the PDFs don't cover it,")
        print("     or they are scanned/garbled (check ingest warnings).")

    print("\n=== 3. HYBRID TOP 10 (before rerank) ===")
    cands = r.search(args.question, category=args.category)
    for c in cands[:10]:
        print(f"  rrf={c['rrf_score']:.4f}  {c['source']} p.{c['page']}  {c['text'][:80].replace(chr(10), ' ')}...")

    print("\n=== 4. RERANKED TOP 5 ===")
    top = Reranker().rerank(args.question, cands)
    for c in top:
        print(f"  score={c['rerank_score']:.4f}  {c['source']} p.{c['page']}")
    best = top[0]["rerank_score"] if top else 0
    print(f"  threshold MIN_RERANK_SCORE = {config.MIN_RERANK_SCORE} -> "
          f"{'PASS' if best >= config.MIN_RERANK_SCORE else 'FAIL (this is why it refused)'}")

    if top:
        print("\n=== 5. CONTEXT SENT TO LLM (first 1500 chars) ===")
        print(build_context(top)[:1500])
        print("\n=== 6. RAW LLM ANSWER ===")
        print(Generator().generate(args.question, top))


if __name__ == "__main__":
    main()
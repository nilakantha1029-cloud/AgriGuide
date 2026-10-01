"""Command-line interface.

Usage:
  python -m src.cli "What fertilizer is recommended for wheat?"
  python -m src.cli "Control of aphids in mustard" --category diseases
"""
import argparse
import sys

from .pipeline import AgriGuide


def format_output(result: dict) -> str:
    lines = ["Answer", "─" * 14, result["answer"]]
    if result["sources"]:
        lines += ["", "Sources"]
        for s in result["sources"]:
            lines.append(f"[{s['ref']}] {s['title']} - Page {s['page']}  ({s['category']})")
    if not result.get("found", True) and result.get("reason"):
        ts = result.get("top_score")
        lines += ["", f"[debug] reason={result['reason']}, top_rerank_score={ts if ts is None else round(ts, 3)}"]
    return "\n".join(lines)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="AgriGuide - Agriculture Advisory RAG")
    parser.add_argument("question", nargs="?", help="Your question (omit for interactive mode)")
    parser.add_argument("--category", help="Restrict search to a data/ subfolder, e.g. soil")
    args = parser.parse_args()

    bot = AgriGuide()
    if args.question:
        print(format_output(bot.ask(args.question, args.category)))
        return

    print(f"AgriGuide ready. Categories: {bot.retriever.available_categories}. Ctrl+C to exit.")
    while True:
        try:
            q = input("\nQuestion: ").strip()
        except (KeyboardInterrupt, EOFError):
            break
        if q:
            print("\n" + format_output(bot.ask(q, args.category)))


if __name__ == "__main__":
    main()
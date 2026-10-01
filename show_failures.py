"""List answerable questions that failed end-to-end (needs eval/results.json from a --full run).

Run from the project root:  python show_failures.py
"""
import json
import sys

sys.stdout.reconfigure(encoding="utf-8")
data = json.load(open("eval/results.json", encoding="utf-8"))
if not data.get("rows"):
    raise SystemExit("No end-to-end rows found. Run:  python -m src.evaluate --full")

n = 0
for x in data["rows"]:
    if x["answerable"] and (not x["found"] or not x.get("keywords_ok") or x.get("faithful") is False):
        n += 1
        print(f"{x['id']} | refused: {not x['found']} | keywords ok: {x.get('keywords_ok')} | faithful: {x.get('faithful')}")
        print("  Q:", x["question"])
        print("  A:", x["answer"][:450].replace("\n", " "))
        print("  Sources:", ", ".join(x["sources"]) or "-")
        print()
print(f"{n} answerable questions need a look.")
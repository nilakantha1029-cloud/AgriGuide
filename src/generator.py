"""Answer generation with Groq, grounded in numbered context."""
import os
import re
import unicodedata

from groq import Groq

from . import config

SYSTEM_PROMPT = """You are AgriGuide, an agriculture advisory assistant for farmers and students.

Rules:
1. Answer ONLY using the numbered context passages provided. Do not use outside knowledge.
2. If the context does not contain the answer, reply exactly: "NOT_FOUND".
3. Cite the passages you use inline like [1] or [2][3] right after the relevant statement.
4. Keep the answer clear and practical. Use short bullet points for lists of nutrients, doses, steps or conditions.
5. Never invent numbers (doses, dates, quantities). Only state numbers that appear in the context.
6. If the context gives conditional recommendations (soil, variety, growth stage), mention those conditions.
7. Copy numbers and units exactly as written in the context (for example "200 g" must never become "200 ml"). Do not convert or round them.
8. If different passages give different values for the same thing (for example seed rate or spacing), list each value with its own citation and say the sources differ. Do not merge them into one range.
9. Do not add recommendations, timings or quantities that are not written in the cited passage.
10. If a passage describes more than one product, method or remedy, keep their effects strictly separate. Never attribute an effect of one product to a different product that happens to be mentioned nearby."""


def build_context(chunks):
    blocks = []
    for i, c in enumerate(chunks, start=1):
        blocks.append(f"[{i}] ({c['source']}, page {c['page']})\n{c['text']}")
    return "\n\n".join(blocks)

def normalize_citations(text: str) -> str:
    """Clean model output: any odd space/invisible char -> normal space, 【1】 -> [1]."""
    out = []
    for ch in text:
        cat = unicodedata.category(ch)
        if ch in "\u200b\u2060\ufeff" or cat in ("Zs", "Zl", "Zp"):
            out.append(" ")            # any kind of space -> plain space
        elif ch == "\u00ad" or cat == "Cf":
            continue                   # soft hyphen / other invisible format chars
        elif ch == "\u2011":
            out.append("-")            # non-breaking hyphen -> normal hyphen
        else:
            out.append(ch)
    text = re.sub(r" {2,}", " ", "".join(out))
    return re.sub(r"【(\d+(?:\s*,\s*\d+)*)(?:†[^】]*)?】", r"[\1]", text)


def cited_indices(answer: str):
    found = set()
    for group in re.findall(r"\[(\d+(?:\s*,\s*\d+)*)\]", answer):
        found.update(int(n) for n in re.split(r"\s*,\s*", group))
    return found


class Generator:
    def __init__(self):
        key = os.getenv("GROQ_API_KEY")
        if not key:
            raise RuntimeError("GROQ_API_KEY not set. Copy .env.example to .env and add your key.")
        self.client = Groq(api_key=key)

    def generate(self, question, chunks):
        user_msg = f"Context:\n{build_context(chunks)}\n\nQuestion: {question}"
        resp = self.client.chat.completions.create(
            model=config.GROQ_MODEL,
            temperature=config.TEMPERATURE,
            max_tokens=config.MAX_TOKENS,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            # Passed via extra_body so it works across groq SDK versions.
            extra_body={"reasoning_effort": config.REASONING_EFFORT},
        )
        choice = resp.choices[0]
        text = (choice.message.content or "").strip()
        if not text:
            raise RuntimeError(
                f"Model returned an empty answer (finish_reason={choice.finish_reason}). "
                "Try increasing MAX_TOKENS in src/config.py."
            )
        return normalize_citations(text)
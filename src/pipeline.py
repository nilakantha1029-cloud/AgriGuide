"""End-to-end RAG pipeline: retrieve -> rerank -> threshold -> generate -> cite."""
import re

from . import config
from .generator import Generator, cited_indices
from .reranker import Reranker
from .retriever import HybridRetriever

RISKY = re.compile(
    r"\b(dose|dosage|kg/ha|kg per|spray|pesticide|fungicide|insecticide|herbicide|"
    r"fertili[sz]er|urea|dap|npk|application)\b",
    re.I,
)


class AgriGuide:
    def __init__(self):
        self.retriever = HybridRetriever()
        self.reranker = Reranker()
        self.generator = Generator()

    def ask(self, question: str, category: str | None = None) -> dict:
        candidates = self.retriever.search(question, category=category)
        top = self.reranker.rerank(question, candidates)

        # Hallucination control #1: refuse if retrieval confidence is low.
        if not top or top[0]["rerank_score"] < config.MIN_RERANK_SCORE:
            return {"answer": config.NOT_FOUND_MESSAGE, "sources": [], "found": False,
                    "reason": "low_rerank_score",
                    "top_score": top[0]["rerank_score"] if top else None}

        answer = self.generator.generate(question, top)

        # Hallucination control #2: the model itself says context is insufficient.
        if "NOT_FOUND" in answer:
            return {"answer": config.NOT_FOUND_MESSAGE, "sources": [], "found": False,
                    "reason": "model_said_not_found", "top_score": top[0]["rerank_score"]}

        # Structured sources (built by code, not by the LLM).
        used = cited_indices(answer) or set(range(1, len(top) + 1))
        sources = []
        for i in sorted(used):
            if 1 <= i <= len(top):
                c = top[i - 1]
                sources.append({
                    "ref": i,
                    "title": c["source"],
                    "page": c["page"],
                    "category": c["category"],
                    "score": round(c["rerank_score"], 3),
                    "snippet": c["text"][:300],
                })

        if RISKY.search(question) or RISKY.search(answer):
            answer = f"{answer}\n\n{config.SAFETY_NOTE}"

        return {"answer": answer, "sources": sources, "found": True}
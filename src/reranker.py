"""Cross-encoder reranking of hybrid candidates."""
import numpy as np
from sentence_transformers import CrossEncoder

from . import config


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


class Reranker:
    def __init__(self):
        self.model = CrossEncoder(config.RERANK_MODEL)

    def rerank(self, query, candidates, top_k=config.RERANK_TOP_K):
        if not candidates:
            return []
        scores = np.asarray(
            self.model.predict([(query, c["text"]) for c in candidates]), dtype=float
        )
        # Depending on library version the output is logits or already 0..1.
        # Normalise so the threshold in config.MIN_RERANK_SCORE is consistent.
        if scores.min() < 0 or scores.max() > 1:
            scores = _sigmoid(scores)

        ranked = []
        for cand, s in zip(candidates, scores):
            item = dict(cand)
            item["rerank_score"] = float(s)
            ranked.append(item)
        ranked.sort(key=lambda c: c["rerank_score"], reverse=True)
        return ranked[:top_k]

"""Hybrid retrieval: FAISS (semantic) + BM25 (keyword), fused with RRF."""
import json
import re

import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from . import config


def tokenize(text: str):
    return re.findall(r"\w+", text.lower())


class HybridRetriever:
    def __init__(self):
        if not config.FAISS_PATH.exists():
            raise FileNotFoundError("Index not found. Run: python -m src.ingest")
        self.chunks = json.loads(config.CHUNKS_PATH.read_text(encoding="utf-8"))
        self.index = faiss.read_index(str(config.FAISS_PATH))
        self.embedder = SentenceTransformer(config.EMBED_MODEL)
        self.bm25 = BM25Okapi([tokenize(c["text"]) for c in self.chunks])
        self.categories = np.array([c["category"] for c in self.chunks])

    @property
    def available_categories(self):
        return sorted(set(self.categories.tolist()))

    # ---- individual retrievers -------------------------------------------------
    def dense_search(self, query, category=None, pool=config.DENSE_POOL):
        qv = self.embedder.encode(
            [config.QUERY_PREFIX + query], normalize_embeddings=True
        ).astype("float32")
        # With a category filter, search everything then filter (flat index is fast enough).
        n = len(self.chunks) if category else min(pool, len(self.chunks))
        _, ids = self.index.search(qv, n)
        ids = [int(i) for i in ids[0] if i != -1]
        if category:
            ids = [i for i in ids if self.categories[i] == category]
        return ids[:pool]

    def bm25_search(self, query, category=None, pool=config.BM25_POOL):
        scores = self.bm25.get_scores(tokenize(query))
        if category:
            scores = np.where(self.categories == category, scores, -np.inf)
        order = np.argsort(scores)[::-1][:pool]
        return [int(i) for i in order if scores[i] > 0]

    # ---- fusion -------------------------------------------------------------
    def search(self, query, top_k=config.HYBRID_TOP_K, category=None, mode="hybrid"):
        """mode: 'dense' | 'bm25' | 'hybrid' (modes make ablation/eval easy)."""
        if category and category not in self.available_categories:
            raise ValueError(f"Unknown category '{category}'. Options: {self.available_categories}")

        dense = self.dense_search(query, category) if mode in ("dense", "hybrid") else []
        sparse = self.bm25_search(query, category) if mode in ("bm25", "hybrid") else []

        fused = {}
        for ranking in (dense, sparse):
            for rank, cid in enumerate(ranking):
                fused[cid] = fused.get(cid, 0.0) + 1.0 / (config.RRF_K + rank + 1)

        top = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
        results = []
        for cid, score in top:
            item = dict(self.chunks[cid])
            item["rrf_score"] = score
            results.append(item)
        return results

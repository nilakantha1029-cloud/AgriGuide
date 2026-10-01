"""Central configuration for AgriGuide."""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
INDEX_DIR = ROOT / "index"
FAISS_PATH = INDEX_DIR / "faiss.index"
CHUNKS_PATH = INDEX_DIR / "chunks.json"

# Models
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "  # bge query instruction
RERANK_MODEL = "BAAI/bge-reranker-base"
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")

# Chunking
CHUNK_SIZE = 900
CHUNK_OVERLAP = 150
MIN_CHUNK_CHARS = 60

# Retrieval
DENSE_POOL = 50          # candidates pulled from FAISS
BM25_POOL = 50           # candidates pulled from BM25
RRF_K = 60               # Reciprocal Rank Fusion constant
HYBRID_TOP_K = 20        # candidates passed to the reranker
RERANK_TOP_K = 8         # chunks passed to the LLM

# Hallucination control: if the best reranked chunk scores below this
# (sigmoid-scaled, 0..1), the system refuses instead of guessing.
# Tune this using your evaluation set.
MIN_RERANK_SCORE = 0.03 

# Generation
TEMPERATURE = 0.1
# gpt-oss is a reasoning model: reasoning tokens count toward this limit,
# so keep it generous or the visible answer can get cut off.
MAX_TOKENS = 1500
REASONING_EFFORT = "medium"   # "low" | "medium" | "high" - low is plenty for grounded RAG answers

NOT_FOUND_MESSAGE = (
    "I couldn't find this in the provided agriculture documents. "
    "Please rephrase your question, or consult your local KVK / agriculture officer."
)
SAFETY_NOTE = (
    "Note: Doses and pesticide choices vary by soil, variety, season and region. "
    "Confirm with your local KVK or agriculture officer before applying."
)
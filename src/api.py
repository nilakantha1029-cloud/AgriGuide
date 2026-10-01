"""FastAPI backend for AgriGuide.

Run:  uvicorn src.api:app --reload
Then open http://127.0.0.1:8000 in a browser.
"""
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import config

bot = None  # built once at startup, reused across requests


@asynccontextmanager
async def lifespan(app: FastAPI):
    global bot
    from .pipeline import AgriGuide  # imported here so /api/health works even if
                                      # models/API key aren't ready yet
    print("Loading AgriGuide (embeddings, reranker, BM25 index)...")
    bot = AgriGuide()
    print(f"Ready. Categories: {bot.retriever.available_categories}")
    yield
    bot = None


app = FastAPI(title="AgriGuide API", lifespan=lifespan)

# Permissive CORS: this is a local single-user dev tool, and it also lets the
# frontend be opened directly as a file:// page during development if needed.
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    category: Optional[str] = None


class Source(BaseModel):
    ref: int
    title: str
    page: int
    category: str
    score: float
    snippet: str


class AskResponse(BaseModel):
    answer: str
    sources: list[Source]
    found: bool


@app.get("/api/health")
def health():
    return {"status": "ok", "ready": bot is not None}


@app.get("/api/categories")
def categories():
    if bot is None:
        raise HTTPException(503, "Still starting up - try again in a moment.")
    return {"categories": bot.retriever.available_categories}


@app.post("/api/ask", response_model=AskResponse)
def ask(req: AskRequest):
    if bot is None:
        raise HTTPException(503, "Still starting up - try again in a moment.")
    question = req.question.strip()
    if not question:
        raise HTTPException(400, "Question cannot be empty.")
    category = req.category or None
    if category and category not in bot.retriever.available_categories:
        raise HTTPException(400, f"Unknown category '{category}'.")
    try:
        result = bot.ask(question, category)
    except Exception as exc:  # e.g. Groq API error
        raise HTTPException(502, f"The answer service failed: {exc}") from None
    return result


# Serve the static frontend at "/". Mounted last so it doesn't shadow /api/*.
app.mount("/", StaticFiles(directory=str(config.ROOT / "frontend"), html=True), name="frontend")
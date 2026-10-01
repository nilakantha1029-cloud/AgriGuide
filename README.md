# 🌾 AgriGuide — Agriculture Advisory RAG

AgriGuide answers agriculture questions **only from a set of trusted PDFs** (ICAR advisories, crop handbooks, soil science, horticulture, etc.) and shows **page-level source citations** for every answer. If the documents don't contain the answer, it refuses instead of guessing.

- **Backend:** FastAPI (Python)
- **Retrieval:** FAISS (semantic) + BM25 (keyword), fused with Reciprocal Rank Fusion, then re-ranked by a cross-encoder
- **LLM:** Groq API (default model `openai/gpt-oss-120b`)
- **Frontend:** plain HTML/CSS/JS served by FastAPI (no build step)
- **Deployment:** single Docker container (API + UI together)

---

## Table of contents

1. [Features](#1-features)
2. [Architecture](#2-architecture)
3. [Project structure](#3-project-structure)
4. [Knowledge base (PDFs)](#4-knowledge-base-pdfs)
5. [Prerequisites](#5-prerequisites)
6. [Get a Groq API key and configure `.env`](#6-get-a-groq-api-key-and-configure-env)
7. [Run locally (step by step)](#7-run-locally-step-by-step)
8. [Run with Docker (step by step)](#8-run-with-docker-step-by-step)
9. [Using the app](#9-using-the-app)
10. [API reference](#10-api-reference)
11. [Configuration reference](#11-configuration-reference)
12. [Adding or changing PDFs](#12-adding-or-changing-pdfs)
13. [Evaluation](#13-evaluation)
14. [Debugging a bad answer](#14-debugging-a-bad-answer)
15. [Troubleshooting](#15-troubleshooting)
16. [Security notes](#16-security-notes)
17. [Known limitations](#17-known-limitations)

---

## 1. Features

- **Grounded answers** — the LLM is instructed to use only the retrieved passages and to cite them as `[1]`, `[2]`, …
- **Page-level citations** — sources (PDF title, page number, category, score, snippet) are built by code, not by the LLM.
- **Hybrid search** — semantic (BGE embeddings + FAISS) and keyword (BM25) results are merged with RRF.
- **Cross-encoder re-ranking** — `BAAI/bge-reranker-base` re-scores the top candidates.
- **Two hallucination guards**
  1. Refuse if the best re-rank score is below `MIN_RERANK_SCORE`.
  2. Refuse if the LLM replies `NOT_FOUND`.
- **Safety note** — questions/answers about doses, pesticides, fertilizers, etc. get an extra "confirm with your local KVK / agriculture officer" note.
- **Topic filter** — restrict search to one category (folder under `data/`).
- **State-aware chunking** for the ICAR Kharif advisory (each chunk is tagged with its state, e.g. `[Punjab] …`).
- **Web UI, REST API and CLI**.
- **Built-in evaluation** — Hit@k / MRR for each retrieval method, refusal-threshold analysis, and LLM-judged faithfulness.

---

## 2. Architecture

### 2.1 Ingestion pipeline (run once, or whenever PDFs change)

```
data/<category>/*.pdf
        │
        ▼
 PyMuPDF text extraction (page by page)  ──►  clean()  (soft hyphens, line-break hyphens, whitespace)
        │
        ▼
 RecursiveCharacterTextSplitter  (chunk size 900, overlap 150, min 60 chars)
        │        └─ ICAR advisory: chunks are prefixed with the current state name
        ▼
 BAAI/bge-small-en-v1.5 embeddings (normalized, 384-dim)
        │
        ▼
 FAISS IndexFlatIP (cosine similarity)  ──►  index/faiss.index
 Chunk metadata (text, source, file, page, category, state)  ──►  index/chunks.json
```

### 2.2 Query pipeline (every question)

```
Question (+ optional category)
   │
   ├──► Dense search:  FAISS, top 50   (query gets the BGE instruction prefix)
   ├──► Sparse search: BM25,  top 50
   │
   ▼
Reciprocal Rank Fusion (k = 60)  ──►  top 20 candidates
   │
   ▼
Cross-encoder rerank (BAAI/bge-reranker-base, sigmoid-scaled 0..1)  ──►  top 8 chunks
   │
   ├── best score < MIN_RERANK_SCORE ?  ──►  "I couldn't find this…"  (refusal #1)
   │
   ▼
Groq LLM (temperature 0.1, strict system prompt, numbered context)
   │
   ├── answer contains "NOT_FOUND" ?  ──►  "I couldn't find this…"  (refusal #2)
   │
   ▼
Extract cited [n] → build structured sources → append safety note if risky topic
   │
   ▼
{ answer, sources[], found }
```

### 2.3 Runtime components

```
┌────────────────────────────── Browser ──────────────────────────────┐
│  frontend/index.html + style.css + app.js  (calls /api/*)           │
└───────────────────────────────┬─────────────────────────────────────┘
                                │ HTTP
┌───────────────────────────────▼─────────────────────────────────────┐
│ FastAPI  (src/api.py)  — serves /api/* and the static frontend at / │
│   └─ AgriGuide pipeline (src/pipeline.py), loaded once at startup   │
│        ├─ HybridRetriever  (src/retriever.py)  FAISS + BM25         │
│        ├─ Reranker         (src/reranker.py)   cross-encoder        │
│        └─ Generator        (src/generator.py)  Groq client          │
└───────────────────────────────┬─────────────────────────────────────┘
                                │ HTTPS (only external call at runtime)
                         ┌──────▼──────┐
                         │  Groq API   │
                         └─────────────┘
```

Embedding and re-ranking models run **locally on CPU**; only the final answer generation calls Groq.

### 2.4 Source files

| File | Responsibility |
|---|---|
| `src/config.py` | All settings: models, chunking, retrieval sizes, thresholds, messages. Loads `.env`. |
| `src/ingest.py` | PDF → pages → chunks → embeddings → FAISS index + `chunks.json`. |
| `src/retriever.py` | `HybridRetriever`: dense, BM25 and RRF fusion, with optional category filter. |
| `src/reranker.py` | `Reranker`: cross-encoder scoring and sorting. |
| `src/generator.py` | System prompt, context building, Groq call, citation/character cleanup. |
| `src/pipeline.py` | `AgriGuide.ask()` — ties everything together with both refusal guards. |
| `src/api.py` | FastAPI app (`/api/health`, `/api/categories`, `/api/ask`) + static frontend. |
| `src/cli.py` | Command-line interface (single question or interactive). |
| `src/evaluate.py` | Retrieval and end-to-end evaluation. |
| `src/debug.py` | Step-by-step diagnosis of one question. |
| `show_failures.py` | Prints failed questions from `eval/results.json`. |

---

## 3. Project structure

```
agriguide/
├── data/                       # Source PDFs — the folder name = category
│   ├── agro_advisory/          #   ICAR-En-Kharif-Agro-Advisories-for-Farmers-2025.pdf
│   ├── crops/                  #   2024-Vegetable-Crop-Handbook.pdf, Field-Crops.pdf
│   ├── diseases/               #   natural pest and disease control.pdf
│   ├── general/                #   farmerbook.pdf
│   ├── horticulture/           #   Horticulture.pdf
│   └── soil/                   #   Soil-Science.pdf
├── index/                      # GENERATED by `python -m src.ingest`
│   ├── .gitkeep
│   ├── faiss.index
│   └── chunks.json
├── src/                        # Python package (see table above)
├── frontend/                   # index.html, style.css, app.js
├── eval/
│   ├── questions.json          # 38 evaluation questions (33 answerable + 5 to refuse)
│   ├── results.md              # last evaluation report
│   └── results.json            # last evaluation raw data
├── Dockerfile
├── docker-compose.yml
├── .dockerignore
├── requirements.txt
├── .env.example                # template — copy to .env
├── .env                        # YOUR secrets (never commit)
├── .gitignore
├── show_failures.py
└── README.md
```

---

## 4. Knowledge base (PDFs)

The current index is built from **7 PDFs → 4,658 chunks**:

| Category (folder) | PDF | Chunks |
|---|---|---:|
| `agro_advisory` | ICAR Kharif Agro-Advisories for Farmers 2025 | 1,158 |
| `crops` | 2024 Vegetable Crop Handbook | 2,403 |
| `crops` | Field Crops | 41 |
| `diseases` | Natural pest and disease control | 34 |
| `general` | Farmer book | 427 |
| `horticulture` | Horticulture | 168 |
| `soil` | Soil Science | 427 |

> The category shown in the UI/API is simply the **name of the sub-folder** under `data/`. A PDF placed directly in `data/` gets the category `general`.
>
> Make sure you have the right to redistribute the PDFs if you publish the repository publicly.

---

## 5. Prerequisites

| Requirement | Local run | Docker run |
|---|:-:|:-:|
| Python **3.10 or newer** (3.11 recommended; the code uses `str \| None` syntax) | ✅ | — (image uses 3.11) |
| Docker Desktop / Docker Engine + Compose v2 | — | ✅ |
| A free **Groq API key** | ✅ | ✅ |
| Internet on first run (downloads models from Hugging Face) | ✅ | ✅ (at image build) |
| Disk space | ~3 GB (PyTorch + models) | ~3–4 GB for the image |
| RAM | 4 GB minimum, 8 GB recommended | same |

No GPU is needed — everything runs on CPU.

---

## 6. Get a Groq API key and configure `.env`

The **only secret** the project needs is `GROQ_API_KEY`.

1. Go to <https://console.groq.com/keys> and sign in (free account).
2. Click **Create API Key**, give it a name, and copy the key (it starts with `gsk_`). It is shown only once.
3. In the project root, create your `.env` from the template:

   **macOS / Linux / Git Bash**
   ```bash
   cp .env.example .env
   ```
   **Windows PowerShell**
   ```powershell
   Copy-Item .env.example .env
   ```
4. Open `.env` and paste your key:

   ```env
   # Required
   GROQ_API_KEY=gsk_your_real_key_here

   # Optional — defaults to openai/gpt-oss-120b
   # GROQ_MODEL=openai/gpt-oss-120b
   ```

Rules for the `.env` file:

- No quotes and no spaces around `=` (`GROQ_API_KEY=gsk_abc`, not `GROQ_API_KEY = "gsk_abc"`).
- Keep it in the **project root** (next to `Dockerfile`).
- It is listed in `.gitignore` and `.dockerignore`, so it is never committed or baked into the image. **Never share the key or commit it.**

| Variable | Required | Default | Purpose |
|---|:-:|---|---|
| `GROQ_API_KEY` | ✅ | — | Authenticates calls to Groq. |
| `GROQ_MODEL` | ❌ | `openai/gpt-oss-120b` | Groq model used to write answers. |

---

## 7. Run locally (step by step)

### Step 1 — Get the project

```bash
git clone <your-repo-url> agriguide
cd agriguide
```
(or unzip the project and `cd` into the folder).

### Step 2 — Create a virtual environment

**macOS / Linux**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Windows (PowerShell)**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```
If PowerShell blocks the script: `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`.

**Windows (cmd)**
```cmd
python -m venv .venv
.venv\Scripts\activate.bat
```

### Step 3 — Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

Optional (Linux/Windows, saves ~1–2 GB by installing CPU-only PyTorch first):
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

### Step 4 — Configure `.env`

Follow [section 6](#6-get-a-groq-api-key-and-configure-env) and put your `GROQ_API_KEY` in `.env`.

### Step 5 — Check that the PDFs are in place

```
data/<category>/<file>.pdf
```

### Step 6 — Build the search index

```bash
python -m src.ingest
```

What happens: each PDF is read, split into chunks, embedded with `BAAI/bge-small-en-v1.5` (downloaded from Hugging Face on first run), and saved to `index/faiss.index` and `index/chunks.json`.

Expected output (abridged):
```
[OK] agro_advisory/ICAR-En-Kharif-Agro-Advisories-for-Farmers-2025.pdf: ... pages -> 1158 chunks (category=agro_advisory)
...
Total chunks: 4658
Saved index -> .../index/faiss.index
```
This takes a few minutes on CPU. You only repeat it when the PDFs or chunking settings change.

> If you cloned a repo that already includes an `index/` folder you can skip this step, but it is safer to rebuild it on your machine. Note that `.gitignore` excludes the index by default.

### Step 7 — Start the web app

```bash
uvicorn src.api:app --reload
```

On startup it loads the embedding model, the re-ranker (downloaded on first run, ~1 GB) and the BM25 index. Wait for:

```
Loading AgriGuide (embeddings, reranker, BM25 index)...
Ready. Categories: ['agro_advisory', 'crops', 'diseases', 'general', 'horticulture', 'soil']
```

### Step 8 — Open it

- Web UI: <http://127.0.0.1:8000>
- Interactive API docs (Swagger): <http://127.0.0.1:8000/docs>
- Health check: <http://127.0.0.1:8000/api/health>

To make it reachable from other devices on your network: `uvicorn src.api:app --host 0.0.0.0 --port 8000` (drop `--reload` for normal use).

### Optional — Use the command line instead

```bash
# one question
python -m src.cli "What fertilizer is recommended for wheat cultivation?"

# restrict to a category
python -m src.cli "leaf blight in rice" --category diseases

# interactive mode (Ctrl+C to quit)
python -m src.cli
```

---

## 8. Run with Docker (step by step)

The Dockerfile creates **one image** that contains the API, the frontend, the pre-built search index and the pre-downloaded models. At runtime the container only needs internet access to reach Groq.

### Step 1 — Install Docker

Install Docker Desktop (Windows/macOS) or Docker Engine + Compose plugin (Linux), then verify:
```bash
docker --version
docker compose version
```

### Step 2 — Get the project and create `.env`

```bash
cd agriguide
cp .env.example .env        # Windows PowerShell: Copy-Item .env.example .env
# edit .env and set GROQ_API_KEY=gsk_...
```

### Step 3 — Make sure the ignore file is named `.dockerignore`

Docker only reads a file named **`.dockerignore`** (with the leading dot). If your copy is named `dockerignore` (no dot), rename it, otherwise `.env`, the local virtual environment and other junk get sent to the build context:

```bash
# macOS / Linux / Git Bash
mv dockerignore .dockerignore
```
```powershell
# Windows PowerShell
Rename-Item dockerignore .dockerignore
```

### Step 4A — Run with Docker Compose (recommended)

```bash
docker compose up --build
```

Run in the background: `docker compose up --build -d`

### Step 4B — Or run with plain Docker

```bash
docker build -t agriguide .
docker run -p 8000:8000 --env-file .env agriguide
```

### What the build does (and why the first build is slow)

| Layer | What happens | Rebuilt when |
|---|---|---|
| 1 | `pip install` with the CPU-only PyTorch wheel index | `requirements.txt` changes |
| 2 | Copies `data/` + ingestion code, runs `python -m src.ingest` (builds the index, downloads the embedding model) and downloads the re-ranker model | PDFs, `config.py` or `ingest.py` change |
| 3 | Copies the rest of `src/` and `frontend/` | any app code change (fast) |

Expect **10–20 minutes** the first time. Later builds reuse the cache.

### Step 5 — Open it

- Web UI: <http://localhost:8000>
- Health: <http://localhost:8000/api/health> → `{"status":"ok","ready":true}`

The container has a built-in health check (`/api/health`, every 30 s, 60 s start period). It reports `ready: true` once the models finish loading (roughly 10–60 s after start).

### Everyday Docker commands

```bash
docker compose logs -f agriguide      # follow logs
docker compose ps                     # status / health
docker compose stop                   # stop
docker compose down                   # stop and remove container
docker compose up --build             # rebuild after code/PDF changes
```

### Iterating on PDFs without rebuilding the image

In `docker-compose.yml`, uncomment:
```yaml
    volumes:
      - ./data:/app/data
      - ./index:/app/index
```
Then rebuild the index inside the running container and restart:
```bash
docker compose exec agriguide python -m src.ingest
docker compose restart agriguide
```

### Changing the port

Edit `docker-compose.yml` (`"8080:8000"` serves the app on <http://localhost:8080>) or use `docker run -p 8080:8000 ...`.

---

## 9. Using the app

1. Open the UI.
2. (Optional) choose a **Topic** on the left to search only one category.
3. Type a question or click one of the **Try asking** examples, e.g.
   - *How should nitrogen be applied in rice cultivation?*
   - *How do I control cutworms without pesticides?*
   - *What is the planting distance for mango?*
   - *What is cation exchange capacity?*
4. Read the answer; inline `[n]` markers link to the **Sources** (title, page, category, relevance score, snippet).
5. If the documents don't cover the question you get: *"I couldn't find this in the provided agriculture documents…"*

---

## 10. API reference

Base URL: `http://localhost:8000`

### `GET /api/health`
```json
{ "status": "ok", "ready": true }
```
`ready` is `false` while models are still loading.

### `GET /api/categories`
```json
{ "categories": ["agro_advisory", "crops", "diseases", "general", "horticulture", "soil"] }
```
Returns `503` while starting up.

### `POST /api/ask`

Request:
```json
{ "question": "What is cation exchange capacity?", "category": "soil" }
```
- `question`: required, 1–2000 characters.
- `category`: optional; must be one of `/api/categories`.

Response:
```json
{
  "answer": "Cation exchange capacity (CEC) is … [1]",
  "found": true,
  "sources": [
    {
      "ref": 1,
      "title": "Soil Science",
      "page": 152,
      "category": "soil",
      "score": 0.987,
      "snippet": "…first 300 characters of the chunk…"
    }
  ]
}
```

Status codes: `200` OK · `400` empty question / unknown category · `422` invalid body · `502` answer service failed (e.g. Groq error, bad API key, rate limit) · `503` still starting.

Example:
```bash
curl -X POST http://localhost:8000/api/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"What is the planting distance for mango?"}'
```

---

## 11. Configuration reference

All tunables live in `src/config.py`.

| Setting | Default | Meaning |
|---|---|---|
| `EMBED_MODEL` | `BAAI/bge-small-en-v1.5` | Embedding model (384-dim). |
| `RERANK_MODEL` | `BAAI/bge-reranker-base` | Cross-encoder re-ranker. |
| `GROQ_MODEL` | `openai/gpt-oss-120b` | Overridable with the `GROQ_MODEL` env var. |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | 900 / 150 | Characters per chunk / overlap. |
| `MIN_CHUNK_CHARS` | 60 | Smaller chunks are dropped. |
| `DENSE_POOL` / `BM25_POOL` | 50 / 50 | Candidates from each retriever. |
| `RRF_K` | 60 | Reciprocal Rank Fusion constant. |
| `HYBRID_TOP_K` | 20 | Candidates sent to the re-ranker. |
| `RERANK_TOP_K` | 8 | Chunks sent to the LLM. |
| `MIN_RERANK_SCORE` | 0.03 | Refuse if the best re-rank score is below this (0–1). |
| `TEMPERATURE` | 0.1 | LLM randomness. |
| `MAX_TOKENS` | 1500 | Output limit (reasoning tokens count too). |
| `REASONING_EFFORT` | `medium` | `low` / `medium` / `high`. |

> After changing `CHUNK_SIZE`, `CHUNK_OVERLAP`, `MIN_CHUNK_CHARS` or `EMBED_MODEL`, **re-run `python -m src.ingest`** (or rebuild the Docker image).

---

## 12. Adding or changing PDFs

1. Put the PDF in `data/<category>/` (create a new folder for a new category).
2. Rebuild the index:
   - Local: `python -m src.ingest`, then restart `uvicorn`.
   - Docker: `docker compose up --build` (or use the volume approach in section 8).
3. Scanned PDFs (images only) print a `[WARN] … no text` message during ingestion. Run OCR first:
   ```bash
   ocrmypdf input.pdf output.pdf
   ```
4. Re-run the evaluation (section 13) if you changed the corpus.

The special state-tagging logic applies only to the `agro_advisory` folder (ICAR state-wise headings).

---

## 13. Evaluation

`eval/questions.json` holds 38 questions: 33 answerable (with expected source + page and expected keywords) and 5 that must be refused (out-of-domain or missing from the PDFs).

```bash
# Retrieval metrics + threshold analysis (no Groq calls)
python -m src.evaluate

# Also run end-to-end answers, refusals, keyword checks and LLM-judged faithfulness (uses Groq)
python -m src.evaluate --full
python -m src.evaluate --full --pause 2     # slower, if you hit rate limits

# List answerable questions that failed (needs a --full run)
python show_failures.py
```

Results are written to `eval/results.md` and `eval/results.json`.

### Latest results (from `eval/results.md`)

**Retrieval quality**

| Method | Hit@1 | Hit@3 | Hit@5 | MRR |
|---|---|---|---|---|
| Vector only (FAISS) | 91% | 100% | 100% | 0.955 |
| Keyword only (BM25) | 73% | 91% | 97% | 0.823 |
| Hybrid (FAISS + BM25, RRF) | 94% | 97% | 100% | 0.961 |
| Hybrid + rerank | 88% | 97% | 100% | 0.927 |

**End-to-end**

| Metric | Score |
|---|---|
| Answerable questions answered | 100% |
| Answers containing the expected key facts | 100% |
| Answers with source citations | 100% |
| Faithfulness (LLM judge) | 94% |
| Correct refusals on unanswerable questions | 100% |

**Refusal threshold:** scores of real questions and out-of-domain questions overlap (lowest answerable 0.459 vs highest out-of-domain 0.506), so the score threshold only filters junk retrieval and the LLM's `NOT_FOUND` check does the rest. The evaluation suggests `MIN_RERANK_SCORE = 0.05` (currently `0.03`).

---

## 14. Debugging a bad answer

```bash
python -m src.debug "What fertilizer is recommended for wheat cultivation?"
python -m src.debug "wheat fertilizer" --keyword wheat --category crops
```

It prints, step by step: index contents per file, keyword presence in the indexed text, hybrid top 10, re-ranked top 5 with the threshold PASS/FAIL, the exact context sent to the LLM, and the raw LLM answer.

---

## 15. Troubleshooting

| Problem | Cause / fix |
|---|---|
| `FileNotFoundError: Index not found. Run: python -m src.ingest` | The index hasn't been built. Run `python -m src.ingest`. |
| `RuntimeError: GROQ_API_KEY not set` | `.env` is missing/empty or not in the project root. Docker: pass `--env-file .env` (Compose does it automatically). |
| UI shows an error / API returns `502` | Groq call failed: invalid key, rate limit, wrong `GROQ_MODEL`, or no internet. Check logs. |
| API returns `503 Still starting up` | Models are still loading; wait ~10–60 s and check `/api/health`. |
| First start is very slow | Models download from Hugging Face on first run (local). In Docker they're baked into the image. |
| `Model returned an empty answer (finish_reason=length)` | Increase `MAX_TOKENS` in `src/config.py`. |
| `No PDFs found under …/data` | Put PDFs in `data/<category>/`. |
| `[WARN] … pages have no text` | Scanned PDF — run `ocrmypdf` first. |
| Docker build fails at the ingest step | Needs internet for Hugging Face downloads and enough RAM (≥ 4 GB in Docker Desktop settings). |
| Docker image is huge / build context is huge | `.dockerignore` is misnamed (`dockerignore`). Rename it as shown in section 8, step 3. |
| Port 8000 already in use | Use another port: `uvicorn src.api:app --port 8001` or change the mapping to `"8001:8000"`. |
| PowerShell won't activate the venv | `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` |
| Garbled characters in a redirected CLI output on Windows | Use `chcp 65001` or view the output in the UI; the CLI forces UTF-8 on stdout. |

---

## 16. Security notes

- **Never commit `.env`.** If a key ever leaks, revoke it at <https://console.groq.com/keys> and create a new one.
- CORS is `allow_origins=["*"]` in `src/api.py` — fine for local use, but restrict it to your own domain before exposing the API publicly.
- There is no authentication or rate limiting on `/api/ask`; anyone who can reach the server can spend your Groq quota. Put it behind a reverse proxy/login if you deploy it publicly.
- Do not use `--reload` in production.

---

## 17. Known limitations

- Answers are only as good as the PDFs; the app can't answer anything outside them.
- Scanned PDFs need OCR first; tables and figures in PDFs are extracted as plain text and can lose structure.
- Chunking is page-by-page, so a passage that spans two pages is split.
- No conversation memory — each question is answered independently.
- Faithfulness is ~94% on the internal evaluation set, so verify doses and pesticide advice with a local KVK / agriculture officer, as the app itself reminds users.
- The whole corpus is loaded in memory at startup (BM25 index over all chunks), and FAISS uses an exact flat index — fine for thousands of chunks, not for millions.

---

## License and credits

MIT License

Copyright (c) 2026 Nilakantha

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## Disclaimer

AgriGuide is an informational tool. Its answers come from the indexed documents and
may be incomplete or out of date. Always confirm pesticide, fertilizer and dosage
advice with your local KVK or agriculture officer before acting on it.
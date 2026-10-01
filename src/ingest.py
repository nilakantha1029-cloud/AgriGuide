"""Build the search index: PDFs -> pages -> chunks -> embeddings -> FAISS.

Run:  python -m src.ingest
"""
import json
import re
from pathlib import Path

import faiss
import fitz  # PyMuPDF
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

from . import config


def clean(text: str) -> str:
    text = text.replace("\u00ad", "")            # soft hyphens
    text = re.sub(r"-\n(?=[a-z])", "", text)     # re-join hyphenated line breaks
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# The ICAR Kharif advisory is organised as one heading per state/UT followed by many
# pages of content; the heading does not repeat on every page (it alternates with a
# generic running title), so a plain page-by-page chunker separates the state name
# from the content it labels. We track the current state as we walk the PDF and tag
# every chunk with it, so retrieval and citations stay correct even deep into a section.
ICAR_STATE_HEADERS = [
    "Andaman and Nicobar Islands", "Andhra Pradesh", "Arunachal Pradesh", "Assam",
    "Bihar", "Chhattisgarh", "Goa", "Gujarat", "Haryana & Delhi", "Himachal Pradesh",
    "Jammu and Kashmir", "Jharkhand", "Karnataka", "Kerala", "Ladakh", "Lakshadweep",
    "Madhya Pradesh", "Maharashtra", "Manipur", "Meghalaya", "Mizoram", "Nagaland",
    "Odisha", "Punjab", "Rajasthan", "Sikkim", "Tamil Nadu & Puducherry", "Telangana",
    "Tripura", "Uttar Pradesh", "Uttarakhand", "West Bengal",
]
_STATE_LOOKUP = {h.lower(): h for h in ICAR_STATE_HEADERS}


def detect_state_header(first_line: str):
    """Return the canonical state name if `first_line` is one of the ICAR section
    headings, else None. Matches exactly (case-insensitive) so it never misfires on
    body text or crop-name subheadings."""
    return _STATE_LOOKUP.get(first_line.strip().lower())


def title_from_path(path: Path) -> str:
    return path.stem.replace("_", " ").replace("-", " ").strip()


def load_pages(pdf_path: Path):
    """Yield (page_number, text). Page numbers are 1-based PDF pages."""
    with fitz.open(pdf_path) as doc:
        for i, page in enumerate(doc):
            yield i + 1, clean(page.get_text("text"))


def build_chunks():
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = []
    pdfs = sorted(config.DATA_DIR.rglob("*.pdf"))
    if not pdfs:
        raise SystemExit(f"No PDFs found under {config.DATA_DIR}. Add files to data/<category>/")

    for pdf in pdfs:
        rel = pdf.relative_to(config.DATA_DIR)
        category = rel.parts[0] if len(rel.parts) > 1 else "general"
        title = title_from_path(pdf)

        pages = list(load_pages(pdf))
        empty = sum(1 for _, t in pages if len(t) < 20)
        if pages and empty / len(pages) > 0.5:
            print(f"[WARN] {rel}: {empty}/{len(pages)} pages have no text. "
                  f"Likely a scanned PDF - run OCR first (e.g. ocrmypdf).")

        is_state_advisory = category == "agro_advisory"
        current_state = None
        added = 0
        for page_no, text in pages:
            if len(text) < config.MIN_CHUNK_CHARS:
                continue
            if is_state_advisory:
                first_line = text.split("\n", 1)[0]
                current_state = detect_state_header(first_line) or current_state
            for piece in splitter.split_text(text):
                if len(piece) < config.MIN_CHUNK_CHARS:
                    continue
                if current_state and not piece.lower().startswith(current_state.lower()):
                    piece = f"[{current_state}] {piece}"
                chunks.append({
                    "id": len(chunks),
                    "text": piece,
                    "source": title,
                    "file": str(rel),
                    "page": page_no,
                    "category": category,
                    "state": current_state,
                })
                added += 1
        print(f"[OK] {rel}: {len(pages)} pages -> {added} chunks (category={category})")
    return chunks


def main():
    chunks = build_chunks()
    print(f"\nTotal chunks: {len(chunks)}")

    print(f"Loading embedding model: {config.EMBED_MODEL}")
    model = SentenceTransformer(config.EMBED_MODEL)
    vectors = model.encode(
        [c["text"] for c in chunks],
        batch_size=32,
        normalize_embeddings=True,
        show_progress_bar=True,
    ).astype("float32")

    index = faiss.IndexFlatIP(vectors.shape[1])  # cosine similarity (vectors are normalized)
    index.add(vectors)

    config.INDEX_DIR.mkdir(exist_ok=True)
    faiss.write_index(index, str(config.FAISS_PATH))
    config.CHUNKS_PATH.write_text(json.dumps(chunks, ensure_ascii=False), encoding="utf-8")
    print(f"Saved index -> {config.FAISS_PATH}\nSaved chunks -> {config.CHUNKS_PATH}")


if __name__ == "__main__":
    main()
"""
build_index.py

Offline pipeline: reads PDFs from data/, extracts text, chunks it,
generates embeddings, builds a FAISS index, and saves the index plus
chunk metadata to vectorstore/.

Run this once (or whenever the PDFs in data/ change):
    python build_index.py
"""

import pickle
from pathlib import Path
from typing import List, Tuple

import faiss
import pdfplumber
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

DATA_DIR = Path("data")
VECTORSTORE_DIR = Path("vectorstore")
INDEX_PATH = VECTORSTORE_DIR / "index.faiss"
METADATA_PATH = VECTORSTORE_DIR / "metadata.pkl"

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100


def extract_pages(pdf_path: Path) -> List[Tuple[str, int, str]]:
    """Extract (filename, page_number, text) for every non-empty page of a PDF."""
    pages = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page_number, page in enumerate(pdf.pages, start=1):
                text = (page.extract_text() or "").strip()
                if text:
                    pages.append((pdf_path.name, page_number, text))
    except Exception as err:
        print(f"Warning: could not read '{pdf_path.name}' ({err}). Skipping.")
    return pages


def load_all_pages() -> List[Tuple[str, int, str]]:
    """Read every PDF in data/ and return a flat list of page-level text."""
    pdf_files = sorted(DATA_DIR.glob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(
            f"No PDF files found in '{DATA_DIR}/'. Add at least 5 official "
            "IIT Bombay academic PDFs before building the index."
        )

    all_pages = []
    for pdf_path in pdf_files:
        print(f"Parsing {pdf_path.name} ...")
        all_pages.extend(extract_pages(pdf_path))

    if not all_pages:
        raise ValueError("No extractable text found in any PDF. Files may be scanned images.")

    return all_pages


def chunk_pages(pages: List[Tuple[str, int, str]]) -> List[dict]:
    """Split page text into overlapping chunks, tagged with source metadata."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = []
    for filename, page_number, text in pages:
        for chunk_text in splitter.split_text(text):
            chunks.append({"filename": filename, "page": page_number, "text": chunk_text})
    return chunks


def build_faiss_index(chunks: List[dict]) -> faiss.Index:
    """Embed all chunks and build a cosine-similarity FAISS index."""
    print(f"Loading embedding model: {EMBEDDING_MODEL_NAME} ...")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    texts = [c["text"] for c in chunks]
    embeddings = model.encode(
        texts, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=True
    ).astype("float32")

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return index


def save_index(index: faiss.Index, chunks: List[dict]) -> None:
    """Persist the FAISS index and chunk metadata to vectorstore/."""
    VECTORSTORE_DIR.mkdir(exist_ok=True)
    faiss.write_index(index, str(INDEX_PATH))
    with open(METADATA_PATH, "wb") as f:
        pickle.dump(chunks, f)
    print(f"Saved {index.ntotal} vectors to '{INDEX_PATH}' and metadata to '{METADATA_PATH}'.")


def main() -> None:
    pages = load_all_pages()
    chunks = chunk_pages(pages)
    print(f"Created {len(chunks)} chunks from {len(pages)} pages.")
    index = build_faiss_index(chunks)
    save_index(index, chunks)
    print("Index build complete.")


if __name__ == "__main__":
    main()

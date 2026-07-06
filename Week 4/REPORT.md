# IITB Academic Assistant — Report

## Introduction

Academic procedural information at IIT Bombay is spread across several
official PDFs. This project is a Retrieval-Augmented Generation (RAG)
assistant that answers questions strictly from those documents and refuses
to answer when the retrieved context doesn't support it.

## Chosen Scope

Covers only: Course Registration, Add/Drop Policy, Academic Calendar,
Examination Rules, Grading Policy, CPI/SPI, and Academic Regulations.
Hostel, mess, clubs, placements, and admissions are out of scope.

## Data Sources

Two official IIT Bombay academic PDFs (e.g. Academic Calendar, Academic Regulations) are placed in `data/` and parsed locally.

## Chunking Strategy

Page text from `pdfplumber` is split with `RecursiveCharacterTextSplitter`
using `chunk_size=500` and `chunk_overlap=100`. This keeps chunks focused
for retrieval while the overlap avoids cutting relevant sentences at
boundaries. Each chunk stores its source filename and page number.

## Embedding Model

`all-MiniLM-L6-v2` (sentence-transformers) — fast, runs on CPU, and
produces a compact 384-dimension embedding, which is enough for short
academic passages.

## Retrieval

Embeddings are L2-normalized and stored in a FAISS `IndexFlatIP`, so
inner-product search behaves as cosine similarity. The question is
embedded with the same model and the top-4 most similar chunks are
retrieved. The chunks are passed to Gemini (`gemini-1.5-flash`) with a
strict system prompt to answer only from context, or return the fixed
refusal string otherwise. If retrieval returns nothing, the app shows the
refusal message directly without calling Gemini.

## Results

On sample IITB academic PDFs, the assistant answers direct factual
questions (e.g. CPI probation thresholds, add/drop windows) with correct
source citations, and refuses questions outside the ingested documents.

## Limitations

- Answer quality depends on which PDFs are supplied.
- Scanned/image-only PDFs without OCR give no extractable text.
- No conversational memory across questions.
- Dense retrieval can miss exact keyword matches (e.g. clause numbers).

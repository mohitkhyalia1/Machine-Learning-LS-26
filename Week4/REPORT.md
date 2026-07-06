# Machine-Learning-LS-26

WnCC LearnerSpace 2026 - Week 4 Report

* IITB Academic Assistant

## Project Overview

Built a RAG-based assistant that answers IIT Bombay academic questions using official PDF documents.

## What It Does

- Reads academic PDFs from the data folder
- Splits the text into smaller chunks
- Converts chunks into embeddings and stores them in FAISS
- Retrives the most relevant chunks for a user question
- Uses Gemini to generate an answer based only on the retrieved context

## Key Feature

- If the answer is not present in the documents, it returns a fixed refusal message.

## Limitations

- Quality depends on the PDFs provided
- Scanned PDFs without OCR may not work well
- No memory across conversations

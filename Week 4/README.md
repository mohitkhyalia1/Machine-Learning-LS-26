# IITB Academic Assistant

A RAG assistant that answers IIT Bombay academic questions — course
registration, add/drop policy, academic calendar, examination rules,
grading policy, CPI/SPI, and academic regulations — using only official
IITB PDF documents.

If the answer isn't in the documents, it replies exactly:
> I don't know based on the available IIT Bombay documents.

## Folder Structure

```
IITB-Academic-Assistant/
├── app.py            # Streamlit UI, retrieval, Gemini call
├── build_index.py    # PDF parsing, chunking, embedding, FAISS index build
├── requirements.txt
├── README.md
├── REPORT.md
├── .env.example
├── .gitignore
├── data/              
└── vectorstore/        # generated: index.faiss + metadata.pkl
```

## Installation

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Setup

1. Copy `.env.example` to `.env` and add your Gemini API key
   (get one free at https://aistudio.google.com/app/apikey):

   ```bash
   cp .env.example .env
   ```

2. Add at least 5 official IIT Bombay academic PDFs to `data/`
   (e.g. Academic Calendar, Examination Rules, Grading Policy,
   Academic Regulations, Course Registration guidelines).

## Running the Project

Build the vector index (run once, or whenever PDFs in `data/` change):

```bash
python build_index.py
```

Start the app:

```bash
streamlit run app.py
```

Open the URL shown in the terminal, type a question, and click **Ask**.

## Example Questions

- What is the minimum CPI required to avoid academic probation?
- How many days are given for the Add/Drop period?
- What is the passing grade in a course at IIT Bombay?
- When does the academic calendar for the even semester begin?

## Known Limitations

- Answer quality depends on which PDFs are placed in `data/`.
- Scanned (image-only) PDFs without OCR produce no extractable text.
- No conversational memory — each question is answered independently.

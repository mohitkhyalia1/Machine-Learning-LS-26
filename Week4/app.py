"""
app.py

Streamlit UI for the IITB Academic Assistant.
Loads the FAISS index built by build_index.py, retrieves the top-4
matching chunks for a question, and asks Gemini to answer strictly
from that context.

Run with:
    streamlit run app.py
"""

import os
import pickle
from pathlib import Path
from typing import List

import faiss
import google.generativeai as genai
import streamlit as st
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent

VECTORSTORE_DIR = BASE_DIR / "vectorstore"
INDEX_PATH = VECTORSTORE_DIR / "index.faiss"
METADATA_PATH = VECTORSTORE_DIR / "metadata.pkl"

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
GEMINI_MODEL_NAME = "gemini-2.5-flash"
TOP_K = 4

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
REFUSAL_MESSAGE = "I don't know based on the available IIT Bombay documents."

SYSTEM_PROMPT = f"""You are an IIT Bombay Academic Assistant.
Answer ONLY using the provided context.
Never use your own knowledge.
If the answer is not present in the context, reply exactly:
'{REFUSAL_MESSAGE}'
Always mention the sources used."""


@st.cache_resource
def load_embedding_model() -> SentenceTransformer:
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


@st.cache_resource
def load_vectorstore():
    """Load the FAISS index and chunk metadata from disk."""
    if not INDEX_PATH.exists() or not METADATA_PATH.exists():
        return None, None
    index = faiss.read_index(str(INDEX_PATH))
    with open(METADATA_PATH, "rb") as f:
        chunks = pickle.load(f)
    return index, chunks


def retrieve_top_chunks(question: str, index, chunks: List[dict]) -> List[dict]:
    """Embed the question and return the top-4 most similar chunks."""
    model = load_embedding_model()
    query_vector = model.encode(
        [question], convert_to_numpy=True, normalize_embeddings=True
    ).astype("float32")

    k = min(TOP_K, index.ntotal)
    if k == 0:
        return []

    scores, indices = index.search(query_vector, k)
    return [chunks[i] for i in indices[0] if i != -1]


def build_context(retrieved: List[dict]) -> str:
    """Format retrieved chunks into a labeled context block for the prompt."""
    blocks = [
        f"[Source: {c['filename']} (Page {c['page']})]\n{c['text']}"
        for c in retrieved
    ]
    return "\n\n".join(blocks)


def ask_gemini(question: str, context: str) -> str:
    """Send the grounded prompt to Gemini and return the answer text."""
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel(model_name=GEMINI_MODEL_NAME, system_instruction=SYSTEM_PROMPT)

    prompt = (
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        "Answer strictly using only the context above. If the context does not "
        f"contain the answer, respond exactly with: '{REFUSAL_MESSAGE}'"
    )
    response = model.generate_content(prompt)
    return (response.text or "").strip()


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------
st.set_page_config(page_title="IITB Academic Assistant", page_icon="🎓")
st.title("IITB Academic Assistant")

if not GEMINI_API_KEY:
    st.error("GEMINI_API_KEY is missing. Add it to your .env file (see .env.example).")

index, chunks = load_vectorstore()
if index is None:
    st.error("Vector store not found. Run 'python build_index.py' first.")

question = st.text_input("Ask a question about IIT Bombay academics:")

if st.button("Ask"):
    if not question.strip():
        st.warning("Please enter a question.")
    elif index is None:
        st.warning("The vector store is not built yet.")
    elif not GEMINI_API_KEY:
        st.warning("Cannot generate an answer without a valid GEMINI_API_KEY.")
    else:
        with st.spinner("Retrieving relevant documents..."):
            retrieved = retrieve_top_chunks(question, index, chunks)

        if not retrieved:
            st.subheader("Answer")
            st.write(REFUSAL_MESSAGE)
        else:
            context = build_context(retrieved)
            try:
                with st.spinner("Generating answer..."):
                    answer = ask_gemini(question, context)
            except Exception as err:
                st.error(f"Gemini API call failed: {err}")
                answer = None

            if answer:
                is_refusal = REFUSAL_MESSAGE.lower() in answer.lower()

                st.subheader("Answer")
                st.write(answer if not is_refusal else REFUSAL_MESSAGE)

                st.subheader("Sources")
                if is_refusal:
                    st.write("No sources used (answer not found in documents).")
                else:
                    for c in retrieved:
                        st.write(f"- {c['filename']} (Page {c['page']})")

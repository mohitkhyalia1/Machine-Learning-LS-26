# Machine Learning — LS 26

A compact collection of weekly lab notebooks and a small Streamlit app built for the "Machine Learning LS 26" course. The repo contains exercises, starter notebooks, and a Week-4 demo application that uses a FAISS vectorstore and Google Gemini to answer grounded questions from IIT Bombay documents.

**Contents**
- `Week1/` — Jupyter notebooks for basic visualization and a movie-critic demo.
- `Week2/` — Sentiment analyser starter notebook.
- `Week3/` — Console chatbot assignment and related notebooks.
- `Week4/` — Streamlit app, index builder, data and vectorstore files.
  
- `Live` - https://iitbacademicassistant.streamlit.app/


**Project notes**
- The Week-4 app (`Week4/app.py`) expects a FAISS index at `Week4/vectorstore/index.faiss` and metadata at `Week4/vectorstore/metadata.pkl`.
- The embedding model used is `sentence-transformers/all-MiniLM-L6-v2` and Gemini is used as the LLM (configured via `GEMINI_API_KEY`).
- If you get errors about missing files, run `python Week4/build_index.py` to recreate the vectorstore.
- A `.devcontainer/` folder is present for development container configuration (see `.devcontainer/devcontainer.json`).

# Optimized RAG – Quick Start

This project is an **Optimized Retrieval-Augmented Generation (RAG) chatbot** built with Streamlit, LangChain, and FAISS. It lets you upload documents, embed them, and ask questions in a conversational interface powered by LLMs.It also includes evaluation methods for retrieval quality and conversational performance.

---

## Quick Setup

```bash
# Clone repo
git clone <repo-url>
cd optimized_rag-main

# Install requirements
pip install -r requirements.txt
```

Add your OpenAI API key to a `.env` file:

```
OPENAI_API_KEY=your_api_key_here
```

---

## Run the App

```bash
streamlit run app.py
```

---

## Features

- Upload PDFs and create embeddings  
- Query your documents through a chatbot interface  
- Uses FAISS for fast retrieval and LangChain for orchestration  
- Supports evaluation (cosine similarity, human review, Ragas)  
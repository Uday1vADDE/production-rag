# DocMind — Production RAG System

> A production-grade Retrieval-Augmented Generation system with hybrid search, semantic reranking, multi-document support, and real-time monitoring.

🔗 **Live Demo:** [rag-system-uvadde.streamlit.app](https://rag-system-uvadde.streamlit.app)

---

## Features

- **Multi-file support** — Upload PDF, DOCX, TXT, and CSV files
- **Multi-document querying** — Query across multiple documents simultaneously
- **Hybrid search** — Combines ChromaDB vector search + BM25 keyword search for better retrieval
- **Cross-encoder reranking** — Re-ranks retrieved chunks using `ms-marco-MiniLM-L-6-v2` for precision
- **Query rewriting** — Automatically rewrites follow-up questions using conversation history
- **Citation enforcement** — Answers always cite source chunks, no hallucination without evidence
- **Source attribution** — Shows which document each chunk came from
- **Streaming responses** — Real-time token streaming like ChatGPT
- **Langfuse monitoring** — Full observability with latency tracking and trace logging
- **Professional dark UI** — Custom-designed interface with DocMind branding

---

## Architecture

```
User Query
    │
    ▼
Query Rewriting (llama-3.1-8b-instant)
    │
    ▼
Hybrid Retrieval
    ├── ChromaDB Vector Search (all-MiniLM-L6-v2)
    └── BM25 Keyword Search
    │
    ▼
Cross-Encoder Reranking (ms-marco-MiniLM-L-6-v2)
    │
    ▼
Prompt Building (versioned YAML prompt)
    │
    ▼
LLM Generation (llama-3.3-70b-versatile via Groq)
    │
    ▼
Citation Validation → Streaming Response
    │
    ▼
Langfuse Monitoring (traces, latency, scores)
```

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Frontend | Streamlit |
| LLM | LLaMA-3.3-70B via Groq |
| Rewrite LLM | LLaMA-3.1-8B via Groq |
| Embeddings | all-MiniLM-L6-v2 (HuggingFace) |
| Vector Store | ChromaDB |
| Keyword Search | BM25Okapi (rank-bm25) |
| Reranker | cross-encoder/ms-marco-MiniLM-L-6-v2 |
| Document Loaders | LangChain (PyPDF, Docx2txt, TextLoader, CSVLoader) |
| Monitoring | Langfuse |
| Deployment | Streamlit Cloud |

---

## Evaluation Results

Evaluated on 10 questions from a test document using a custom LLM-based evaluator:

| Metric | Score |
|--------|-------|
| Faithfulness | 1.000 |
| Answer Relevancy | 0.900 |
| Context Precision | 0.850 |
| **Overall** | **0.917** |

---

## Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/Uday1vADDE/production-rag.git
cd production-rag
```

### 2. Create virtual environment

```bash
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Mac/Linux
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set up environment variables

Create a `.env` file in the root directory:

```
GROQ_API_KEY=your_groq_api_key
LANGFUSE_SECRET_KEY=your_langfuse_secret_key
LANGFUSE_PUBLIC_KEY=your_langfuse_public_key
LANGFUSE_HOST=https://us.cloud.langfuse.com
```

Get your Groq API key at [console.groq.com](https://console.groq.com)  
Get your Langfuse keys at [cloud.langfuse.com](https://cloud.langfuse.com)

### 5. Run the app

```bash
streamlit run app.py
```

---

## Project Structure

```
production-rag/
├── src/
│   ├── ingest.py          # Document loading, chunking, ChromaDB + BM25 storage
│   ├── retrieval.py       # Hybrid search + cross-encoder reranking
│   ├── pipeline.py        # Query rewriting, prompt building, LLM generation
│   └── evaluate.py        # Custom LLM-based evaluation
├── prompts/
│   └── rag_prompt.yaml    # Versioned RAG prompt
├── evaluation/
│   ├── golden_dataset.json
│   └── results.json
├── app.py                 # Streamlit frontend
├── requirements.txt
└── .env                   # API keys (not committed)
```

---

## How It Works

1. **Upload** a PDF, DOCX, TXT, or CSV file
2. The document is **chunked** into 2000-character segments with 400-character overlap
3. Chunks are **embedded** using `all-MiniLM-L6-v2` and stored in ChromaDB
4. A **BM25 index** is also built for keyword search
5. When you ask a question, it's optionally **rewritten** to resolve pronouns
6. **Hybrid search** retrieves top candidates from both ChromaDB and BM25
7. A **cross-encoder reranker** scores and selects the best 5 chunks
8. The LLM generates a **cited answer** referencing specific chunks
9. Every query is **traced** in Langfuse with latency metrics

---

## Monitoring

This project uses [Langfuse](https://langfuse.com) for production monitoring. Each query generates a trace with:
- Input query and source documents
- Retrieval latency
- LLM generation latency  
- Final answer with citations

---
## Learnings & Challenges

### What I Learned
- How to build a production RAG pipeline from scratch — not just calling an LLM but understanding every component: chunking strategy, embedding, hybrid retrieval, reranking
- The difference between naive RAG (just vector search) and production RAG (hybrid search + reranking + citation enforcement)
- How ChromaDB collection naming rules can break your app with real-world filenames — learned to sanitize inputs defensively
- How Langfuse monitoring works and why observability matters in production AI systems
- How to structure a multi-file Python project with clean separation between ingestion, retrieval, and generation
- Real streaming with generators (`yield`) vs fake streaming — and why it matters for user experience

### Challenges I Overcame
- **ChromaDB version conflicts** on Streamlit Cloud — `langchain-chroma==1.1.0` required `chromadb>=1.3.5` but old pinned versions kept breaking. Fixed by understanding dependency resolution.
- **Collection name validation errors** — real-world filenames like `Assignment2.1 (1).pdf` or `USCIS - I-765.pdf` broke ChromaDB's strict naming rules. Built a robust `sanitize_collection_name()` function using regex.
- **Langfuse SDK breaking changes** — v4 removed `.trace()` and `.create_trace()` methods. Had to inspect the actual SDK source to find the correct `@observe` decorator API.
- **Query rewriting false positives** — the rewrite LLM was rewriting simple queries unnecessarily. Fixed with pronoun detection before calling the rewrite LLM.
- **Citation enforcement** — LLMs often answer without citing sources. Solved with regex validation that rejects uncited answers entirely.
- **Streaming + citation check conflict** — can't check citations mid-stream. Solved with two-phase approach: stream tokens to user, validate full response after completion.

*Built with LangChain, Groq, ChromaDB, and Streamlit*

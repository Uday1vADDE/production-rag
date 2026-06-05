import os
import pickle
from pathlib import Path
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
from dotenv import load_dotenv
from ingest import sanitize_collection_name

load_dotenv()

embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


def load_retrievers(file_path):
    name = sanitize_collection_name(file_path)
    bm25_path = f"vectorstore/{name}_bm25.pkl"
    texts_path = f"vectorstore/{name}_texts.pkl"

    if os.path.exists(bm25_path):
        with open(bm25_path, "rb") as f:
            bm25 = pickle.load(f)
        with open(texts_path, "rb") as f:
            texts = pickle.load(f)

        vectorstore = Chroma(
            embedding_function=embeddings,
            collection_name=name,
            persist_directory="vectorstore"
        )
        return vectorstore, bm25, texts, name
    else:
        raise FileNotFoundError(f"'{name}' not processed yet. Please upload and process it first.")


def hybrid_search(query, vectorstore, bm25, texts, k, source_name):
    query_words = query.lower().split()
    bm25_chunks = bm25.get_top_n(query_words, texts, n=k)
    vectorstore_chunks = vectorstore.similarity_search(query, k=k)

    # keep source metadata
    results = []
    for chunk in bm25_chunks:
        results.append({"text": chunk, "source": source_name})
    for doc in vectorstore_chunks:
        results.append({"text": doc.page_content, "source": source_name})

    # deduplicate by text
    seen = set()
    unique = []
    for r in results:
        if r["text"] not in seen:
            seen.add(r["text"])
            unique.append(r)
    return unique


def rerank(query, chunks_with_source, top_k):
    pairs = [[query, c["text"]] for c in chunks_with_source]
    scores = reranker.predict(pairs)
    scored = sorted(zip(scores, chunks_with_source), reverse=True)
    return [c for score, c in scored[:top_k]]


def retrieve(file_paths, query, k=10, top_k=5):
    all_chunks = []

    for file_path in file_paths:
        vectorstore, bm25, texts, name = load_retrievers(file_path)
        source_name = Path(file_path).name  # original filename e.g. mcp_guide.pdf
        chunks = hybrid_search(query, vectorstore, bm25, texts, k, source_name)
        all_chunks += chunks

    # deduplicate across files
    seen = set()
    unique = []
    for c in all_chunks:
        if c["text"] not in seen:
            seen.add(c["text"])
            unique.append(c)

    return rerank(query, unique, top_k)
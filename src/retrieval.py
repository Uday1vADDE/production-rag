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


def load_retrievers(pdf_path):
    pdf_name=sanitize_collection_name(pdf_path)

    bm25_path=f"vectorstore/{pdf_name}_bm25.pkl"
    texts_path=f"vectorstore/{pdf_name}_texts.pkl"

    if os.path.exists(bm25_path):
        with open(bm25_path,"rb") as f:
            bm25=pickle.load(f)
        with open (texts_path,"rb")as f:
            texts=pickle.load(f)
        
        vectorstore=Chroma(
            embedding_function=embeddings,
            collection_name=pdf_name,
            persist_directory="vectorstore"
        )
        return vectorstore,bm25,texts,pdf_name
    else:
        raise FileNotFoundError(f"PDF '{pdf_name}' not processed yet. Please upload and process it first.")
    
def hybrid_search(query,vectorstore,bm25,texts,k):
    query_words=query.lower().split()
    bm25_chunks=bm25.get_top_n(query_words,texts,n=k)
    vectorstore_chunks=vectorstore.similarity_search(query,k=k)
    vector_text=[doc.page_content for doc in vectorstore_chunks]
    total_chunks=bm25_chunks+vector_text 
    return list(dict.fromkeys(total_chunks))


def rerank(query,hybrid_search_chunks,top_k):
    pairs=[[query,chunk] for chunk in hybrid_search_chunks]
    scores=reranker.predict(pairs)
    scored_chunks=sorted(zip(scores,hybrid_search_chunks),reverse=True)
    top_chunks=[chunk for score,chunk in scored_chunks[:top_k]]
    return top_chunks

def retrieve(pdf_path,query,k=10,top_k=5):
     vectorstore,bm25,texts,pdf_name=load_retrievers(pdf_path)
     hybrid_search_chunks=hybrid_search(query,vectorstore,bm25,texts,k)
     return rerank(query,hybrid_search_chunks,top_k)
    
     




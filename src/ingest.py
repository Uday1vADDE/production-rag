import os
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
#from langchain_community.vectorstores import Chroma
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from rank_bm25 import BM25Okapi
from dotenv import load_dotenv
import pickle

load_dotenv()


embeddings=HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}

    )

def load_pdf(pdf_path):
    print(f"Loading pdf:{pdf_path}")
    loader=PyPDFLoader(pdf_path)
    pages=loader.load()
    print(f"loaded{len(pages)} pages")
    return pages

def split_documents(pages):
    print("Splitting documents into chunks...")
    splitter=RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=400,
        length_function=len,
        separators=["\n\n","\n","."," ",""]

     )
    chunks=splitter.split_documents(pages)
    print(f"Created {len(chunks)} chunks")
    return chunks

def store_in_vectordb(chunks,pdf_path):
    print("Storing chunks in ChromaDB...")
    #embeddings = get_embedding_model()
    
    # Create a clean collection name from PDF filename
    collection_name = Path(pdf_path).stem.replace(" ", "_").lower()
    
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=collection_name,
        persist_directory="vectorstore"
    )
    
    print(f"Stored {len(chunks)} chunks in collection: {collection_name}")
    return vectorstore, collection_name

def store_bm25(chunks,pdf_path):
    print("Creating BM25 index...")
    texts=[chunk.page_content for chunk in chunks]
    tokenized_chunks=[text.lower().split() for text in texts]

    #create bm25 index
    bm25=BM25Okapi(tokenized_chunks)

    #save to vectorstore using pickle
    collection_name=Path(pdf_path).stem.replace(" ","_").lower()
    bm25_path=f"vectorstore/{collection_name}_bm25.pkl"
    with open(bm25_path,"wb") as f:
        pickle.dump(bm25,f)

    #also save original text
    texts_path=f"vectorstore/{collection_name}_texts.pkl"
    with open(texts_path,"wb") as f:
        pickle.dump(texts,f)

    print(f"BM25 index saved to {bm25_path}")
    return bm25, texts


def process_pdf(pdf_path):
    print(f"\n{'='*50}")
    print(f"Processing: {pdf_path}")
    print(f"{'='*50}")

    pdf_name=Path(pdf_path).stem.replace(" ","_").lower()
    bm25_path=f"vectorstore/{pdf_name}_bm25.pkl"

    if os.path.exists(bm25_path):
        print(f"PDF already processed! Loading from cache...")

        #embeddings=get_embedding_model()
        vectorstore=Chroma(
            collection_name=pdf_name,
            embedding_function=embeddings,
            persist_directory="vectorstore"
        )

        with open(bm25_path,"rb") as f:
            bm25=pickle.load(f)
        

        texts_path=f"vectorstore/{pdf_name}_texts.pkl"

        with open(texts_path,"rb") as f:
            texts=pickle.load(f)

        print("Loaded from cache successfully!")
        return vectorstore,bm25,texts,pdf_name
    
    #fresh proceesing
    pages=load_pdf(pdf_path)
    chunks=split_documents(pages)
    vectorstore,collection_name=store_in_vectordb(chunks,pdf_path)
    bm25,texts=store_bm25(chunks,pdf_path)
    print(f"\n✓ PDF processed successfully!")
    print(f"✓ Total chunks: {len(chunks)}")
    return vectorstore, bm25, texts, collection_name

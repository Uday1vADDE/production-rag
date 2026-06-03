import os
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader, CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from rank_bm25 import BM25Okapi
from dotenv import load_dotenv
import pickle
import re

load_dotenv()

embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)


def sanitize_collection_name(file_path):
    stem = Path(file_path).stem.lower()
    collection_name = re.sub(r'[^a-z0-9._-]', '_', stem)
    collection_name = re.sub(r'_+', '_', collection_name)
    collection_name = re.sub(r'[^a-z0-9]+$', '', collection_name)  # remove trailing non-alphanumeric
    collection_name = re.sub(r'^[^a-z0-9]+', '', collection_name)  # remove leading non-alphanumeric
    if len(collection_name) < 3:
        collection_name = "doc_" + collection_name
    return collection_name


def load_document(file_path):
    extension = Path(file_path).suffix.lower()
    print(f"Loading {extension} file: {file_path}")

    if extension == ".pdf":
        loader = PyPDFLoader(file_path)
    elif extension == ".docx":
        loader = Docx2txtLoader(file_path)
    elif extension == ".txt":
        loader = TextLoader(file_path)
    elif extension == ".csv":
        loader = CSVLoader(file_path)
    else:
        raise ValueError(f"Unsupported file type: {extension}")

    pages = loader.load()
    print(f"Loaded {len(pages)} pages")
    return pages


def split_documents(pages):
    print("Splitting documents into chunks...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=400,
        length_function=len,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    chunks = splitter.split_documents(pages)
    print(f"Created {len(chunks)} chunks")
    return chunks


def store_in_vectordb(chunks, file_path):
    print("Storing chunks in ChromaDB...")
    collection_name = sanitize_collection_name(file_path)

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=collection_name,
        persist_directory="vectorstore"
    )

    print(f"Stored {len(chunks)} chunks in collection: {collection_name}")
    return vectorstore, collection_name


def store_bm25(chunks, file_path):
    print("Creating BM25 index...")
    texts = [chunk.page_content for chunk in chunks]
    tokenized_chunks = [text.lower().split() for text in texts]

    bm25 = BM25Okapi(tokenized_chunks)

    collection_name = sanitize_collection_name(file_path)
    bm25_path = f"vectorstore/{collection_name}_bm25.pkl"
    with open(bm25_path, "wb") as f:
        pickle.dump(bm25, f)

    texts_path = f"vectorstore/{collection_name}_texts.pkl"
    with open(texts_path, "wb") as f:
        pickle.dump(texts, f)

    print(f"BM25 index saved to {bm25_path}")
    return bm25, texts


def process_document(file_path):
    print(f"\n{'='*50}")
    print(f"Processing: {file_path}")
    print(f"{'='*50}")

    collection_name = sanitize_collection_name(file_path)
    bm25_path = f"vectorstore/{collection_name}_bm25.pkl"

    if os.path.exists(bm25_path):
        print(f"Already processed! Loading from cache...")

        vectorstore = Chroma(
            collection_name=collection_name,
            embedding_function=embeddings,
            persist_directory="vectorstore"
        )

        with open(bm25_path, "rb") as f:
            bm25 = pickle.load(f)

        texts_path = f"vectorstore/{collection_name}_texts.pkl"
        with open(texts_path, "rb") as f:
            texts = pickle.load(f)

        print("Loaded from cache successfully!")
        return vectorstore, bm25, texts, collection_name

    # fresh processing
    pages = load_document(file_path)
    chunks = split_documents(pages)
    vectorstore, collection_name = store_in_vectordb(chunks, file_path)
    bm25, texts = store_bm25(chunks, file_path)
    print(f"\n✓ Document processed successfully!")
    print(f"✓ Total chunks: {len(chunks)}")
    return vectorstore, bm25, texts, collection_name
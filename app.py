import sys
sys.path.append("src")
import streamlit as st
import os
import json
from ingest import process_pdf
from pipeline import get_answers
from retrieval import retrieve

st.set_page_config(
    page_title="Production RAG System",
    page_icon="📄",
)

# Session state
if "pdf_processed" not in st.session_state:
    st.session_state.pdf_processed = False
if "pdf_path" not in st.session_state:
    st.session_state.pdf_path = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "chunks_history" not in st.session_state:
    st.session_state.chunks_history = []

# Sidebar
with st.sidebar:
    st.title("📄 RAG System")
    st.markdown("---")
    
    st.subheader("Upload PDF")
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf", label_visibility="collapsed")
    
    if uploaded_file is not None:
        save_path = f"data/{uploaded_file.name}"
        
        if st.session_state.pdf_path != save_path:
            with open(save_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            with st.spinner("Processing..."):
                process_pdf(save_path)
            
            st.session_state.pdf_processed = True
            st.session_state.pdf_path = save_path
            st.session_state.chat_history = []
            st.session_state.chunks_history = []
        
        file_size = os.path.getsize(save_path)
        st.success(f"✅ {uploaded_file.name}")
        st.caption(f"Size: {file_size/1024:.1f} KB")
    
    else:
        # User removed the file
        if st.session_state.pdf_processed:
            st.session_state.pdf_processed = False
            st.session_state.pdf_path = None
            st.session_state.chat_history = []
            st.session_state.chunks_history = []
    
    st.markdown("---")
    
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.chat_history = []
        st.session_state.chunks_history = []
        st.rerun()

# Main area
st.title("Chat with your PDF")

if not st.session_state.pdf_processed:
    st.info("👈 Upload a PDF from the sidebar to start chatting.")
else:
    for i, message in enumerate(st.session_state.chat_history):
        with st.chat_message(message["role"]):
            st.write(message["content"])
            
            # Show chunks only for assistant messages that have citations
            if message["role"] == "assistant" and "I don't have enough" not in message["content"]:
                assistant_count = sum(1 for m in st.session_state.chat_history[:i] if m["role"] == "assistant")
                if assistant_count < len(st.session_state.chunks_history):
                    chunks = st.session_state.chunks_history[assistant_count]
                    with st.expander("📄 View source chunks"):
                        for j, chunk in enumerate(chunks):
                            st.markdown(f"**Chunk {j+1}:**")
                            st.caption(chunk[:300] + "..." if len(chunk) > 300 else chunk)
                            st.markdown("---")
    
    query = st.chat_input("Ask a question about your PDF...")
    
    if query:
        st.session_state.chat_history.append({
            "role": "user",
            "content": query
        })
        
        with st.spinner("Thinking..."):
            answer = get_answers(
                st.session_state.pdf_path,
                query,
                st.session_state.chat_history
            )
            chunks = retrieve(
                st.session_state.pdf_path,
                query,
                k=10,
                top_k=5
            )
        
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": answer
        })
        st.session_state.chunks_history.append(chunks)
        
        st.rerun()
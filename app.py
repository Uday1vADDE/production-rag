import sys
sys.path.append("src")
import streamlit as st
import os
from ingest import process_document
from pipeline import get_answers
from retrieval import retrieve

os.makedirs("data", exist_ok=True)
os.makedirs("vectorstore", exist_ok=True)

st.set_page_config(
    page_title="DocMind — RAG System",
    page_icon="🧠",
    layout="wide"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:wght@300;400;500&display=swap');

html, body, [class*="css"] { font-family: 'DM Mono', monospace; }

.stApp { background: #0a0a0f; color: #e8e6f0; }

[data-testid="stSidebar"] {
    background: #0f0f18 !important;
    border-right: 1px solid #1e1e2e;
}
[data-testid="stSidebar"] > div { padding: 2rem 1.5rem; }

.logo-block { margin-bottom: 2rem; }
.logo-block .logo-text {
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: 1.6rem;
    color: #c8b8ff;
    letter-spacing: -0.02em;
    line-height: 1;
}
.logo-block .logo-sub {
    font-size: 0.65rem;
    color: #555577;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    margin-top: 4px;
}

.section-label {
    font-size: 0.6rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: #444466;
    margin-bottom: 0.75rem;
    font-family: 'DM Mono', monospace;
}

[data-testid="stFileUploader"] {
    background: #13131f !important;
    border: 1px dashed #2a2a44 !important;
    border-radius: 10px !important;
    padding: 0.5rem !important;
}
[data-testid="stFileUploader"]:hover { border-color: #7c6aff !important; }

.file-card {
    background: linear-gradient(135deg, #13131f, #1a1a2e);
    border: 1px solid #2a2a44;
    border-left: 3px solid #7c6aff;
    border-radius: 8px;
    padding: 0.75rem 1rem;
    margin-top: 0.5rem;
}
.file-card .file-name {
    font-size: 0.75rem;
    color: #c8b8ff;
    font-weight: 500;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}
.file-card .file-meta { font-size: 0.6rem; color: #555577; margin-top: 2px; }
.file-card .file-dot {
    display: inline-block;
    width: 6px; height: 6px;
    background: #4ade80;
    border-radius: 50%;
    margin-right: 6px;
    animation: pulse 2s infinite;
}
@keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.4; } }

.custom-divider { border: none; border-top: 1px solid #1e1e2e; margin: 1.5rem 0; }

.stButton > button {
    background: transparent !important;
    border: 1px solid #2a2a44 !important;
    color: #888899 !important;
    border-radius: 8px !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.7rem !important;
    letter-spacing: 0.05em !important;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    border-color: #7c6aff !important;
    color: #c8b8ff !important;
    background: #13131f !important;
}

.main-header { padding: 3rem 0 2rem; border-bottom: 1px solid #1e1e2e; margin-bottom: 2rem; }
.main-header h1 {
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: 2.5rem;
    color: #e8e6f0;
    letter-spacing: -0.03em;
    margin: 0;
    line-height: 1;
}
.main-header h1 span { color: #7c6aff; }
.main-header p { color: #555577; font-size: 0.75rem; margin-top: 0.5rem; letter-spacing: 0.05em; }

.empty-state { text-align: center; padding: 5rem 2rem; color: #333355; }
.empty-state .empty-icon { font-size: 3rem; margin-bottom: 1rem; opacity: 0.4; }
.empty-state p { font-size: 0.75rem; letter-spacing: 0.1em; text-transform: uppercase; }

[data-testid="stChatMessage"] { background: transparent !important; border: none !important; padding: 0.5rem 0 !important; }

.user-bubble {
    background: #13131f;
    border: 1px solid #1e1e2e;
    border-radius: 12px 12px 4px 12px;
    padding: 0.9rem 1.2rem;
    max-width: 75%;
    margin-left: auto;
    font-size: 0.82rem;
    color: #d0cee8;
    line-height: 1.6;
}
.assistant-bubble {
    background: linear-gradient(135deg, #0f0f1a, #13131f);
    border: 1px solid #2a2a44;
    border-left: 3px solid #7c6aff;
    border-radius: 4px 12px 12px 12px;
    padding: 0.9rem 1.2rem;
    max-width: 85%;
    font-size: 0.82rem;
    color: #d0cee8;
    line-height: 1.7;
}

[data-testid="stExpander"] {
    background: #0d0d18 !important;
    border: 1px solid #1e1e2e !important;
    border-radius: 8px !important;
    margin-top: 0.5rem !important;
}
[data-testid="stExpander"] summary { font-size: 0.65rem !important; color: #555577 !important; letter-spacing: 0.1em !important; }

[data-testid="stChatInput"] {
    background: #0f0f18 !important;
    border: 1px solid #2a2a44 !important;
    border-radius: 12px !important;
}
[data-testid="stChatInput"]:focus-within {
    border-color: #7c6aff !important;
    box-shadow: 0 0 0 2px rgba(124, 106, 255, 0.1) !important;
}
[data-testid="stChatInput"] textarea {
    font-family: 'DM Mono', monospace !important;
    font-size: 0.8rem !important;
    color: #1a1a2e !important;
    background: transparent !important;
}

[data-testid="stSpinner"] { color: #7c6aff !important; }

::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: #0a0a0f; }
::-webkit-scrollbar-thumb { background: #2a2a44; border-radius: 2px; }
::-webkit-scrollbar-thumb:hover { background: #7c6aff; }

#MainMenu, footer, header { visibility: hidden; }
[data-testid="stDecoration"] { display: none; }
</style>
""", unsafe_allow_html=True)

# ── Session state ──
for key, default in [
    ("file_paths", []),
    ("chat_history", []),
    ("chunks_history", []),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ── Sidebar ──
with st.sidebar:
    st.markdown("""
    <div class="logo-block">
        <div class="logo-text">DocMind</div>
        <div class="logo-sub">RAG · Hybrid Search · Reranking</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-label">Documents</div>', unsafe_allow_html=True)

    uploaded_files = st.file_uploader(
        "upload",
        type=["pdf", "docx", "txt", "csv"],
        accept_multiple_files=True,
        label_visibility="collapsed"
    )

    if uploaded_files:
        current_names = [f.name for f in uploaded_files]
        existing_names = [os.path.basename(p) for p in st.session_state.file_paths]

        # process any new files
        for uploaded_file in uploaded_files:
            save_path = f"data/{uploaded_file.name}"

            if uploaded_file.name not in existing_names:
                with open(save_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                with st.spinner(f"Processing {uploaded_file.name}..."):
                    process_document(save_path)
                st.session_state.file_paths.append(save_path)

        # remove files that were unselected
        st.session_state.file_paths = [
            p for p in st.session_state.file_paths
            if os.path.basename(p) in current_names
        ]

        # show file cards
        for path in st.session_state.file_paths:
            file_size = os.path.getsize(path) / 1024
            fname = os.path.basename(path)
            st.markdown(f"""
            <div class="file-card">
                <div class="file-name"><span class="file-dot"></span>{fname}</div>
                <div class="file-meta">{file_size:.1f} KB · ready</div>
            </div>
            """, unsafe_allow_html=True)

    else:
        # all files removed
        if st.session_state.file_paths:
            st.session_state.file_paths = []
            st.session_state.chat_history = []
            st.session_state.chunks_history = []

    st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)

    if st.button("↺  Clear conversation", use_container_width=True):
        st.session_state.chat_history = []
        st.session_state.chunks_history = []
        st.rerun()

    st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)
    st.markdown('<div class="section-label">Stack</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size:0.6rem; color:#333355; line-height:2;">
    ChromaDB · BM25 · CrossEncoder<br>
    LLaMA-3.3-70B · Groq<br>
    all-MiniLM-L6-v2
    </div>
    """, unsafe_allow_html=True)

# ── Main ──
st.markdown("""
<div class="main-header">
    <h1>Chat with your <span>documents.</span></h1>
    <p>Hybrid search · Semantic reranking · Cited answers</p>
</div>
""", unsafe_allow_html=True)

if not st.session_state.file_paths:
    st.markdown("""
    <div class="empty-state">
        <div class="empty-icon">⬡</div>
        <p>Upload documents to begin</p>
    </div>
    """, unsafe_allow_html=True)
else:
    for i, message in enumerate(st.session_state.chat_history):
        if message["role"] == "user":
            st.markdown(f'<div class="user-bubble">{message["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="assistant-bubble">{message["content"]}</div>', unsafe_allow_html=True)

            has_info = "I don't have enough" not in message["content"]
            if has_info:
                assistant_count = sum(1 for m in st.session_state.chat_history[:i] if m["role"] == "assistant")
                if assistant_count < len(st.session_state.chunks_history):
                    chunks = st.session_state.chunks_history[assistant_count]
                    with st.expander("▸  view source chunks"):
                        for j, chunk in enumerate(chunks):
                            st.markdown(f"<span style='font-size:0.6rem;color:#555577;letter-spacing:0.1em;'>CHUNK {j+1}</span>", unsafe_allow_html=True)
                            #st.caption(chunk[:300] + "…" if len(chunk) > 300 else chunk)
                            st.markdown(f"<span style='font-size:0.6rem;color:#7c6aff;'>📄 {chunk['source']}</span>", unsafe_allow_html=True)
                            st.caption(chunk['text'][:300] + "…" if len(chunk['text']) > 300 else chunk['text'])
                            if j < len(chunks) - 1:
                                st.markdown('<hr style="border-color:#1e1e2e;margin:0.5rem 0">', unsafe_allow_html=True)

        st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

    query = st.chat_input("Ask anything about your documents…")

    if query:
        st.session_state.chat_history.append({"role": "user", "content": query})

        #st.write(st.session_state.file_paths)  # debug

        chunks = retrieve(
            st.session_state.file_paths,
            query,
            k=10,
            top_k=5
        )

        with st.chat_message("assistant"):
            answer = st.write_stream(
            get_answers(
                st.session_state.file_paths,
                query,
                st.session_state.chat_history
             )
            )

        st.session_state.chat_history.append({"role": "assistant", "content": answer})
        st.session_state.chunks_history.append(chunks)
        st.rerun()
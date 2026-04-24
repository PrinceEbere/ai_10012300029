# Student: Prince Ebere Enoch, Index: 10012300029
# CS4241 - Introduction to Artificial Intelligence - 2026

import os
import logging
import streamlit as st
import numpy as np
import faiss

from src.loader import load_csv, load_pdf
from src.cleaner import clean_text
from src.chunker import chunk_text
from src.embedder import create_embeddings, get_model
from src.retriever import Retriever
from src.generator import generate_response


# ----------------------------
# PAGE CONFIG
# ----------------------------
st.set_page_config(
    page_title="AI RAG Assistant",
    page_icon="💬",
    layout="wide"
)


# ----------------------------
# UI STYLING
# ----------------------------
st.markdown("""
<style>
.stApp {
    background-color: #f8fafc;
}
.block-container {
    padding-top: 2rem;
    max-width: 1100px;
}
.stChatInput input {
    border-radius: 12px !important;
    padding: 12px !important;
}
section[data-testid="stSidebar"] {
    background-color: #ffffff;
    border-right: 1px solid #e5e7eb;
}
</style>
""", unsafe_allow_html=True)


# ----------------------------
# LOGGING SETUP
# ----------------------------
os.makedirs("logs", exist_ok=True)

logging.basicConfig(
    filename='logs/experiment_logs.txt',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


# ----------------------------
# DATA LOADING
# ----------------------------
@st.cache_data
def load_data():
    csv_data = load_csv("Data/Ghana_Election_Result.csv")
    pdf_text = load_pdf("Data/2025-Budget.pdf")

    logging.info(f"Loaded CSV rows={len(csv_data)}, PDF chars={len(pdf_text)}")

    csv_text = csv_data.astype(str).to_csv(index=False)
    return pdf_text + "\n" + csv_text


# ----------------------------
# VECTOR STORE (FIXED)
# ----------------------------
@st.cache_resource
def build_vector_store():
    combined_text = load_data()

    cleaned = clean_text(combined_text)
    chunks = chunk_text(cleaned, chunk_size=500, overlap=50)

    chunks = [c for c in chunks if isinstance(c, str) and c.strip()]

    embeddings = create_embeddings(chunks)

    # 🔥 IMPORTANT FIX: Normalize embeddings (cosine similarity)
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings.astype("float32"))

    retriever = Retriever(index, chunks)

    logging.info(f"Vector store built with {len(chunks)} chunks")

    return retriever, chunks


# ----------------------------
# RAG HELPERS
# ----------------------------
def expand_query(query):
    return query + " economic policy budget election government spending"


def select_context(chunks, scores, max_chars=1200):
    ranked = sorted(zip(chunks, scores), key=lambda x: x[1], reverse=True)

    selected = []
    total = 0

    for chunk, score in ranked:
        if total + len(chunk) > max_chars:
            break
        selected.append(chunk)
        total += len(chunk)

    return selected


# ----------------------------
# RAG PIPELINE (IMPROVED)
# ----------------------------
def rag_pipeline(query, retriever):
    expanded = expand_query(query)

    model = get_model()
    query_embedding = model.encode(expanded).astype("float32")

    # 🔥 Normalize query embedding
    query_embedding = query_embedding / np.linalg.norm(query_embedding)

    results, scores = retriever.search(query_embedding, k=5)

    # 🔥 If nothing useful found
    if len(results) == 0:
        logging.warning("No relevant chunks retrieved")
        return [], [], [], "I couldn't find relevant information in the documents."

    # Optional rerank
    if len(scores) > 0 and np.max(scores) < 0.45:
        results, scores = retriever.rerank(results, scores)

    context = select_context(results, scores)

    # 🔥 Stronger prompt control
    prompt = f"""
Answer the question using ONLY the context below.

Context:
{context}

Question:
{query}

If the answer is not in the context, say "I don't know".
"""

    answer = generate_response(prompt, context)

    return results, scores, context, answer


# ----------------------------
# HEADER
# ----------------------------
st.markdown("""
<div style="text-align:center; padding:1.5rem 0;">
    <h1 style="color:#0f172a;">💬 AI RAG Assistant</h1>
    <p style="color:#64748b;">
        Ghana Budget & Election Intelligence System
    </p>
</div>
""", unsafe_allow_html=True)


# ----------------------------
# SIDEBAR
# ----------------------------
with st.sidebar:
    st.title("🎓 AI Assistant")

    st.markdown("### 👤 Student")
    st.write("Prince Ebere Enoch")
    st.write("10012300029")

    st.markdown("### ⚙️ System")
    st.write("RAG + FAISS + LLM")

    st.markdown("### 📊 Capabilities")
    st.write("• Budget Analysis")
    st.write("• Election Insights")
    st.write("• Document QA")

    st.markdown("---")

    if st.button("🧹 Clear Chat"):
        st.session_state.chat_history = []


# ----------------------------
# CHAT MEMORY
# ----------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# ----------------------------
# LOAD VECTOR STORE
# ----------------------------
retriever, chunks = build_vector_store()


# ----------------------------
# DISPLAY CHAT
# ----------------------------
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


# ----------------------------
# INPUT
# ----------------------------
query = st.chat_input("Ask about Ghana budget or elections...")

if query:

    st.session_state.chat_history.append({
        "role": "user",
        "content": query
    })

    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Analyzing documents..."):

            results, scores, context, answer = rag_pipeline(query, retriever)

            st.markdown(f"""
            <div style="
                background-color: #ffffff;
                border: 1px solid #e5e7eb;
                border-left: 5px solid #2563eb;
                padding: 1.2rem;
                border-radius: 12px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.04);
                color: #111827;
                line-height: 1.6;
            ">
            {answer}
            </div>
            """, unsafe_allow_html=True)

            with st.expander("📄 Retrieved Context"):
                for i, (res, score) in enumerate(zip(results, scores)):
                    st.markdown(f"**Source {i+1} | Score: {score:.2f}**")
                    st.write(res[:250])

    st.session_state.chat_history.append({
        "role": "assistant",
        "content": answer
    })

    logging.info(f"Query: {query} | Answer: {answer}")


# ----------------------------
# FOOTER
# ----------------------------
st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:#94a3b8; font-size:0.85rem;'>"
    "AI RAG System • Academic City • 2026"
    "</div>",
    unsafe_allow_html=True
)

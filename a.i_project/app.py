# Student: Prince Ebere Enoch, Index: [Your Index Number]
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
# LOGGING SETUP
# ----------------------------
os.makedirs("logs", exist_ok=True)

logging.basicConfig(
    filename='logs/experiment_logs.txt',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


# ----------------------------
# DATA LOADING (CACHED)
# ----------------------------
@st.cache_data
def load_data():
    csv_data = load_csv("Data/Ghana_Election_Result.csv")
    pdf_text = load_pdf("Data/2025-Budget.pdf")

    logging.info(f"Loaded CSV rows={len(csv_data)}, PDF chars={len(pdf_text)}")

    csv_text = csv_data.astype(str).to_csv(index=False)
    return pdf_text + "\n" + csv_text


# ----------------------------
# VECTOR STORE (CACHED)
# ----------------------------
@st.cache_resource
def build_vector_store():
    combined_text = load_data()

    cleaned = clean_text(combined_text)
    chunks = chunk_text(cleaned, chunk_size=500, overlap=50)

    chunks = [c for c in chunks if isinstance(c, str) and c.strip()]

    embeddings = create_embeddings(chunks)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings.astype("float32"))

    retriever = Retriever(index, chunks)

    return retriever, chunks


# ----------------------------
# RAG HELPERS
# ----------------------------
def expand_query(query):
    synonyms = [
        "economic policy",
        "budget statement",
        "election results",
        "government spending"
    ]
    return query + " " + " ".join(synonyms)


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
# RAG PIPELINE (FIXED)
# ----------------------------
def rag_pipeline(query, retriever):
    expanded = expand_query(query)

    model = get_model()

    # FIX: use correct variable
    query_embedding = model.encode(expanded).astype("float32")

    results, scores = retriever.search(query_embedding, k=5)

    # FIX: FAISS IP similarity (higher = better)
    if len(scores) > 0 and np.max(scores) < 0.45:
        results, scores = retriever.rerank(results, scores)

    context = select_context(results, scores)

    answer = generate_response(query, context)

    return results, scores, context, answer


# ----------------------------
# STREAMLIT UI
# ----------------------------
def main():
    st.set_page_config(
        page_title="AI RAG System",
        page_icon="🇬🇭",
        layout="wide"
    )

    st.title("🇬🇭 AI RAG System - Academic City Project")
    st.write("Ask questions about Ghana election results or budget documents.")

    retriever, chunks = build_vector_store()

    query = st.text_input("Enter your question:")

    if st.button("Ask AI") and query:
        with st.spinner("Thinking..."):
            results, scores, context, answer = rag_pipeline(query, retriever)

        st.subheader("🤖 Answer")
        st.success(answer)

        st.subheader("📄 Retrieved Context")

        for i, (chunk, score) in enumerate(zip(results, scores)):
            st.markdown(f"**Chunk {i+1} | Score: {score:.3f}**")
            st.write(chunk[:300])


# ----------------------------
# RUN APP
# ----------------------------
if __name__ == "__main__":
    main()

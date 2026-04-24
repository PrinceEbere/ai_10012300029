# Student: Prince Ebere Enoch, Index: [Your Index Number]

import streamlit as st
import logging
import faiss
import numpy as np
from src.loader import load_csv, load_pdf
from src.cleaner import clean_text
from src.chunker import chunk_text
from src.embedder import create_embeddings, model as embedder_model
from src.retriever import Retriever
from src.generator import generate_response


# ----------------------------
# PAGE CONFIG
# ----------------------------
st.set_page_config(
    page_title="Academic City AI Assistant",
    page_icon="🎓",
    layout="wide"
)

# ----------------------------
# GLOBAL STYLING
# ----------------------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(180deg, #f7faff 0%, #ffffff 100%);
}
.block-container {
    padding-top: 2rem;
    max-width: 1100px;
}
.stButton>button {
    background: linear-gradient(90deg, #3b82f6, #2563eb);
    color: white;
    border-radius: 12px;
    padding: 0.7rem 1.4rem;
    border: none;
}
.stTextInput input {
    border-radius: 12px !important;
    padding: 0.7rem !important;
}
</style>
""", unsafe_allow_html=True)


# ----------------------------
# LOGGING
# ----------------------------
logging.basicConfig(
    filename='logs/experiment_logs.txt',
    level=logging.INFO,
    format='%(asctime)s - %(message)s'
)


# ----------------------------
# DATA PIPELINE (CACHED)
# ----------------------------
@st.cache_data
def load_and_prepare_data():
    csv_data = load_csv("Data/Ghana_Election_Result.csv")
    pdf_text = load_pdf("Data/2025-Budget.pdf")

    csv_text = csv_data.astype(str).to_csv(index=False)
    combined_text = pdf_text + "\n" + csv_text

    clean = clean_text(combined_text)
    chunks = chunk_text(clean, chunk_size=300, overlap=30)

    # 🔥 MEMORY FIX
    chunks = [c for c in chunks if c.strip()][:200]

    embeddings = create_embeddings(chunks)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings.astype("float32"))

    retriever = Retriever(index, chunks)

    return retriever


retriever = load_and_prepare_data()


# ----------------------------
# RAG FUNCTIONS
# ----------------------------
def expand_query(query):
    return query + " economic policy budget election government"


def select_context(chunks, scores, max_chars=1000):
    ranked = sorted(zip(chunks, scores), key=lambda x: x[1], reverse=True)

    selected = []
    total = 0

    for chunk, score in ranked:
        if total + len(chunk) > max_chars:
            break
        selected.append(chunk)
        total += len(chunk)

    return selected


def rag_pipeline(query):
    expanded = expand_query(query)

    query_embedding = embedder_model.encode(expanded).astype("float32")

    results, scores = retriever.search(query_embedding, k=5)

    context = select_context(results, scores)

    answer = generate_response(query, context)

    return results, scores, answer


# ----------------------------
# HERO SECTION
# ----------------------------
st.markdown("""
<div style='text-align:center; padding:1.5rem 0;'>
    <h1>🎓 Academic City AI Assistant</h1>
    <p style='color:#5b7db1;'>Smart insights on Ghana’s economy, budget, and elections</p>
</div>
""", unsafe_allow_html=True)


# ----------------------------
# INFO BAR
# ----------------------------
c1, c2, c3 = st.columns(3)

c1.markdown("**👤 Student**  \nPrince Ebere Enoch")
c2.markdown("**🆔 Index**  \n10012300029")
c3.markdown("**⚙️ System**  \nRAG + FAISS + LLM")

st.markdown("---")


# ----------------------------
# INPUT SECTION
# ----------------------------
st.markdown("### 💬 Ask a Question")

query = st.text_input(
    "",
    placeholder="e.g. What is Ghana's economic policy?",
    label_visibility="collapsed"
)

ask = st.button("🚀 Generate Answer")


# ----------------------------
# RESPONSE SECTION
# ----------------------------
if ask:
    if query.strip():
        with st.spinner("Thinking..."):
            results, scores, answer = rag_pipeline(query)

        st.markdown("### 🤖 AI Response")

        st.markdown(f"""
        <div style="
            background:white;
            padding:1.2rem;
            border-radius:14px;
            box-shadow:0 8px 20px rgba(0,0,0,0.05);
            border-left:4px solid #3b82f6;
            line-height:1.6;
        ">
        {answer}
        </div>
        """, unsafe_allow_html=True)

        with st.expander("📄 Retrieved Context"):
            for i, (res, score) in enumerate(zip(results, scores)):
                st.markdown(f"""
                <div style="
                    background:#f9fbff;
                    padding:0.8rem;
                    border-radius:10px;
                    margin-bottom:0.6rem;
                ">
                <strong>Chunk {i+1} (Score: {score:.2f})</strong><br>
                {res[:200]}...
                </div>
                """, unsafe_allow_html=True)

        logging.info(f"Query: {query} | Answer: {answer}")

    else:
        st.warning("⚠️ Please enter a question")


# ----------------------------
# FOOTER
# ----------------------------
st.markdown("---")

st.markdown("""
<div style='text-align:center; color:#6b8bb6; font-size:0.9rem;'>
Built with RAG • Academic City Project • 2026
</div>
""", unsafe_allow_html=True)


# ----------------------------
# DETAILS
# ----------------------------
with st.expander("⚙️ System Architecture"):
    st.markdown("""
- Load → Clean → Chunk → Embed → FAISS → Retrieve → Generate  
- Uses Sentence Transformers + Groq LLM  
- Optimized for low memory Streamlit deployment  
""")

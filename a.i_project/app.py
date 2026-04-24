# Student: Prince Ebere Enoch, Index: [Your Index Number]
# CS4241 - Introduction to Artificial Intelligence - 2026

import os
import logging
import faiss
import numpy as np
from src.loader import load_csv, load_pdf
from src.cleaner import clean_text
from src.chunker import chunk_text
from src.embedder import create_embeddings, model as embedder_model
from src.retriever import Retriever
from src.generator import generate_response

os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    filename='logs/experiment_logs.txt',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def load_data():
    csv_data = load_csv("Data/Ghana_Election_Result.csv")
    pdf_text = load_pdf("Data/2025-Budget.pdf")
    logging.info(f"Loaded CSV rows={len(csv_data)}, PDF chars={len(pdf_text)}")
    csv_text = csv_data.astype(str).to_csv(index=False)
    return pdf_text + "\n" + csv_text

def expand_query(query):
    synonyms = ["economic policy", "budget statement", "election results", "government spending"]
    expanded = query + " " + " ".join(synonyms)
    logging.info(f"Expanded query: {expanded}")
    return expanded

def build_prompt(query, context_chunks):
    if context_chunks:
        context = "\n\n".join(context_chunks)
        return (
            "You are an academic assistant for Academic City. "
            "Use ONLY the provided context to answer the question. "
            "If the information is not present, say 'I don't know.'\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {query}"
        )
    return f"Answer the question: {query}"

def select_context(chunks, scores, max_chars=1200):
    ranked = sorted(zip(chunks, scores), key=lambda x: x[1], reverse=True)
    selected = []
    total_chars = 0
    for chunk, score in ranked:
        if total_chars + len(chunk) > max_chars:
            break
        selected.append(chunk)
        total_chars += len(chunk)
    logging.info(f"Selected {len(selected)} context chunks with {total_chars} chars")
    return selected

def rag_pipeline(query, retriever):
    logging.info(f"Query received: {query}")
    expanded_query = expand_query(query)
    query_embedding = embedder_model.encode(expanded_query).astype("float32")
    results, scores = retriever.search(query_embedding, k=5)
    logging.info(f"Retrieved {len(results)} chunks. Scores: {scores.tolist()}")
    if len(scores) > 0 and np.min(scores) < 0.45:
        results, scores = retriever.rerank(results, scores)
        logging.info("Applied re-ranking due to low similarity scores")

    context_chunks = select_context(results, scores, max_chars=1200)
    prompt = build_prompt(query, context_chunks)
    logging.info(f"Final prompt:\n{prompt}")
    answer = generate_response(query, context_chunks)
    logging.info(f"Generated answer: {answer}")
    return results, scores, prompt, answer

def main():
    print("🚀 Starting program...")
    logging.info("Program started.")
    combined_text = load_data()
    clean_text_combined = clean_text(combined_text)
    chunks = chunk_text(clean_text_combined, chunk_size=500, overlap=50)
    chunks = [chunk for chunk in chunks if isinstance(chunk, str) and chunk.strip()]
    logging.info(f"Chunking complete: {len(chunks)} chunks")
    print("✅ Chunking complete")
    print("Valid chunks:", len(chunks))

    embeddings = create_embeddings(chunks)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings.astype('float32'))
    logging.info("Embeddings stored in FAISS index")
    print("✅ Embeddings created")

    retriever = Retriever(index, chunks)

    query = "What is Ghana's economic policy?"
    results, scores, prompt, answer = rag_pipeline(query, retriever)

    print("\n🔎 Query:", query)
    print("\n📄 Retrieved Context and Scores:")
    for i, (res, score) in enumerate(zip(results, scores)):
        print(f"\nChunk {i+1} (Score: {score:.2f}): {res[:200]}")
    print("\n🧠 Final Prompt:")
    print(prompt[:800])
    print("\n🤖 AI Answer:")
    print(answer)

    adversarial_queries = [
        "What about the elections?",
        "Tell me about the budget without mentioning numbers"
    ]
    logging.info("Adversarial testing started")
    for adv_query in adversarial_queries:
        adv_results, adv_scores, adv_prompt, adv_answer = rag_pipeline(adv_query, retriever)
        pure_llm_answer = generate_response(adv_query, [])
        print(f"\nAdversarial Query: {adv_query}")
        print("RAG Answer:", adv_answer)
        print("Pure LLM Answer:", pure_llm_answer)
        logging.info(
            f"Adversarial query '{adv_query}' | RAG answer: {adv_answer} | Pure LLM answer: {pure_llm_answer}"
        )

    if results and np.min(scores) < 0.5:
        logging.warning("Failure case detected: low similarity retrieval scores.")
    print("\n✅ Program completed successfully!")

if __name__ == "__main__":
    main()

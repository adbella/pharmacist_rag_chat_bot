
import os
import json
import logging
import math
import torch
from dotenv import load_dotenv
from retriever import load_embeddings, load_vector_db, load_reranker, build_bm25_retriever, get_ensemble_results, rerank_docs
from generator import build_context

logging.basicConfig(level=logging.INFO)
load_dotenv()

def test_retrieval_and_rerank(query):
    db_path = os.getenv("DB_PATH", "./chroma_db_combined_1771477980")
    print(f"Testing retrieval for: {query}")
    print(f"DB Path: {db_path}")

    embeddings = load_embeddings()
    vector_db = load_vector_db(db_path)
    reranker = load_reranker()
    bm25, kiwi = build_bm25_retriever(db_path)
    
    ensemble_docs = get_ensemble_results(
        query=query,
        kiwi=kiwi,
        bm25_retriever=bm25,
        vector_db=vector_db,
        k=20,
        weight_bm25=0.8,
    )
    
    print(f"\nRetrieved {len(ensemble_docs)} ensemble documents.")
    
    # Reranking
    top_k = 5
    ranked_pairs = rerank_docs(
        query=query,
        docs=ensemble_docs,
        reranker=reranker,
        top_k=top_k
    )
    
    print(f"\nTop {len(ranked_pairs)} re-ranked documents:")
    for i, (score, doc) in enumerate(ranked_pairs):
        source = doc.metadata.get("source", "Unknown")
        relevance = 1.0 / (1.0 + math.exp(-max(min(score, 30.0), -30.0)))
        print(f"[{i+1}] Source: {source} | Score: {score:.4f} | Relevance: {relevance:.4f}")
        print(f"Content: {doc.page_content[:300]}...\n")

if __name__ == "__main__":
    test_retrieval_and_rerank("눈이 침침한데 뭐 먹으면 될까요?")

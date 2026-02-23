
import os
import json
import logging
import math
import torch
import sys
from dotenv import load_dotenv

# Add current directory to path
sys.path.append(os.getcwd())

from retriever import load_embeddings, load_vector_db, load_reranker, build_bm25_retriever, get_ensemble_results, rerank_docs
from generator import build_context

logging.basicConfig(level=logging.INFO)
load_dotenv()

def test_weights_optimized(query):
    db_path = os.getenv("DB_PATH", "./chroma_db_combined_1771477980")
    
    print(f"Loading resources...")
    embeddings = load_embeddings()
    vector_db = load_vector_db(db_path)
    reranker = load_reranker()
    bm25, kiwi = build_bm25_retriever(db_path)
    
    weights = [0.8, 0.5, 0.2]
    
    for w_bm25 in weights:
        print(f"\n[{'='*20}] Weight BM25: {w_bm25} | Weight Vector: {1-w_bm25:.1f} [{'='*20}]")
        
        ensemble_docs = get_ensemble_results(
            query=query,
            kiwi=kiwi,
            bm25_retriever=bm25,
            vector_db=vector_db,
            k=20,
            weight_bm25=w_bm25,
        )
        
        # Reranking
        top_k = 5
        ranked_pairs = rerank_docs(
            query=query,
            docs=ensemble_docs,
            reranker=reranker,
            top_k=top_k
        )
        
        for i, (score, doc) in enumerate(ranked_pairs):
            source = doc.metadata.get("source", "Unknown")
            relevance = 1.0 / (1.0 + math.exp(-max(min(score, 30.0), -30.0)))
            content_snippet = doc.page_content.replace("\n", " ")[:150]
            print(f"[{i+1}] {relevance:.4f} | {source} | {content_snippet}...")

if __name__ == "__main__":
    query = "눈이 침침한데 뭐 먹으면 될까요?"
    test_weights_optimized(query)

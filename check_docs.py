from retriever import load_embeddings, load_vector_db
import os

def check_db():
    db_path = './chroma_db_combined_1771477980'
    print(f"Checking DB at {db_path}...")
    db = load_vector_db(db_path)
    embed = load_embeddings()
    
    query = '타이레놀 이부프로펜 병용'
    print(f"Searching for: {query}")
    docs = db.similarity_search(query, k=10)
    
    for i, d in enumerate(docs, 1):
        source = d.metadata.get("source", "Unknown")
        print(f"[{i}] Source: {source}")
        print(f"Content: {d.page_content[:300]}...\n")

if __name__ == "__main__":
    check_db()

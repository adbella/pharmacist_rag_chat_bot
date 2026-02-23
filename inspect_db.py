import sys
import os
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import torch

# Add current directory to path
sys.path.append(os.getcwd())

from retriever import load_embeddings

def inspect_db(db_path):
    print(f"Inspecting DB at: {db_path}")
    embeddings = load_embeddings()
    vector_db = Chroma(
        persist_directory=db_path,
        embedding_function=embeddings,
    )
    
    collection = vector_db._collection
    count = collection.count()
    print(f"Total documents: {count}")
    
    # Peek at first 20 documents
    results = collection.get(limit=20, include=["documents", "metadatas"])
    
    for i in range(len(results["ids"])):
        print(f"\n--- Document {i+1} ---")
        print(f"ID: {results['ids'][i]}")
        print(f"Metadata: {results['metadatas'][i]}")
        content = results['documents'][i]
        try:
            # Ensure output is UTF-8
            sys.stdout.reconfigure(encoding='utf-8')
        except:
            pass
        print(f"Content length: {len(content)}")
        print(f"Content preview:\n{content}")

if __name__ == "__main__":
    DB_PATH = "./chroma_db_combined_1771477980"
    inspect_db(DB_PATH)

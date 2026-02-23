
import sys
import os
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

# Add current directory to path
sys.path.append(os.getcwd())
from retriever import load_embeddings

load_dotenv()

def search_db(db_path, query_keyword):
    print(f"Searching for '{query_keyword}' in DB at: {db_path}")
    embeddings = load_embeddings()
    vector_db = Chroma(
        persist_directory=db_path,
        embedding_function=embeddings,
    )
    
    collection = vector_db._collection
    # Use 'where_document' to find exact string match in content
    results = collection.get(
        where_document={"$contains": query_keyword},
        limit=10, 
        include=["documents", "metadatas"]
    )
    
    count = len(results["ids"])
    print(f"Found {count} documents containing {query_keyword}")
    
    for i in range(count):
        print(f"\n--- Document {i+1} ---")
        print(f"Metadata: {results['metadatas'][i]}")
        print(f"Content preview: {results['documents'][i][:200]}...")

if __name__ == "__main__":
    DB_PATH = "./chroma_db_combined_1771477980"
    search_db(DB_PATH, "오메가3")
    print("-" * 50)
    search_db(DB_PATH, "오메가-3")

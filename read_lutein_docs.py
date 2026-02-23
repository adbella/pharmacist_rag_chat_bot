
import os
import sys
from langchain_chroma import Chroma
from dotenv import load_dotenv

# Add current directory to path
sys.path.append(os.getcwd())
from retriever import load_embeddings

load_dotenv()

def read_lutein_docs(db_path):
    print(f"Reading Lutein documents from: {db_path}")
    embeddings = load_embeddings()
    vector_db = Chroma(
        persist_directory=db_path,
        embedding_function=embeddings,
    )
    
    collection = vector_db._collection
    results = collection.get(
        where_document={"$contains": "루테인"},
        limit=5, 
        include=["documents", "metadatas"]
    )
    
    for i, (content, meta) in enumerate(zip(results["documents"], results["metadatas"])):
        print(f"\n--- Document {i+1} ---")
        print(f"Source: {meta}")
        print(f"Content:\n{content}")

if __name__ == "__main__":
    DB_PATH = "./chroma_db_combined_1771477980"
    read_lutein_docs(DB_PATH)

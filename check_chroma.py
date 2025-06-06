import sys
from pathlib import Path
from typing import List, Dict, Any, Union

# Ensure src/ is on the path for project-level imports
# Adjust PROJECT_ROOT based on whether check_chroma.py is in root or src
# If check_chroma.py is in the project root, PROJECT_ROOT is just Path(__file__).parent
PROJECT_ROOT = Path(__file__).parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
    print(f"Added {SRC_DIR} to sys.path for module imports.")

# Import ChromaDB client functions
from chroma_db.chroma_client import get_chroma_client, get_or_create_collection

# LangChain Imports - Using langchain_community as recommended
from langchain_community.vectorstores import Chroma 
from langchain_community.embeddings import HuggingFaceBgeEmbeddings 

import torch
import os # Make sure os is imported for environment variables if needed by get_chroma_client

# Configuration - Ensure these match your embed_chunks.py and test_case_generator.py
COLLECTION_NAME = "code_chunks_collection"
EMBEDDING_MODEL_NAME_BGE = "BAAI/bge-small-en-v1.5"
DEVICE_FOR_EMBEDDINGS = "cuda" if torch.cuda.is_available() else "cpu"

def check_chroma_content(query_text: str):
    print(f"--- Checking ChromaDB for query: '{query_text}' ---")

    # Initialize Embedding Model
    print(f"Loading embedding model: {EMBEDDING_MODEL_NAME_BGE} on {DEVICE_FOR_EMBEDDINGS}...")
    embeddings = HuggingFaceBgeEmbeddings(
        model_name=EMBEDDING_MODEL_NAME_BGE,
        model_kwargs={'device': DEVICE_FOR_EMBEDDINGS},
        encode_kwargs={'normalize_embeddings': True}
    )
    print("Embedding model loaded.")

    # Connect to ChromaDB
    print(f"Connecting to ChromaDB collection: {COLLECTION_NAME}...")
    chroma_client = get_chroma_client() # This should load the existing client with persistence
    vectorstore = Chroma(
        client=chroma_client,
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings
    )
    print("ChromaDB vectorstore instantiated.")

    # Perform a direct similarity search
    print("Performing similarity search...")
    # You can adjust k here if you want to see more results
    retrieved_docs = vectorstore.similarity_search(query_text, k=10) 

    if not retrieved_docs:
        print("\n[!] No documents retrieved from ChromaDB for this query.")
        print("    This likely means the relevant chunks were not successfully ingested or are not semantically close enough.")
    else:
        print(f"\n[✓] Retrieved {len(retrieved_docs)} documents:")
        for i, doc in enumerate(retrieved_docs):
            print(f"\n--- Document {i+1} ---")
            print(f"Source: {doc.metadata.get('source', 'N/A')}")
            print(f"Start Line: {doc.metadata.get('start_line', 'N/A')}")
            print(f"End Line: {doc.metadata.get('end_line', 'N/A')}")
            print("Content (first 500 chars):")
            print(doc.page_content[:500] + ("..." if len(doc.page_content) > 500 else ""))
            print("------------------------")
    
    print("\n--- ChromaDB Check Complete ---")

if __name__ == '__main__':
    # Use the exact query you are using in test_case_generator.py
    query = "Write a JUnit 5 test for the `BengGenservice`'s encryption() method"
    check_chroma_content(query)

    # You can also try a more direct query like just the method signature
    # query_direct = "public void encryption() throws NoSuchAlgorithmException, NoSuchPaddingException"
    # print("\n--- Trying a more direct query (uncomment to test) ---")
    # # check_chroma_content(query_direct)
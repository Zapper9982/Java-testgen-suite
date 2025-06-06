import sys
from pathlib import Path

# Ensure src/ is on the path for project-level imports
PROJECT_ROOT = Path(__file__).parent # Assuming inspect_chroma.py is in project root
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
    print(f"Added {SRC_DIR} to sys.path for module imports.")

from chroma_db.chroma_client import get_chroma_client, get_or_create_collection
from langchain_community.vectorstores import Chroma # Use langchain_community
from langchain_community.embeddings import HuggingFaceBgeEmbeddings # Use langchain_community

import torch

COLLECTION_NAME = "code_chunks_collection"
EMBEDDING_MODEL_NAME_BGE = "BAAI/bge-small-en-v1.5"
DEVICE_FOR_EMBEDDINGS = "cuda" if torch.cuda.is_available() else "cpu"

def inspect_chroma_metadata():
    print(f"--- Inspecting ChromaDB Collection: {COLLECTION_NAME} ---")

    # Initialize Embedding Model (needed for Chroma LangChain integration)
    embeddings = HuggingFaceBgeEmbeddings(
        model_name=EMBEDDING_MODEL_NAME_BGE,
        model_kwargs={'device': DEVICE_FOR_EMBEDDINGS},
        encode_kwargs={'normalize_embeddings': True}
    )

    chroma_client = get_chroma_client()
    collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME)

    print(f"Total documents in collection '{COLLECTION_NAME}': {collection.count()}")

    # Retrieve a few documents to inspect their metadata
    # We'll fetch all metadata for up to 100 documents to see what's really there.
    # If your collection is very large, consider a smaller peek or a proper query.
    peek_ids = []
    try:
        peek_ids = collection.peek(100)['ids'] # Get first 100 IDs
    except Exception as e:
        print(f"Error peeking into collection (might be empty): {e}")
        return # Exit if collection is empty or inaccessible


    results = collection.get(
        ids=peek_ids,
        include=['metadatas'] # Ask to include metadata
    )

    filenames_found = set()
    print("\n--- Sample Metadatas ---")
    if results and results['metadatas']:
        for i, metadata in enumerate(results['metadatas']):
            if i < 10: # Print details for first 10 for brevity
                print(f"Document {i+1} Metadata: {metadata}")
            if 'filename' in metadata:
                filenames_found.add(metadata['filename'])
    else:
        print("No documents or metadata found in the collection.")

    print("\n--- All Unique Filenames Found in Collection ---")
    if filenames_found:
        for fname in sorted(list(filenames_found)):
            print(f"- '{fname}'") # Added quotes to show exact string
    else:
        print("No 'filename' metadata found in retrieved documents.")

    print("\n--- Inspection Complete ---")

if __name__ == '__main__':
    inspect_chroma_metadata()
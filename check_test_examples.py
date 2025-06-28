#!/usr/bin/env python3

import sys
from pathlib import Path

# Add the src directory to sys.path
TESTGEN_AUTOMATION_ROOT = Path(__file__).parent
TESTGEN_AUTOMATION_SRC_DIR = TESTGEN_AUTOMATION_ROOT / "src"
if str(TESTGEN_AUTOMATION_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(TESTGEN_AUTOMATION_SRC_DIR))

from chroma_db.chroma_client import get_chroma_client
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import torch

print("Checking test examples collection...")

# Initialize embeddings
if torch.backends.mps.is_available():
    DEVICE_FOR_EMBEDDINGS = "mps"
else:
    DEVICE_FOR_EMBEDDINGS = "cpu"

embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5",
    encode_kwargs={'normalize_embeddings': True}
)

# Get ChromaDB client
chroma_client = get_chroma_client()

# Check if test examples collection exists
test_examples_collection_name = "test_examples_collection"
collections = chroma_client.list_collections()
collection_names = [col.name for col in collections]

print(f"Available collections: {collection_names}")

if test_examples_collection_name in collection_names:
    print(f"\n--- Found test examples collection: {test_examples_collection_name} ---")
    
    # Get the collection
    test_examples_vectorstore = Chroma(
        client=chroma_client,
        collection_name=test_examples_collection_name,
        embedding_function=embeddings
    )
    
    # Count documents
    count = test_examples_vectorstore._collection.count()
    print(f"Total documents in test examples collection: {count}")
    
    if count > 0:
        # Get a sample document
        sample = test_examples_vectorstore._collection.get(limit=1)
        print(f"\n--- Sample document metadata ---")
        if sample['metadatas']:
            print(f"Metadata: {sample['metadatas'][0]}")
        if sample['documents']:
            doc_content = sample['documents'][0]
            print(f"Document preview: {doc_content[:200]}...")
    else:
        print("Test examples collection is empty!")
        
else:
    print(f"\n--- Test examples collection '{test_examples_collection_name}' NOT FOUND ---")
    print("This is why RAG is hanging - the collection doesn't exist!")
    print("You need to index test examples first using the index_test_examples.py script.") 
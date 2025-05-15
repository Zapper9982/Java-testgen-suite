# testgen-automation/src/chroma_db/chroma_client.py
"""
ChromaDB Client Setup (v0.4+)
Using PersistentClient for on-disk persistence.
"""
from chromadb import PersistentClient

# Change this path as needed (will be created if missing)
PERSIST_DIRECTORY = "testgen-automation/data/chroma_db"

# No Settings import needed for new PersistentClient

def get_chroma_client():
    """Return a persistent ChromaDB client instance."""
    # Initialize a PersistentClient pointing at PERSIST_DIRECTORY
    return PersistentClient(path=PERSIST_DIRECTORY)


def get_or_create_collection(client, name: str = "code_chunks"):
    """
    Retrieve an existing collection or create a new one.
    """
    return client.get_or_create_collection(
        name=name,
        metadata={"description": "SpringBoot code chunks"}
    )
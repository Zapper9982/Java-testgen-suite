from chromadb import PersistentClient

PERSIST_DIRECTORY = "chromadata/data/chroma_db"

def get_chroma_client():
    """Return a persistent ChromaDB client instance."""
    return PersistentClient(path=PERSIST_DIRECTORY)


def get_or_create_collection(client, name: str = "code_chunks"):
    return client.get_or_create_collection(
        name=name,
        metadata={"description": "SpringBoot code chunks"}
    )
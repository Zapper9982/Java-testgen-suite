import yaml
from chromadb import Client

class ChromaClient:
    def __init__(self):
        cfg = yaml.safe_load(open('config/chroma_config.yaml'))
        self.client = Client(path=cfg['db_path'])
        self.col = self.client.get_or_create_collection(cfg['collection_name'])

    def add(self, ids, embeddings, metadatas, documents):
        self.col.add(ids=ids, embeddings=embeddings, metadatas=metadatas, documents=documents)

    def query(self, query_emb, n_results=5):
        res = self.col.query(query_embeddings=[query_emb], n_results=n_results)
        return res
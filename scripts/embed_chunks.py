#!/usr/bin/env python3
import os
import sys
import json
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModel

# ensure src/ is on the path
PROJECT_ROOT = Path(__file__).parent.parent
SRC_DIR      = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from chroma_db.chroma_client import get_chroma_client, get_or_create_collection

# ─── Configuration ─────────────────────────────────────────────────────────────
CHUNKS_JSON = PROJECT_ROOT / "chunked_output" / "langchain_chunks.json"
MODEL_NAME  = "BAAI/bge-small-en-v1.5"
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"

# ─── Load the model ─────────────────────────────────────────────────────────────
print(f"Loading embedding model {MODEL_NAME} on {DEVICE}…")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model     = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE)
model.eval()

def embed_text(text: str, max_retries: int = 3, backoff: float = 1.0):
    """Encode text to a vector with retry/back‑off on OOM or HTTP errors."""
    for attempt in range(1, max_retries + 1):
        try:
            inputs = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=512,
            ).to(DEVICE)
            with torch.no_grad():
                outputs = model(**inputs, return_dict=True)
            # mean‐pool
            embeddings = outputs.last_hidden_state  # (1, L, D)
            mask       = inputs.attention_mask.unsqueeze(-1)
            summed     = (embeddings * mask).sum(1)
            counts     = mask.sum(1)
            vector     = (summed / counts).squeeze().cpu().tolist()
            return vector
        except RuntimeError as e:
            # catch CUDA OOM or other runtime errors
            if attempt < max_retries:
                wait = backoff * attempt
                print(f"[WARN] Embed failure ({e}), retrying in {wait}s…")
                time.sleep(wait)
                continue
            else:
                raise

def main():
    if not CHUNKS_JSON.exists():
        print(f"Chunks file not found: {CHUNKS_JSON}")
        return

    chunks = json.loads(CHUNKS_JSON.read_text(encoding="utf-8"))
    client     = get_chroma_client()
    collection = get_or_create_collection(client)

    for entry in chunks:
        meta = entry["metadata"]
        text = entry["text"]
        emb  = embed_text(text)

        uid = f"{meta['source_path']}::{meta['chunk_index']}"
        collection.add(
            ids=[uid],
            embeddings=[emb],
            metadatas=[meta],
            documents=[text],
        )

    # client.persist()
    print(f"[✓] Ingested {len(chunks)} chunks with HuggingFace embeddings.")

if __name__ == "__main__":
    main()

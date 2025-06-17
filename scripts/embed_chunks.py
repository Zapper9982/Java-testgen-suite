#!/usr/bin/env python3
import os
import sys
import json
import time
from pathlib import Path
import hashlib # Import hashlib for content hashing

import torch
from transformers import AutoTokenizer, AutoModel

# ensure src/ is on the path
PROJECT_ROOT = Path(__file__).parent.parent
SRC_DIR      = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# Assuming chroma_db.chroma_client is in the src directory
try:
    from chroma_db.chroma_client import get_chroma_client, get_or_create_collection
except ImportError:
    print("Error: Could not import chroma_client. Make sure 'src/chroma_db/chroma_client.py' exists and is correctly configured.")
    sys.exit(1)

# ─── Configuration ─────────────────────────────────────────────────────────────
# Updated to match the output filename from the last chunker script
CHUNKS_JSON = PROJECT_ROOT / "chunked_output" / "chunks.json"
MODEL_NAME  = "BAAI/bge-small-en-v1.5" # Excellent choice for general-purpose embeddings
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"

# ─── Load the model ─────────────────────────────────────────────────────────────
print(f"Loading embedding model {MODEL_NAME} on {DEVICE}…")
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model     = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE)
    model.eval()
except Exception as e:
    print(f"Error loading model {MODEL_NAME}: {e}")
    print("Please ensure you have an active internet connection or the model is cached locally.")
    sys.exit(1)


def embed_text(text: str, max_retries: int = 3, backoff: float = 1.0):
    """Encode text to a vector with retry/back‑off on OOM or HTTP errors."""
    if not text or not text.strip(): # Handle empty or whitespace-only strings
        # Return a zero vector of the correct dimensionality (model.config.hidden_size)
        # BGE-small-en-v1.5 has a hidden_size of 768
        return [0.0] * model.config.hidden_size 

    for attempt in range(1, max_retries + 1):
        try:
            inputs = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=512, # BGE models typically use 512 as max length
            ).to(DEVICE)
            with torch.no_grad():
                outputs = model(**inputs, return_dict=True)
            
            # Mean-pooling: Take the average of the last hidden states,
            # masked by the attention mask to ignore padding tokens.
            embeddings = outputs.last_hidden_state  # (batch_size, sequence_length, hidden_size)
            mask       = inputs.attention_mask.unsqueeze(-1).expand(embeddings.size()).float()
            
            # Sum embeddings multiplied by mask, then divide by the sum of mask (number of non-padded tokens)
            summed_embeddings = (embeddings * mask).sum(1)
            num_tokens = mask.sum(1)
            
            # Avoid division by zero for empty inputs (though handled by strip() check)
            num_tokens = torch.clamp(num_tokens, min=1e-9) # Ensure at least a tiny value
            
            vector = (summed_embeddings / num_tokens).squeeze().cpu().tolist()
            return vector
        except RuntimeError as e:
            if "CUDA out of memory" in str(e) and DEVICE == "cuda":
                print(f"[WARN] CUDA OOM, consider reducing batch size or using CPU for next attempt...")
                # If processing in batches, you might want to try a smaller batch here
            if attempt < max_retries:
                wait = backoff * attempt
                print(f"[WARN] Embed failure ({e}), retrying in {wait}s… (Attempt {attempt}/{max_retries})")
                time.sleep(wait)
                continue
            else:
                print(f"[ERROR] Failed to embed text after {max_retries} attempts: {e}")
                raise # Re-raise the exception if all retries fail
        except Exception as e:
            print(f"[ERROR] An unexpected error occurred during embedding: {e}")
            raise # Re-raise unexpected exceptions


def main():
    if not CHUNKS_JSON.exists():
        print(f"Error: Chunks file not found at {CHUNKS_JSON}")
        print("Please ensure your chunker script has run successfully and outputs to this path.")
        return

    try:
        chunks = json.loads(CHUNKS_JSON.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from {CHUNKS_JSON}: {e}")
        print("The chunks file might be corrupted or malformed.")
        return

    if not chunks:
        print("No chunks found in the JSON file. Nothing to embed.")
        return

    client     = get_chroma_client()
    # collection = get_or_create_collection(client)
    collection = get_or_create_collection(client, name="code_chunks_collection")

    # Process chunks in batches (optional, but good for performance)
    batch_size = 128
    num_chunks = len(chunks)
    
    print(f"Starting ingestion of {num_chunks} chunks into ChromaDB...")

    # Dictionary to keep track of indices per file path for sequential chunking
    file_chunk_counters = {}

    for i in range(0, num_chunks, batch_size):
        batch_chunks = chunks[i:i + batch_size]
        
        batch_ids = []
        batch_embeddings = []
        batch_metadatas = []
        batch_documents = []

        for entry_idx_in_batch, entry in enumerate(batch_chunks):
            # Ensure 'chunk_content' and 'chunk_metadata' keys exist from the chunker
            if "chunk_content" not in entry or "chunk_metadata" not in entry:
                print(f"Warning: Skipping malformed chunk entry: {entry}")
                continue

            meta = entry["chunk_metadata"]
            text = entry["chunk_content"]

            # Base ID from the original file path (relative path from processed_output)
            base_filepath_id = meta.get('filepath_txt', 'unknown_path').replace(str(PROJECT_ROOT / "processed_output"), "")
            
            # Ensure a unique identifier for each chunk within its file
            chunk_specific_id = ""
            chunk_type = meta.get('type')

            if chunk_type in ['java_class_header', 'java_method', 'java_class_header_sub_chunk', 'java_method_sub_chunk']:
                # For Java, use class_name, method_name, and sub_chunk_index if present
                class_name = meta.get('class_name', 'NoClass')
                if chunk_type.startswith('java_method'):
                    method_name = meta.get('method_name', 'NoMethod')
                    chunk_specific_id = f"{class_name}::{method_name}"
                else: # java_class_header
                    chunk_specific_id = class_name
                
                # Append sub_chunk_index for semantic sub-chunks if it exists
                if 'sub_chunk_index' in meta:
                    chunk_specific_id += f"::sub_chunk::{meta['sub_chunk_index']}"
                elif 'start_char' in meta: # Use character offset for method uniqueness within a class
                     chunk_specific_id += f"::offset::{meta['start_char']}"

            elif chunk_type.startswith('config'):
                # For config, use a line/entry index or a hash of the content to ensure uniqueness
                # Initialize counter for this file if not present
                if base_filepath_id not in file_chunk_counters:
                    file_chunk_counters[base_filepath_id] = 0
                else:
                    file_chunk_counters[base_filepath_id] += 1
                chunk_specific_id = f"config_entry::{file_chunk_counters[base_filepath_id]}"
                
            elif chunk_type == 'unclassified_text_chunk':
                # For generic text chunks, use the sub_chunk_index
                if 'sub_chunk_index' in meta:
                    chunk_specific_id = f"text_chunk::{meta['sub_chunk_index']}"
                else:
                    # Fallback for truly generic chunks without sub_chunk_index
                    # Use a hash of the content itself, less ideal but ensures uniqueness
                    chunk_specific_id = f"text_chunk_hash::{hashlib.sha256(text.encode('utf-8')).hexdigest()[:8]}"
            else:
                # Fallback for any other unexpected chunk types or missing identifiers
                # Use a combination of entry_idx_in_batch and overall chunk index 'i'
                chunk_specific_id = f"generic_chunk::{i + entry_idx_in_batch}"


            # Combine all parts to form the final unique ID
            uid = f"{base_filepath_id}::{chunk_type}::{chunk_specific_id}".replace("::", "__") # Replace :: with __ for readability in some systems, or keep ::
            uid = uid.replace(" ", "_").strip() # Clean up spaces in filename/path parts

            try:
                emb = embed_text(text)
            except Exception as e:
                print(f"[ERROR] Skipping chunk due to embedding failure for ID '{uid}': {e}")
                continue # Skip this chunk if embedding fails

            batch_ids.append(uid)
            batch_embeddings.append(emb)
            batch_metadatas.append(meta)
            batch_documents.append(text)
        
        if batch_ids: # Only add if the batch is not empty
            try:
                collection.add(
                    ids=batch_ids,
                    embeddings=batch_embeddings,
                    metadatas=batch_metadatas,
                    documents=batch_documents,
                )
                print(f"Ingested {len(batch_ids)} chunks (batch {i//batch_size + 1}/{(num_chunks + batch_size - 1)//batch_size})...")
            except Exception as e:
                print(f"[ERROR] Failed to add batch to ChromaDB: {e}")
                # You might want to log which IDs failed or retry the batch
                continue # Continue to the next batch even if one fails


    # client.persist() # If using a persistent ChromaDB client, uncomment this.
                       # The client used by get_chroma_client might handle persistence automatically
                       # depending on its configuration (e.g., if it's a DuckDB-based client).
    print(f"[✓] Ingested {num_chunks} chunks with HuggingFace embeddings into ChromaDB.")

if __name__ == "__main__":
    main()
#!/usr/bin/env bash
set -e

# 1) Pre‑req: have processed_output/ from previous step
# 2) Chunk
python3 scripts/chunker.py

# 3) Embed & ingest
python3 scripts/embed_chunks.py

echo "[Pipeline] Chunks embedded — ready for RAG queries."

#!/usr/bin/env bash
set -e

python3 pre-processing/processing.py

python3 scripts/chunker.py

python3 scripts/embed_chunks.py

echo "[Pipeline] Chunks embedded — ready for RAG queries."

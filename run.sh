#!/usr/bin/env bash
set -e

export BUILD_TOOL="maven"
echo "[Pipeline] Starting preprocessing..."
python3 pre-processing/processing.py
echo "[Pipeline] Starting Spring Boot application analysis..."
python3 src/analyzer/code_analyzer.py
echo "[Pipeline] Starting code chunking..."
python3 scripts/chunker.py
echo "[Pipeline] Starting embedding chunks into ChromaDB..."
python3 scripts/embed_chunks.py
echo "[Pipeline] Starting test case generation with feedback loop..."
python3 src/llm/test_case_generator.py
echo "[Pipeline] All steps completed. Test generation pipeline finished."
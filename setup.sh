#!/usr/bin/env bash
set -e

# Detect OS
OS_TYPE="$(uname -s 2>/dev/null || echo Windows)"

if [[ "$OS_TYPE" == "Darwin" || "$OS_TYPE" == "Linux" ]]; then
  echo "[Setup] Detected macOS/Linux."
  PYTHON_CMD="python3"
  PIP_CMD="pip3"
else
  echo "[Setup] Detected Windows."
  PYTHON_CMD="python"
  PIP_CMD="pip"
fi

# 1. Install Python dependencies (with protobuf pin for ChromaDB)
echo "[Setup] Installing Python dependencies..."
$PIP_CMD install --upgrade pip
$PIP_CMD install -r requirements.txt

# 2. Compile Java bridge if needed (javabridge/JavaParserBridge.class)
echo "[Setup] Compiling Java bridge if needed..."
if [ ! -f javabridge/JavaParserBridge.class ]; then
  javac -cp lib/javaparser-core-3.25.4.jar javabridge/JavaParserBridge.java
fi

# 3. Run interactive configuration (Spring Boot path, LLM, etc)
echo "[Setup] Running interactive configuration..."
$PYTHON_CMD configure_llm.py

echo "[Setup] Setup complete! You can now run: bash run.sh"

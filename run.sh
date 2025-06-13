#!/usr/bin/env bash
set -e # Exit immediately if a command exits with a non-zero status.

# --- Configuration ---
# REQUIRED: Set the absolute path to your Spring Boot project
export SPRING_BOOT_PROJECT_ROOT="/path/to/your/spring-boot-project"

# REQUIRED: Set your Google API Key for the LLM
export GOOGLE_API_KEY="YOUR_GOOGLE_API_KEY"

# OPTIONAL: Override default settings from src/main.py
# export MAX_ITERATIONS=3       # Default is 5
# export TARGET_COVERAGE=0.85   # Default is 0.9 (90%)
# export BUILD_TOOL="gradle"    # Default is "maven"

# --- Sanity Checks ---
if [ -z "$SPRING_BOOT_PROJECT_ROOT" ] || [ "$SPRING_BOOT_PROJECT_ROOT" == "/path/to/your/spring-boot-project" ]; then
  echo "ERROR: SPRING_BOOT_PROJECT_ROOT is not set or is still the placeholder value."
  echo "Please edit run.sh and set it to the absolute path of your Spring Boot project."
  exit 1
fi

if [ -z "$GOOGLE_API_KEY" ] || [ "$GOOGLE_API_KEY" == "YOUR_GOOGLE_API_KEY" ]; then
  echo "ERROR: GOOGLE_API_KEY is not set or is still the placeholder value."
  echo "Please edit run.sh and set your Google API key."
  exit 1
fi

if [ ! -d "$SPRING_BOOT_PROJECT_ROOT" ]; then
  echo "ERROR: SPRING_BOOT_PROJECT_ROOT directory does not exist: $SPRING_BOOT_PROJECT_ROOT"
  exit 1
fi

# --- Run the main pipeline ---
echo "[Pipeline] Starting Java Test Generation Suite via src/main.py..."
echo "Spring Boot Project: $SPRING_BOOT_PROJECT_ROOT"
echo "Build Tool: ${BUILD_TOOL:-maven}" # Print default if not set
echo "Max Iterations: ${MAX_ITERATIONS:-5}"
echo "Target Coverage: ${TARGET_COVERAGE:-0.9}"

# Ensure src/main.py is executable or called with python
if [ -f "src/main.py" ]; then
  python3 src/main.py
else
  echo "ERROR: src/main.py not found!"
  exit 1
fi

echo "[Pipeline] Execution finished."

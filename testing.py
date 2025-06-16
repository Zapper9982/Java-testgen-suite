import os
from pathlib import Path
import sys
from typing import Any, Dict, List

TESTGEN_AUTOMATION_ROOT = Path(__file__).parent

PROCESSED_OUTPUT_DIR = TESTGEN_AUTOMATION_ROOT / 'processed_output'

def find_largest_files(directory: Path, num_files_to_report: int = 10) -> List[Dict[str, Any]]:
 
    if not directory.exists() or not directory.is_dir():
        print(f"Error: Directory not found: {directory}")
        return []

    print(f"Scanning '{directory}' for file sizes...")
    file_sizes = []

    for root, _, files in os.walk(directory):
        for file_name in files:
            file_path = Path(root) / file_name
            try:
              
                content = file_path.read_text(encoding='utf-8')
                char_count = len(content)
                file_sizes.append({"filepath": str(file_path), "char_count": char_count})
            except Exception as e:
                print(f"Warning: Could not read file {file_path}: {e}")

    file_sizes.sort(key=lambda x: x['char_count'], reverse=True)

    return file_sizes[:num_files_to_report]

if __name__ == "__main__":
    print("--- Finding Largest Files in Processed Output ---")
    
    TOP_N_FILES = 15 

    largest_files = find_largest_files(PROCESSED_OUTPUT_DIR, TOP_N_FILES)

    if largest_files:
        print(f"\nTop {len(largest_files)} largest files by character count in '{PROCESSED_OUTPUT_DIR}':")
        for i, file_info in enumerate(largest_files):
            print(f"{i+1}. File: {file_info['filepath']} | Characters: {file_info['char_count']}")
    else:
        print("No files found or an error occurred during scanning.")

    print("\n----------------------------------------------")
    print("Use this information to adjust chunking thresholds in scripts/chunker.py:")
    print(f"- JAVA_FILE_MAX_CHARS_FOR_SINGLE_CHUNK (current: 4000)")
    print(f"- LARGE_JAVA_CHUNK_THRESHOLD (current: 5000)")
    print(f"- METHOD_GROUP_MAX_CHARS (current: 2000)")
    print("Consider your LLM's context window size when setting these.")

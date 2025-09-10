import os
import sys
from pathlib import Path
import json
from typing import List, Dict, Any, Union

from dotenv import load_dotenv
load_dotenv()

#imports
TESTGEN_AUTOMATION_ROOT = Path(__file__).parent.parent.parent 
SPRING_BOOT_PROJECT_ROOT = os.getenv("SPRING_BOOT_PROJECT_PATH")
SPRING_BOOT_MAIN_JAVA_DIR = Path(SPRING_BOOT_PROJECT_ROOT) / "src" / "main" / "java"
PROCESSED_OUTPUT_ROOT = TESTGEN_AUTOMATION_ROOT / "processed_output"
TESTGEN_AUTOMATION_SRC_DIR = TESTGEN_AUTOMATION_ROOT / "src"


if str(TESTGEN_AUTOMATION_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(TESTGEN_AUTOMATION_SRC_DIR))
    print(f"Added {TESTGEN_AUTOMATION_SRC_DIR} to sys.path for internal module imports.")

from analyzer.code_analysis_utils import SpringBootAnalyser 

ANALYSIS_RESULTS_DIR = TESTGEN_AUTOMATION_ROOT / "analysis_results"
ANALYSIS_RESULTS_FILE = ANALYSIS_RESULTS_DIR / "spring_boot_targets.json"

def main():

    print("Starting Spring Boot application analysis for test generation...")
    
    ANALYSIS_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    spring_boot_analyser = SpringBootAnalyser(
        project_main_java_dir=SPRING_BOOT_MAIN_JAVA_DIR,
        processed_output_root=PROCESSED_OUTPUT_ROOT
    )

    #calling function from code_analysis_utils yo
    discovered_targets_metadata = spring_boot_analyser.discover_targets()
    print(f"\nDiscovered {len(discovered_targets_metadata)} Spring Boot Service/Controller targets.")

    if not discovered_targets_metadata:
        print("No Spring Boot Service or Controller targets found. Exiting analysis.")
        sys.exit(0)

    # preparing data for serialisation 
    serializable_targets = []
    for target in discovered_targets_metadata:
        serializable_target = target.copy() 
        if isinstance(serializable_target.get('java_file_path_abs'), Path):
            serializable_target['java_file_path_abs'] = str(serializable_target['java_file_path_abs'])
        serializable_targets.append(serializable_target)

    # saves  the discovered targets metadata to a JSON file
    try:
        with open(ANALYSIS_RESULTS_FILE, 'w', encoding='utf-8') as f:
            json.dump(serializable_targets, f, indent=4)
        print(f"[SUCCESS] Discovered Spring Boot targets saved to: {ANALYSIS_RESULTS_FILE}")
    except Exception as e:
        print(f"[ERROR] Could not save analysis results to '{ANALYSIS_RESULTS_FILE}': {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()


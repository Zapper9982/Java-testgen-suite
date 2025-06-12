import sys
from pathlib import Path
import json
from typing import List, Dict, Any, Union

# Define the root of your testgen-automation project 
# If script is at 'testgen-automation/src/analyzer/code_analyzer.py':
# Path(__file__).parent is '.../src/analyzer/'
# Path(__file__).parent.parent is '.../src/'
# Path(__file__).parent.parent.parent is '.../testgen-automation/' (This is the correct root)
TESTGEN_AUTOMATION_ROOT = Path(__file__).parent.parent.parent 

# Define the root of your Spring Boot project.
# This path is crucial for SpringBootAnalyser to locate source files.
SPRING_BOOT_PROJECT_ROOT = Path("/Users/tanmay/Desktop/AMRIT/BeneficiaryID-Generation-API")
SPRING_BOOT_MAIN_JAVA_DIR = SPRING_BOOT_PROJECT_ROOT / "src" / "main" / "java"

# The 'processed_output' directory is assumed to be directly under TESTGEN_AUTOMATION_ROOT
PROCESSED_OUTPUT_ROOT = TESTGEN_AUTOMATION_ROOT / "processed_output"

# Add the 'src' directory of testgen-automation to sys.path
# This makes modules directly under 'src' (like analyzer/code_analysis_utils.py) discoverable
# by using absolute imports like 'from analyzer.code_analysis_utils import ...'
TESTGEN_AUTOMATION_SRC_DIR = TESTGEN_AUTOMATION_ROOT / "src"
if str(TESTGEN_AUTOMATION_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(TESTGEN_AUTOMATION_SRC_DIR))
    print(f"Added {TESTGEN_AUTOMATION_SRC_DIR} to sys.path for internal module imports.")

# --- CRITICAL FIX: Use absolute import relative to 'src' ---
# This assumes 'code_analysis_utils.py' is located at 'src/analyzer/code_analysis_utils.py'
# and 'src/' is in sys.path AND 'src/analyzer/__init__.py' exists.
from analyzer.code_analysis_utils import SpringBootAnalyser 


# Define the output directory and file for the discovered targets metadata
ANALYSIS_RESULTS_DIR = TESTGEN_AUTOMATION_ROOT / "analysis_results"
ANALYSIS_RESULTS_FILE = ANALYSIS_RESULTS_DIR / "spring_boot_targets.json"

def main():
    """
    Main function to discover Spring Boot targets and save their metadata to a JSON file.
    """
    print("Starting Spring Boot application analysis for test generation...")
    
    # Ensure the analysis results directory exists
    ANALYSIS_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Initialize SpringBootAnalyser
    spring_boot_analyser = SpringBootAnalyser(
        project_main_java_dir=SPRING_BOOT_MAIN_JAVA_DIR,
        processed_output_root=PROCESSED_OUTPUT_ROOT
    )

    # Discover all relevant Service/Controller targets in the project
    discovered_targets_metadata = spring_boot_analyser.discover_targets()
    print(f"\nDiscovered {len(discovered_targets_metadata)} Spring Boot Service/Controller targets.")

    if not discovered_targets_metadata:
        print("No Spring Boot Service or Controller targets found. Exiting analysis.")
        sys.exit(0)

    # Prepare data for JSON serialization (convert Path objects to strings)
    serializable_targets = []
    for target in discovered_targets_metadata:
        serializable_target = target.copy() # Create a shallow copy
        # Convert Path object to string for JSON serialization
        if isinstance(serializable_target.get('java_file_path_abs'), Path):
            serializable_target['java_file_path_abs'] = str(serializable_target['java_file_path_abs'])
        serializable_targets.append(serializable_target)

    # Save the discovered targets metadata to a JSON file
    try:
        with open(ANALYSIS_RESULTS_FILE, 'w', encoding='utf-8') as f:
            json.dump(serializable_targets, f, indent=4)
        print(f"[SUCCESS] Discovered Spring Boot targets saved to: {ANALYSIS_RESULTS_FILE}")
    except Exception as e:
        print(f"[ERROR] Could not save analysis results to '{ANALYSIS_RESULTS_FILE}': {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()


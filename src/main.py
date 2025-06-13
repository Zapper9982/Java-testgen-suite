import os
import sys
from pathlib import Path
import subprocess
import logging
import json # Added for loading analysis results

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration ---
# Add TESTGEN_AUTOMATION_ROOT to sys.path to allow importing sibling modules
TESTGEN_AUTOMATION_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(TESTGEN_AUTOMATION_ROOT)) # src.module can now be imported

# Import TestCaseGenerator and get_test_paths after sys.path modification
from src.llm.test_case_generator import TestCaseGenerator, get_test_paths
from src.test_runner.java_test_runner import JavaTestRunner # Added for coverage analysis


SPRING_BOOT_PROJECT_ROOT_STR = os.getenv("SPRING_BOOT_PROJECT_ROOT")
if not SPRING_BOOT_PROJECT_ROOT_STR:
    logging.error("Environment variable SPRING_BOOT_PROJECT_ROOT not set.")
    sys.exit(1)
SPRING_BOOT_PROJECT_ROOT = Path(SPRING_BOOT_PROJECT_ROOT_STR)

MAX_ITERATIONS = int(os.getenv("MAX_ITERATIONS", "5")) # Default to 5 iterations
TARGET_COVERAGE = float(os.getenv("TARGET_COVERAGE", "0.9")) # Default to 90%
methods_to_target = None # Initially, target all discovered classes/methods

# Define paths to scripts - adjust if these scripts are refactored into functions later
PRE_PROCESSING_SCRIPT = TESTGEN_AUTOMATION_ROOT / "pre-processing" / "processing.py"
CODE_ANALYZER_SCRIPT = TESTGEN_AUTOMATION_ROOT / "src" / "analyzer" / "code_analyzer.py"
CHUNKER_SCRIPT = TESTGEN_AUTOMATION_ROOT / "scripts" / "chunker.py"
EMBED_CHUNKS_SCRIPT = TESTGEN_AUTOMATION_ROOT / "scripts" / "embed_chunks.py"

# Output directory for analysis results (consistent with code_analyzer.py)
ANALYSIS_RESULTS_DIR = TESTGEN_AUTOMATION_ROOT / "analysis_results"
ANALYSIS_RESULTS_FILE = ANALYSIS_RESULTS_DIR / "spring_boot_targets.json"


def run_script(script_path: Path, description: str) -> bool:
    """Runs a python script and logs its execution status."""
    logging.info(f"Starting: {description}...")
    try:
        process = subprocess.run(['python3', str(script_path)], capture_output=True, text=True, check=True)
        logging.info(f"Output from {description}:\n{process.stdout}")
        if process.stderr:
            logging.warning(f"Stderr from {description}:\n{process.stderr}")
        logging.info(f"Successfully completed: {description}.")
        return True
    except subprocess.CalledProcessError as e:
        logging.error(f"Error during: {description}.")
        logging.error(f"Return code: {e.returncode}")
        logging.error(f"Stdout: {e.stdout}")
        logging.error(f"Stderr: {e.stderr}")
        return False
    except FileNotFoundError:
        logging.error(f"Error: Script not found at {script_path}")
        return False

def main_pipeline():
    logging.info("Starting main test generation pipeline...")
    logging.info(f"Spring Boot project root: {SPRING_BOOT_PROJECT_ROOT}")
    logging.info(f"Max iterations: {MAX_ITERATIONS}")
    logging.info(f"Target coverage: {TARGET_COVERAGE*100}%")

    # 1. Pre-processing
    if not run_script(PRE_PROCESSING_SCRIPT, "Pre-processing Java code"):
        logging.error("Pre-processing failed. Exiting pipeline.")
        sys.exit(1)

    # 2. Code Analysis
    # Ensure the analysis results directory exists (code_analyzer.py should also do this)
    ANALYSIS_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if not run_script(CODE_ANALYZER_SCRIPT, "Analyzing Spring Boot application"):
        logging.error("Code analysis failed. Exiting pipeline.")
        sys.exit(1)

    if not ANALYSIS_RESULTS_FILE.exists():
        logging.error(f"Code analysis did not produce the expected output file: {ANALYSIS_RESULTS_FILE}. Exiting.")
        sys.exit(1)
    logging.info(f"Code analysis results available at: {ANALYSIS_RESULTS_FILE}")

    # --- Main Loop (to be detailed in subsequent steps) ---
    current_coverage = 0.0
    for iteration in range(1, MAX_ITERATIONS + 1):
        logging.info(f"--- Starting Iteration {iteration}/{MAX_ITERATIONS} ---")

        # 3. Langchain/ChromaDB Setup (Chunking and Embedding)
        logging.info("Running chunker script...")
        if not run_script(CHUNKER_SCRIPT, "Chunking source files"):
            logging.warning("Chunking script failed. This might impact test generation quality.")
            # Potentially skip this iteration or exit if critical
            # continue

        logging.info("Running embed chunks script...")
        if not run_script(EMBED_CHUNKS_SCRIPT, "Embedding chunks into ChromaDB"):
            logging.warning("Embedding script failed. This might impact test generation quality.")
            # Potentially skip this iteration or exit if critical
            # continue

        # 4. Test Case Generation
        logging.info("Initializing Test Case Generator...")
        try:
            if not ANALYSIS_RESULTS_FILE.exists():
                logging.error(f"Analysis results file not found: {ANALYSIS_RESULTS_FILE}. Cannot proceed with test generation.")
                break # Break the loop for this iteration

            with open(ANALYSIS_RESULTS_FILE, 'r', encoding='utf-8') as f:
                discovered_targets_metadata = json.load(f)

            if not discovered_targets_metadata:
                logging.info("No targets found in analysis results. Skipping test generation for this iteration.")
                break # Break the loop

            # Determine build tool (e.g. from env var, default to maven)
            build_tool = os.getenv("BUILD_TOOL", "maven").lower()
            logging.info(f"Using build tool: {build_tool} for JavaTestRunner initialization.")

            test_generator = TestCaseGenerator(collection_name="code_chunks_collection", build_tool=build_tool)

            logging.info(f"Loaded {len(discovered_targets_metadata)} targets for test generation.")

            for target_info in discovered_targets_metadata:
                java_file_path_abs = Path(target_info['java_file_path_abs'])
                relative_processed_txt_path = Path(target_info['relative_processed_txt_path'])
                target_class_name = target_info['class_name']
                target_package_name = target_info['package_name']
                identified_dependencies_filenames = target_info['dependent_filenames']
                custom_imports_list = target_info['custom_imports']

                relevant_java_files_for_context = [java_file_path_abs.name] + identified_dependencies_filenames
                relevant_java_files_for_context = list(set(relevant_java_files_for_context))

                paths = get_test_paths(str(relative_processed_txt_path), SPRING_BOOT_PROJECT_ROOT)
                test_output_dir = paths["test_output_dir"]
                test_output_file_path = paths["test_output_file_path"]

                # Ensure test output directory exists
                test_output_dir.mkdir(parents=True, exist_ok=True)

                logging.info(f"Generating test for: {target_class_name} ({target_package_name})")

                generated_test_code = test_generator.generate_test_case(
                    target_class_name=target_class_name,
                    target_package_name=target_package_name,
                    custom_imports=custom_imports_list,
                    relevant_java_files_for_context=relevant_java_files_for_context,
                    test_output_file_path=test_output_file_path,
                    focused_methods=methods_to_target, # New parameter
                    additional_query_instructions=""
                )

                if generated_test_code and test_generator.last_test_run_results and test_generator.last_test_run_results.get("status") == "SUCCESS":
                    logging.info(f"Successfully generated and passed test for {target_class_name}.")
                else:
                    logging.warning(f"Failed to generate a passing test for {target_class_name} after retries. See logs from TestCaseGenerator.")

        except ImportError as e:
            logging.error(f"Failed to import necessary modules for test generation: {e}")
            logging.error("Ensure 'src' is in PYTHONPATH or sys.path is configured correctly.")
            break # Critical error, break the loop
        except Exception as e:
            logging.error(f"An error occurred during test case generation setup or execution: {e}", exc_info=True)
            break # Critical error, break the loop

        # 5. Build and JaCoCo Analysis
        logging.info("Initializing Java Test Runner for coverage analysis...")
        try:
            # BUILD_TOOL is already defined from TestCaseGenerator section, or define again if needed
            build_tool = os.getenv("BUILD_TOOL", "maven").lower()
            java_test_runner = JavaTestRunner(project_root=SPRING_BOOT_PROJECT_ROOT, build_tool=build_tool)

            logging.info("Attempting to run build and get JaCoCo coverage...")
            coverage_data = java_test_runner.get_coverage()

            if coverage_data:
                current_coverage = coverage_data.get("overall_line_coverage", 0.0)
                logging.info(f"Overall line coverage: {current_coverage * 100:.2f}%")

                methods_below_threshold_current_iteration = []
                if current_coverage < TARGET_COVERAGE:
                    logging.info(f"Coverage {current_coverage * 100:.2f}% is below target {TARGET_COVERAGE * 100}%.")
                    for method_info in coverage_data.get("methods", []):
                        if method_info.get("line_coverage", 0.0) < TARGET_COVERAGE:
                            methods_below_threshold_current_iteration.append(method_info)

                    if methods_below_threshold_current_iteration:
                        logging.info(f"{len(methods_below_threshold_current_iteration)} methods found below target coverage. These will be focused on in the next iteration.")
                        methods_to_target = methods_below_threshold_current_iteration
                    else:
                        logging.info("Overall coverage is low, but no specific methods identified as below threshold in this iteration. Will revert to general targeting if this isn't the last iteration.")
                        methods_to_target = None
                else:
                    logging.info(f"Target coverage of {TARGET_COVERAGE * 100}% reached!")
                    methods_to_target = None
            else:
                logging.error("Failed to get coverage data. Assuming 0% coverage for this iteration.")
                current_coverage = 0.0
                methods_to_target = None

        except ImportError as e:
            logging.error(f"Failed to import JavaTestRunner: {e}", exc_info=True)
            current_coverage = 0.0 # Assume failure
            methods_to_target = None
            # break # Critical error
        except Exception as e:
            logging.error(f"An error occurred during build and coverage analysis: {e}", exc_info=True)
            current_coverage = 0.0 # Assume failure
            methods_to_target = None
            # break # Critical error

        # --- Full Loop Logic ---
        if current_coverage >= TARGET_COVERAGE:
            logging.info(f"Target coverage of {TARGET_COVERAGE*100:.2f}% reached or exceeded. Stopping iterations.")
            break # Exit loop

        if iteration == MAX_ITERATIONS:
            logging.warning(f"Reached max iterations ({MAX_ITERATIONS}) without achieving target coverage.")
            break # Exit loop

        if current_coverage < TARGET_COVERAGE: # This condition is implicitly met if loop continues
            logging.info(f"Coverage is {current_coverage*100:.2f}%. Preparing for next iteration.")
            if methods_to_target:
                 logging.info(f"Next iteration will focus on {len(methods_to_target)} specific methods.")
            else:
                 logging.info("Next iteration will use general targeting due to no specific low-coverage methods identified or an issue in coverage data.")
            # TODO: Implement logic to use 'methods_to_target' to guide the next round of test generation.
            # This involves:
            # 1. Modifying the targets for TestCaseGenerator for the next iteration if methods_to_target is not None.
            #    If methods_to_target IS None but coverage is low, it means we couldn't pinpoint methods,
            #    so TestCaseGenerator should proceed as it did in the first iteration (general targeting).
            # 2. Adjusting prompts in TestCaseGenerator to focus on these specific methods.
            # 3. Potentially re-running chunking/embedding if the context needs to be refined for these methods.
            pass # Explicitly pass for now

    # --- End of Main Loop ---

    if current_coverage >= TARGET_COVERAGE:
        logging.info("Pipeline completed successfully. Target coverage achieved.")
    else:
        logging.warning(f"Pipeline completed. Target coverage of {TARGET_COVERAGE*100}% was NOT achieved after {MAX_ITERATIONS} iterations. Final coverage: {current_coverage * 100:.2f}%.")

    # Determine exit code based on coverage
    if current_coverage >= TARGET_COVERAGE:
        sys.exit(0) # Success
    else:
        sys.exit(1) # Failure (for GitHub Actions)


if __name__ == "__main__":
    # Ensure SPRING_BOOT_PROJECT_ROOT is set before running
    if not SPRING_BOOT_PROJECT_ROOT_STR:
        print("ERROR: The environment variable SPRING_BOOT_PROJECT_ROOT must be set.")
        print("Example: export SPRING_BOOT_PROJECT_ROOT=/path/to/your/spring-boot-project")
        sys.exit(1)

    main_pipeline()

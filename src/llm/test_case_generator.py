import sys
from pathlib import Path
import os
import json 
from typing import List, Dict, Any, Union

# Assuming the script is located within the project, e.g., 'your_project_root/src/your_module/script.py'
# TESTGEN_AUTOMATION_ROOT will be 'your_project_root/'
TESTGEN_AUTOMATION_ROOT = Path(__file__).parent.parent.parent

# --- Read Spring Boot project root from environment variable ---
SPRING_BOOT_PROJECT_ROOT_ENV = os.getenv("SPRING_BOOT_PROJECT_PATH")
if not SPRING_BOOT_PROJECT_ROOT_ENV:
    print("ERROR: SPRING_BOOT_PROJECT_PATH environment variable is not set.")
    print("Please set it in your run.sh script or before running test_case_generator.py.")
    sys.exit(1)
SPRING_BOOT_PROJECT_ROOT = Path(SPRING_BOOT_PROJECT_ROOT_ENV)
SPRING_BOOT_MAIN_JAVA_DIR = SPRING_BOOT_PROJECT_ROOT / "src" / "main" / "java" # Still define this if needed elsewhere

# The 'processed_output' directory is assumed to be directly under TESTGEN_AUTOMATION_ROOT
PROCESSED_OUTPUT_ROOT = TESTGEN_AUTOMATION_ROOT / "processed_output" 

# Add the 'src' directory of testgen-automation to sys.path
# This makes modules directly under 'src' (like analyzer/code_analysis_utils.py, test_runner/java_test_runner.py) discoverable
# by using absolute imports like 'from analyzer.code_analysis_utils import ...'
TESTGEN_AUTOMATION_SRC_DIR = TESTGEN_AUTOMATION_ROOT / "src"
if str(TESTGEN_AUTOMATION_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(TESTGEN_AUTOMATION_SRC_DIR))
    print(f"Added {TESTGEN_AUTOMATION_SRC_DIR} to sys.path for internal module imports.")

# Import necessary utilities and the new JavaTestRunner
from analyzer.code_analysis_utils import extract_custom_imports_from_chunk_file 
from test_runner.java_test_runner import JavaTestRunner 

from chroma_db.chroma_client import get_chroma_client, get_or_create_collection
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import torch 
from llm import test_prompt_templates


# --- Google API Configuration ---
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") 
if not GOOGLE_API_KEY:
    print("WARNING: GOOGLE_API_KEY environment variable not set. Please set it for Gemini API calls.")
    print("Example: export GOOGLE_API_KEY='your_google_api_key_here'")

# LLM model definition - Using Gemini 1.5 Flash for potentially better rate limits and speed
LLM_MODEL_NAME_GEMINI = "gemini-1.5-flash" 

EMBEDDING_MODEL_NAME_BGE = "BAAI/bge-small-en-v1.5"
# Use 'mps' for Apple Silicon (M1/M2/M3 chips) if available, otherwise 'cpu'
if torch.backends.mps.is_available():
    DEVICE_FOR_EMBEDDINGS = "mps"
    print("Detected Apple Silicon (M1/M2/M3). Using 'mps' device for embeddings.")
else:
    DEVICE_FOR_EMBEDDINGS = "cpu"
    print("Apple Silicon MPS not available or detected. Falling back to 'cpu' for embeddings.")

# Define the expected location of the pre-analyzed Spring Boot targets JSON file
ANALYSIS_RESULTS_DIR = TESTGEN_AUTOMATION_ROOT / "analysis_results"
ANALYSIS_RESULTS_FILE = ANALYSIS_RESULTS_DIR / "spring_boot_targets.json"

# Max retries for test generation + fixing
MAX_TEST_GENERATION_RETRIES = 3 

def get_test_paths(relative_filepath_from_processed_output: str, project_root: Path):
    """
    Generates the expected output paths for the test file based on the path
    relative to the 'processed_output' directory.

    Args:
        relative_filepath_from_processed_output: Path string like "com/package/file.txt"
                                                 which is relative to PROCESSED_OUTPUT_ROOT.
        project_root: The root directory of the Spring Boot project.

    Returns:
        A dictionary containing various path components.
    """
    # Create a Path object from the relative string
    relative_path_obj = Path(relative_filepath_from_processed_output)

    # 1. Get the package path (directory part)
    package_path = relative_path_obj.parent

    # 2. Get the original filename base (without .txt or .java)
    original_filename_base = relative_path_obj.stem 

    # 3. Construct the original .java filename (e.g., "MyService.java")
    original_java_filename = f"{original_filename_base}.java" 

    # 4. Construct the test class name (e.g., "MyServiceTest.java")
    test_class_name = f"{original_filename_base}Test.java"

    # 5. Construct the full output directory for the test file
    test_output_dir = project_root / "src" / "test" / "java" / package_path

    # 6. Construct the full output file path
    test_output_file_path = test_output_dir / test_class_name

    return {
        "original_java_filename": original_java_filename,
        "test_output_dir": test_output_dir,
        "test_output_file_path": test_output_file_path
    }


class TestCaseGenerator:
    def __init__(self, collection_name: str = "code_chunks_collection", build_tool: str = "maven"): 
        print("Initializing TestCaseGenerator with LangChain components (Google Gemini LLM)...")
        
        print(f"Loading embedding model: {EMBEDDING_MODEL_NAME_BGE} on {DEVICE_FOR_EMBEDDINGS}...")
        self.embeddings = HuggingFaceBgeEmbeddings(
            model_name=EMBEDDING_MODEL_NAME_BGE, 
            model_kwargs={'device': DEVICE_FOR_EMBEDDINGS},
            encode_kwargs={'normalize_embeddings': True}
        )
        print("Embedding model loaded.")
          
        print(f"Connecting to ChromaDB collection: {collection_name}...")
        self.chroma_client = get_chroma_client()
        self.vectorstore = Chroma(
            client=self.chroma_client,
            collection_name=collection_name,
            embedding_function=self.embeddings 
        )
        # Initialize retriever with a default k. It will be updated dynamically later.
        self.retriever = self.vectorstore.as_retriever(
            search_type="mmr", 
            search_kwargs={"k": 15},
        )
        print("ChromaDB retriever instantiated (default settings).")

        self.llm = self._instantiate_llm()
        print("Google Gemini LLM instantiated for LangChain.")

        # QA chain will be initialized/updated dynamically in generate_test_case
        self.qa_chain = None 

        project_root_from_env = os.getenv("SPRING_BOOT_PROJECT_PATH")
        if not project_root_from_env:
            raise ValueError("SPRING_BOOT_PROJECT_PATH environment variable is not set. Cannot initialize JavaTestRunner.")
        self.java_test_runner = JavaTestRunner(project_root=Path(project_root_from_env), build_tool=build_tool)
        self.last_test_run_results = None # Initialize to store feedback

    def _instantiate_llm(self) -> ChatGoogleGenerativeAI:
        if not GOOGLE_API_KEY:
            raise ValueError("GOOGLE_API_KEY environment variable is not set. Cannot initialize Gemini LLM.")
        
        print(f"Using Google Gemini LLM: {LLM_MODEL_NAME_GEMINI}...")
        return ChatGoogleGenerativeAI(model=LLM_MODEL_NAME_GEMINI, temperature=0.7)

    def _update_retriever_filter(self, filenames: Union[str, List[str]]):
        """
        Updates the retriever's filter to target a specific filename or a list of filenames.
        This allows the retriever to fetch chunks from the target file and its dependencies.
        """
        if isinstance(filenames, str):
            filter_filenames = [filenames]
        else:
            filter_filenames = filenames 

        # Using $in operator to filter by multiple filenames
        self.retriever = self.vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": 15, # Still a TODO for dynamic adjustment, consider increasing if context is too small
                "filter": {"filename": {"$in": filter_filenames}}
            },
        )
        # Update the QA chain with the new retriever if it's already initialized.
        if self.qa_chain:
            self.qa_chain.retriever = self.retriever
        print(f"Retriever filter updated to target filenames: '{filter_filenames}'")

    def _detect_test_type(self, target_info: Dict[str, Any]) -> str:
        ttype = target_info.get('type', '').lower()
        if 'controller' in ttype:
            return 'controller'
        elif 'service' in ttype:
            return 'service'
        elif 'repository' in ttype:
            return 'repository'
        return 'service'  # Default fallback

    def _get_prompt_template(self, test_type: str, target_class_name: str, target_package_name: str, custom_imports: list, additional_query_instructions: str, dependency_signatures: dict = None) -> str:
        if test_type == 'controller':
            return test_prompt_templates.get_controller_test_prompt_template(
                target_class_name, target_package_name, custom_imports, additional_query_instructions, dependency_signatures
            )
        elif test_type == 'repository':
            return test_prompt_templates.get_repository_test_prompt_template(
                target_class_name, target_package_name, custom_imports, additional_query_instructions, dependency_signatures
            )
        else:
            return test_prompt_templates.get_service_test_prompt_template(
                target_class_name, target_package_name, custom_imports, additional_query_instructions, dependency_signatures
            )

    def generate_test_case(self, 
                           target_class_name: str, 
                           target_package_name: str, 
                           custom_imports: List[str],
                           relevant_java_files_for_context: List[str],
                           test_output_file_path: Path, 
                           additional_query_instructions: str,
                           requires_db_test: bool,
                           dependency_signatures: Dict[str, str] = None,
                           target_info: Dict[str, Any] = None) -> str:
        self._update_retriever_filter(relevant_java_files_for_context)
        test_type = self._detect_test_type(target_info or {})
        print(f"Generating {test_type.upper()} test for {target_class_name}...")
        base_template = self._get_prompt_template(
            test_type, target_class_name, target_package_name, custom_imports,
            additional_query_instructions, dependency_signatures
        )

        generated_code = ""
        for retry_attempt in range(MAX_TEST_GENERATION_RETRIES):
            print(f"\nAttempt {retry_attempt + 1}/{MAX_TEST_GENERATION_RETRIES} for test generation for {target_class_name}...")
            
            # Construct the prompt for the current attempt
            current_prompt_query = f"Generate tests for {target_class_name}."
            if retry_attempt > 0 and self.last_test_run_results:
                # Safely get stdout/stderr and escape any curly braces
                # Fix: Escape curly braces in stdout/stderr to prevent f-string formatting errors
                stdout_content = self.last_test_run_results.get('stdout', '').replace('{', '{{').replace('}', '}}')
                stderr_content = self.last_test_run_results.get('stderr', '').replace('{', '{{').replace('}', '}}')

                # Prepare detailed error feedback
                detailed_feedback = ""
                if self.last_test_run_results.get('detailed_errors'):
                    comp_errors = self.last_test_run_results['detailed_errors'].get('compilation_errors', [])
                    test_failures = self.last_test_run_results['detailed_errors'].get('test_failures', [])
                    general_msgs = self.last_test_run_results['detailed_errors'].get('general_messages', [])

                    if comp_errors:
                        detailed_feedback += "\n--- COMPILATION ERRORS ---\n"
                        for err in comp_errors:
                            detailed_feedback += f"File: {err.get('file', 'N/A')}, Line: {err.get('line', 'N/A')}, Col: {err.get('column', 'N/A')}, Message: {err.get('message', 'N/A')}\n"
                    if test_failures:
                        detailed_feedback += "\n--- TEST FAILURES ---\n"
                        for failure in test_failures:
                            detailed_feedback += f"Summary: {failure.get('summary', 'N/A')}\n"
                            for detail in failure.get('details', []):
                                detailed_feedback += f"  - {detail}\n"
                    if general_msgs:
                        detailed_feedback += "\n--- GENERAL BUILD MESSAGES ---\n"
                        for msg in general_msgs:
                            detailed_feedback += f"  - {msg}\n"
                
                # --- NEW DEBUGGING PRINTS ---
                print("\n--- DETAILED ERROR FEEDBACK SENT TO LLM (for debugging) ---")
                print(detailed_feedback)
                print("\n--- COMPLETE ERROR FEEDBACK MESSAGE SENT TO LLM (for debugging) ---")
                # Also print the full message for complete context
                error_feedback_message = (
                    "\n\n--- PREVIOUS ATTEMPT FEEDBACK ---\n"
                    f"The previously generated test for `{target_class_name}` encountered the following issue:\n"
                    f"Status: {self.last_test_run_results.get('status', 'N/A')}\n"
                    f"Message: {self.last_test_run_results.get('message', 'N/A')}\n"
                    f"{detailed_feedback}" # Insert detailed feedback here
                    f"Full STDOUT (truncated):\n{stdout_content[:2000]}...\n" # Truncate for prompt length
                    f"Full STDERR (truncated):\n{stderr_content[:2000]}...\n" # Truncate for prompt length
                    f"Please analyze the errors/failures and revise the test code to fix them."
                    f"Ensure you still adhere to all original instructions (Mockito usage, coverage, imports, etc.)."
                    f"\n--- END PREVIOUS ATTEMPT FEEDBACK ---\n"
                )
                print(error_feedback_message)
                print("\n--------------------------------------------------------------")
                # --- END NEW DEBUGGING PRINTS ---

                # Modify the template to include feedback
                template_with_feedback = base_template + error_feedback_message 
                QA_CHAIN_PROMPT = PromptTemplate.from_template(template_with_feedback)
            else:
                QA_CHAIN_PROMPT = PromptTemplate.from_template(base_template)

            # Initialize or update the QA chain with the dynamic prompt
            # This handles cases where QA_CHAIN_PROMPT might be re-assigned in the loop
            if self.qa_chain: # Update existing chain if it exists
                self.qa_chain.combine_documents_chain.llm_chain.prompt = QA_CHAIN_PROMPT
            else: # Initialize new chain if it doesn't exist
                self.qa_chain = RetrievalQA.from_chain_type(
                    llm=self.llm,
                    chain_type="stuff",
                    retriever=self.retriever,
                    return_source_documents=True, 
                    chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
                )


            try:
                result = self.qa_chain({"query": current_prompt_query})
                response_text = result["result"]

                # Extract code block
                temp_generated_code = response_text.strip()
                if "```java" in temp_generated_code:
                    start_marker = "```java"
                    end_marker = "```"
                    start_index = temp_generated_code.find(start_marker)
                    if start_index != -1:
                        code_start = start_index + len(start_marker)
                        code_end = temp_generated_code.find(end_marker, code_start)
                        if code_end != -1:
                            temp_generated_code = temp_generated_code[code_start:code_end].strip()
                            if temp_generated_code.startswith("// Begin generated test code"):
                                 temp_generated_code = temp_generated_code.replace("// Begin generated test code", "", 1).strip()
                
                if not temp_generated_code or temp_generated_code.lower().startswith("here is the"):
                    print(f"LLM generated an incomplete or conversational response. Retrying...")
                    self.last_test_run_results = {"status": "ERROR", "message": "LLM generated incomplete or conversational code.", "stdout": response_text, "stderr": "No executable code block found."}
                    continue # Retry if LLM output is not a proper code block

                generated_code = temp_generated_code # Store the extracted code

                # Write the generated test case to a temporary file for execution
                os.makedirs(test_output_file_path.parent, exist_ok=True)
                with open(test_output_file_path, 'w', encoding='utf-8') as f:
                    f.write(generated_code)
                print(f"Temporarily saved test file for execution: {test_output_file_path}")

                # Execute the generated test
                test_run_results = self.java_test_runner.run_test(test_output_file_path)
                self.last_test_run_results = test_run_results # Store for potential next iteration

                if test_run_results["status"] == "SUCCESS":
                    print(f"Test for {target_class_name} PASSED on attempt {retry_attempt + 1}.")
                    return generated_code # Test passed, return the code
                elif test_run_results["status"] == "FAILED" or test_run_results["status"] == "ERROR":
                    print(f"Test for {target_class_name} FAILED/ERRORED on attempt {retry_attempt + 1}. Message: {test_run_results['message']}")
                    # --- START MODIFICATION FOR DEBUGGING ---
                    print("\n--- FULL STDOUT from Maven execution (for debugging) ---")
                    print(test_run_results.get('stdout', '[No stdout available]'))
                    print("\n--- FULL STDERR from Maven execution (for debugging) ---")
                    print(test_run_results.get('stderr', '[No stderr available]'))
                    # --- END MODIFICATION FOR DEBUGGING ---
                    print("Feeding error feedback to LLM for next attempt...")
                    # Loop continues for next retry
                else: # UNKNOWN status
                    print(f"Test for {target_class_name} returned UNKNOWN status on attempt {retry_attempt + 1}. Message: {test_run_results['message']}")
                    # --- START MODIFICATION FOR DEBUGGING ---
                    print("\n--- FULL STDOUT from Maven execution (for debugging) ---")
                    print(test_run_results.get('stdout', '[No stdout available]'))
                    print("\n--- FULL STDERR from Maven execution (for debugging) ---")
                    print(test_run_results.get('stderr', '[No stderr available]'))
                    # --- END MODIFICATION FOR DEBUGGING ---
                    print("Feeding unknown status feedback to LLM for next RegEx...")
                    # Loop continues for next retry

            except Exception as e:
                print(f"An error occurred during LLM call or test execution setup: {e}. Retrying...")
                self.last_test_run_results = {"status": "ERROR", "message": f"Internal generation/execution error: {e}", "stdout": "", "stderr": str(e)}
                # The loop will continue for the next retry attempt

        print(f"Failed to generate a passing test for {target_class_name} after {MAX_TEST_GENERATION_RETRIES} attempts.")
        return generated_code # Return the last generated code, even if it failed

if __name__ == "__main__":
    try:
        # User can specify build tool via env var or hardcode it here
        BUILD_TOOL = os.getenv("BUILD_TOOL", "maven").lower() 
        test_generator = TestCaseGenerator(collection_name="code_chunks_collection", build_tool=BUILD_TOOL) 
        
        # --- Preprocessing/Discovery Phase ---
        # The data is loaded from the JSON file generated by src/analyzer/code_analyzer.py
        if not ANALYSIS_RESULTS_FILE.exists():
            print(f"ERROR: Analysis results file not found: {ANALYSIS_RESULTS_FILE}")
            print("Please run 'python3 src/analyzer/code_analyzer.py' first to generate analysis data.")
            sys.exit(1)

        print(f"\nLoading discovered targets from: {ANALYSIS_RESULTS_FILE}")
        with open(ANALYSIS_RESULTS_FILE, 'r', encoding='utf-8') as f:
            discovered_targets_metadata = json.load(f)
        
        print(f"Loaded {len(discovered_targets_metadata)} Spring Boot Service/Controller targets for test generation.")

        if not discovered_targets_metadata:
            print("No Spring Boot Service or Controller targets found. Exiting test generation.")
            sys.exit(0) 

        # --- Test Case Generation Phase: Iterate through each discovered target ---
        for target_info in discovered_targets_metadata:
            # Convert paths back to Path objects as they were stored as strings in JSON
            java_file_path_abs = Path(target_info['java_file_path_abs'])
            relative_processed_txt_path = Path(target_info['relative_processed_txt_path']) 
            target_class_name = target_info['class_name']
            target_package_name = target_info['package_name']
            identified_dependencies_filenames = target_info['dependent_filenames']
            custom_imports_list = target_info['custom_imports'] 
            # NEW: Read the requires_db_test flag from analyzer output
            requires_db_test = target_info.get('requires_db_test', False) 
            target_class_type = target_info.get('type', 'Unknown') # Get class type (Controller/Service/Repository)


            # Combine the target file's own filename with its identified dependencies for retrieval
            relevant_java_files_for_context = [java_file_path_abs.name] + identified_dependencies_filenames
            relevant_java_files_for_context = list(set(relevant_java_files_for_context)) # Ensure uniqueness
            
            # --- Prepare output paths ---
            paths = get_test_paths(str(relative_processed_txt_path), SPRING_BOOT_PROJECT_ROOT)
            test_output_dir = paths["test_output_dir"]
            test_output_file_path = paths["test_output_file_path"]

            print("\n" + "="*80)
            print(f"GENERATING UNIT TEST FOR: {target_class_name} ({target_package_name})") # Updated message
            print(f"SOURCE FILE: {java_file_path_abs}")
            print(f"FILES TO RETRIEVE CHUNKS FROM (for context): {relevant_java_files_for_context}")
            print(f"EXTRACTED CUSTOM IMPORTS (for prompt): {custom_imports_list}")
            print(f"DATABASE MOCKING REQUIRED: {requires_db_test}") # New print statement
            print(f"EXPECTED TEST OUTPUT PATH: '{test_output_file_path}'")
            print("="*80)

            # --- Generate the test case with feedback loop ---
            generated_test_code = test_generator.generate_test_case(
                target_class_name=target_class_name,
                target_package_name=target_package_name,
                custom_imports=custom_imports_list,
                relevant_java_files_for_context=relevant_java_files_for_context,
                test_output_file_path=test_output_file_path, # Pass the output path
                additional_query_instructions="and make sure there are no errors, and you don't cause mismatch in return types and stuff.",
                requires_db_test=requires_db_test, # Pass the flag to the prompt template
                dependency_signatures=None, # Pass None for now, as dependency_signatures are not provided in the input
                target_info=target_info # Pass the target_info for test type detection
            )
            
            os.makedirs(test_output_dir, exist_ok=True)

            # Write the final generated test case to the file (could be the corrected one)
            try:
                with open(test_output_file_path, 'w', encoding='utf-8') as f:
                    f.write(generated_test_code)
                print(f"\n[FINAL SUCCESS] Generated test case saved to: '{test_output_file_path}'")
            except Exception as e:
                print(f"\n[FINAL ERROR] Could not save test case to '{test_output_file_path}': {e}")

            print("\n--- FINAL GENERATED TEST CASE (Printed to Console for review) ---")
            print(generated_test_code) # Corrected variable name
            print("\n" + "="*80 + "\n")
            
        # --- NEW: Run full project verification after all tests are generated ---
        print("\nInitiating full project test verification (mvn clean verify / gradle clean test)...")
        full_project_test_results = test_generator.java_test_runner.run_project_tests(is_full_verify=True)
        
        print("\n--- Full Project Test Verification Results ---")
        print(f"Status: {full_project_test_results['status']}")
        print(f"Message: {full_project_test_results['message']}")
        if full_project_test_results.get('summary'): # Use .get() for safety
            print(f"Summary: {full_project_test_results['summary']}")
        else:
            print("No detailed test summary available from build tool output.")
        
        if full_project_test_results['status'] != "SUCCESS":
            print("\nWARNING: Full project verification FAILED or had ERRORS. Check logs above.")
            # You might want to add more sophisticated error handling or logging here.
        print("\n--- Full Project Test Verification Completed ---")


    except ValueError as ve:
        print(f"Configuration Error: {ve}")
        print("Please ensure GOOGLE_API_KEY is set and other configurations are correct, especially file paths.")
    except Exception as e:
        print(f"An unexpected error occurred during main execution: {e}")
        print("Verify your ChromaDB setup and network connection for Google Generative AI API, and that file paths are correct.")


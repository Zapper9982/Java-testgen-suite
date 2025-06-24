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
from analyzer.code_analysis_utils import extract_custom_imports_from_chunk_file, resolve_transitive_dependencies
from test_runner.java_test_runner import JavaTestRunner 

from chroma_db.chroma_client import get_chroma_client, get_or_create_collection
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
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
MAX_TEST_GENERATION_RETRIES = 15

# --- Token Estimation Utility ---
def estimate_token_count(text: str) -> int:
    """
    Roughly estimate the number of tokens for Gemini (1 token ≈ 4 characters).
    """
    return len(text) // 4

MAX_TOTAL_TOKENS = 45000  # Stay below 50K for safety
MIN_K = 3  # Minimum number of context chunks

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
        self.embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME_BGE,
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

    def _update_retriever_filter(self, filenames: Union[str, List[str]], k_override: int = None):
        """
        Updates the retriever's filter to target a specific filename or a list of filenames.
        This allows the retriever to fetch chunks from the target file and its dependencies.
        Optionally, k_override can be used to force a lower k if token budget is exceeded.
        """
        if isinstance(filenames, str):
            filter_filenames = [filenames]
        else:
            filter_filenames = filenames 

        # Always include test utility/config files
        always_include = [
            "BaseTest.java", "TestUtils.java", "application-test.yml", "application-test.properties"
        ]
        for util_file in always_include:
            if util_file not in filter_filenames:
                filter_filenames.append(util_file)

        # Dynamically set k based on number of files, but allow override
        k = min(30, 5 + 2 * len(filter_filenames))
        if k_override is not None:
            k = max(MIN_K, k_override)

        self.retriever = self.vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": k,
                "filter": {"filename": {"$in": filter_filenames}}
            },
        )
        # Update the QA chain with the new retriever if it's already initialized.
        if self.qa_chain:
            self.qa_chain.retriever = self.retriever
        print(f"Retriever filter updated to target filenames: '{filter_filenames}' (k={k})")

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

    def generate_test_plan(self, target_class_name: str, class_code: str) -> List[Dict[str, str]]:
        """
        Step 1: Ask the LLM to generate a test plan (method names + descriptions) for the class.
        Returns a list of dicts: [{"method_name": ..., "description": ...}, ...]
        """
        plan_prompt = f"""
You are an expert Java test designer. Given the following class, list all public methods and, for each, describe the test cases you would write. For each test case, provide:
- The test method name (e.g., shouldReturnXWhenY)
- A one-sentence description of what it tests (including edge cases, exceptions, etc.)

Do NOT output any code, just a structured test plan in this format:
Test method: <methodName>
Description: <description>

--- BEGIN CLASS UNDER TEST ---
{class_code}
--- END CLASS UNDER TEST ---
"""
        print("\n--- DEBUG: TEST PLAN PROMPT ---\n" + plan_prompt + "\n--- END TEST PLAN PROMPT ---\n")
        # Use LLM directly (not RetrievalQA)
        result = self.llm.invoke(plan_prompt)
        if hasattr(result, "content"):
            result = result.content
        print("\n--- DEBUG: TEST PLAN OUTPUT ---\n" + result + "\n--- END TEST PLAN OUTPUT ---\n")
        # Parse the output into a list of test cases
        test_cases = []
        for block in result.split("Test method:"):
            lines = block.strip().splitlines()
            if not lines or not lines[0].strip():
                continue
            method_name = lines[0].strip()
            description = ""
            for l in lines[1:]:
                if l.lower().startswith("description:"):
                    description = l[len("description:"):].strip()
            if method_name:
                test_cases.append({"method_name": method_name, "description": description})
        return test_cases

    def generate_test_method(self, target_class_name: str, class_code: str, test_case: Dict[str, str], custom_imports: List[str], test_type: str, error_feedback: str = None) -> str:
        """
        Step 2: For each test case, generate the test method code using a focused prompt.
        Optionally, provide error feedback for retry/fix attempts.
        """
        imports_section = '\n'.join(custom_imports) if custom_imports else ''
        method_prompt = f"""
You are an expert Java test writer. Write a JUnit5+Mockito test method for the following scenario.

Class under test:
{class_code}

Test method name: {test_case['method_name']}
Description: {test_case['description']}

STRICT REQUIREMENTS:
- Output ONLY the Java test method code, no explanations or markdown.
- Use Mockito for mocking dependencies.
- Use JUnit5 assertions.
- Do NOT repeat the class under test.
- Use the following imports if needed:
{imports_section}
"""
        if error_feedback:
            method_prompt += f"\n--- ERROR FEEDBACK FROM PREVIOUS ATTEMPT ---\n{error_feedback}\n--- END ERROR FEEDBACK ---\n"
        print(f"\n--- DEBUG: TEST METHOD PROMPT for {test_case['method_name']} ---\n" + method_prompt + "\n--- END TEST METHOD PROMPT ---\n")
        result = self.llm.invoke(method_prompt)
        if hasattr(result, "content"):
            result = result.content
        print(f"\n--- DEBUG: TEST METHOD OUTPUT for {test_case['method_name']} ---\n" + result + "\n--- END TEST METHOD OUTPUT ---\n")
        # Extract code block if present
        code = result.strip()
        if "```" in code:
            code = code.split("```", 2)[1]
            if code.startswith("java"): code = code[len("java"):].strip()
        return code

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
        # --- Two-step approach for service/controller/repository ---
        test_type = self._detect_test_type(target_info or {})
        if test_type in ("service", "controller", "repository"):
            self._update_retriever_filter(relevant_java_files_for_context, k_override=10)
            retrieved_docs = self.retriever.get_relevant_documents(f"Full code for {target_class_name}")
            class_code = "\n\n".join([doc.page_content for doc in retrieved_docs if target_class_name in doc.page_content])
            if not class_code:
                class_code = "\n\n".join([doc.page_content for doc in retrieved_docs])
            test_plan = self.generate_test_plan(target_class_name, class_code)
            if not test_plan:
                print("[ERROR] LLM did not return a valid test plan. Aborting test generation.")
                return ""
            test_methods = []
            method_errors = {}
            for test_case in test_plan:
                error_feedback = None
                for attempt in range(MAX_TEST_GENERATION_RETRIES):
                    method_code = self.generate_test_method(target_class_name, class_code, test_case, custom_imports, test_type, error_feedback=error_feedback)
                    # Assemble a temporary test class with all previous methods + this one
                    imports_section = '\n'.join(custom_imports) if custom_imports else ''
                    temp_test_class_code = f"""
package {target_package_name};

{imports_section}
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.InjectMocks;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;

@ExtendWith(MockitoExtension.class)
class {target_class_name}Test {{
{chr(10).join(test_methods + [method_code])}
}}
"""
                    # Write to file and try to compile/run
                    os.makedirs(test_output_file_path.parent, exist_ok=True)
                    with open(test_output_file_path, 'w', encoding='utf-8') as f:
                        f.write(temp_test_class_code)
                    test_run_results = self.java_test_runner.run_test(test_output_file_path)
                    if test_run_results["status"] == "SUCCESS":
                        test_methods.append(method_code)
                        break
                    else:
                        # Prepare error feedback for next attempt
                        detailed_feedback = ""
                        if test_run_results.get('detailed_errors'):
                            comp_errors = test_run_results['detailed_errors'].get('compilation_errors', [])
                            test_failures = test_run_results['detailed_errors'].get('test_failures', [])
                            general_msgs = test_run_results['detailed_errors'].get('general_messages', [])
                            if comp_errors:
                                detailed_feedback += "\n--- COMPILATION ERRORS ---\n"
                                for err in comp_errors:
                                    detailed_feedback += f"File: {str(err.get('file', 'N/A'))}, Line: {str(err.get('line', 'N/A'))}, Col: {str(err.get('column', 'N/A'))}, Message: {str(err.get('message', 'N/A'))}\n"
                            if test_failures:
                                detailed_feedback += "\n--- TEST FAILURES ---\n"
                                for failure in test_failures:
                                    detailed_feedback += f"Summary: {str(failure.get('summary', 'N/A'))}\n"
                                    for detail in failure.get('details', []):
                                        detailed_feedback += f"  - {str(detail)}\n"
                            if general_msgs:
                                detailed_feedback += "\n--- GENERAL BUILD MESSAGES ---\n"
                                for msg in general_msgs:
                                    detailed_feedback += f"  - {str(msg)}\n"
                        stdout_content = test_run_results.get('stdout', '')
                        stderr_content = test_run_results.get('stderr', '')
                        error_feedback = (
                            f"Status: {test_run_results.get('status', 'N/A')}\n"
                            f"Message: {test_run_results.get('message', 'N/A')}\n"
                            f"{detailed_feedback}"
                            f"Full STDOUT (truncated):\n{stdout_content[:2000]}...\n"
                            f"Full STDERR (truncated):\n{stderr_content[:2000]}...\n"
                            "Please revise the test method to fix these errors."
                        )
                        print(f"\n--- DEBUG: ERROR FEEDBACK FOR {test_case['method_name']} (Attempt {attempt+1}) ---\n{error_feedback}\n--- END ERROR FEEDBACK ---\n")
                        if attempt == MAX_TEST_GENERATION_RETRIES - 1:
                            print(f"[ERROR] Failed to generate a passing test method for {test_case['method_name']} after {MAX_TEST_GENERATION_RETRIES} attempts.")
                            test_methods.append(method_code)  # Save the last attempt anyway
            # Step 3: Assemble the final test class
            imports_section = '\n'.join(custom_imports) if custom_imports else ''
            test_class_code = f"""
package {target_package_name};

{imports_section}
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.InjectMocks;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;

@ExtendWith(MockitoExtension.class)
class {target_class_name}Test {{
{chr(10).join(test_methods)}
}}
"""
            # Write the generated test class to file (overwrite, not append)
            with open(test_output_file_path, 'w', encoding='utf-8') as f:
                f.write(test_class_code)
            print(f"[TWO-STEP] Generated test class saved to: '{test_output_file_path}'")
            print("\n--- FINAL GENERATED TEST CLASS (Two-Step) ---\n" + test_class_code + "\n" + "="*80 + "\n")
            # Run the generated test as before
            test_run_results = self.java_test_runner.run_test(test_output_file_path)
            self.last_test_run_results = test_run_results
            if test_run_results["status"] == "SUCCESS":
                print(f"Test for {target_class_name} PASSED (two-step approach).")
                return test_class_code
            else:
                print(f"Test for {target_class_name} FAILED/ERRORED (two-step approach). Message: {test_run_results['message']}")
                print("\n--- FULL STDOUT from Maven execution (for debugging) ---")
                print(test_run_results.get('stdout', '[No stdout available]'))
                print("\n--- FULL STDERR from Maven execution (for debugging) ---")
                print(test_run_results.get('stderr', '[No stderr available]'))
                print("[TWO-STEP] Final test class still has errors after all retries.")
                return test_class_code
        # --- fallback: original approach for other types ---
        return super().generate_test_case(
            target_class_name=target_class_name,
            target_package_name=target_package_name,
            custom_imports=custom_imports,
            relevant_java_files_for_context=relevant_java_files_for_context,
            test_output_file_path=test_output_file_path,
            additional_query_instructions=additional_query_instructions,
            requires_db_test=requires_db_test,
            dependency_signatures=dependency_signatures,
            target_info=target_info
        )

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

            # --- NEW: Use transitive dependency resolution ---
            relevant_java_files_for_context = resolve_transitive_dependencies(
                java_file_path_abs, SPRING_BOOT_MAIN_JAVA_DIR
            )
            # Always include test utility/config files (handled in _update_retriever_filter)
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


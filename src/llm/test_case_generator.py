import sys
from pathlib import Path
import os
import json 
from typing import List, Dict, Any, Union

TESTGEN_AUTOMATION_ROOT = Path(__file__).parent.parent.parent
import os
SPRING_BOOT_PROJECT_ROOT_STR = os.getenv("SPRING_BOOT_PROJECT_ROOT")
if not SPRING_BOOT_PROJECT_ROOT_STR:
    raise ValueError("Environment variable SPRING_BOOT_PROJECT_ROOT not set.")
SPRING_BOOT_PROJECT_ROOT = Path(SPRING_BOOT_PROJECT_ROOT_STR)
SPRING_BOOT_MAIN_JAVA_DIR = SPRING_BOOT_PROJECT_ROOT / "src" / "main" / "java"
PROCESSED_OUTPUT_ROOT = TESTGEN_AUTOMATION_ROOT / "processed_output" 
TESTGEN_AUTOMATION_SRC_DIR = TESTGEN_AUTOMATION_ROOT / "src"
if str(TESTGEN_AUTOMATION_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(TESTGEN_AUTOMATION_SRC_DIR))
    print(f"Added {TESTGEN_AUTOMATION_SRC_DIR} to sys.path for internal module imports.")

from analyzer.code_analysis_utils import extract_custom_imports_from_chunk_file 
from test_runner.java_test_runner import JavaTestRunner 

from chroma_db.chroma_client import get_chroma_client, get_or_create_collection
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import torch
import logging # Added for logger


# --- Google API Configuration ---
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") 
if not GOOGLE_API_KEY:
    print("WARNING: GOOGLE_API_KEY environment variable not set. Please set it for Gemini API calls.")
    print("Example: export GOOGLE_API_KEY='your_google_api_key_here'")

# LLM model definition - Using Gemini 1.5 Flash for potentially better rate limits and speed
LLM_MODEL_NAME_GEMINI = "gemini-1.5-flash" 

EMBEDDING_MODEL_NAME_BGE = "BAAI/bge-small-en-v1.5"
DEVICE_FOR_EMBEDDINGS = "cuda" if torch.cuda.is_available() else "cpu" 

# Define the expected location of the pre-analyzed Spring Boot targets JSON file
ANALYSIS_RESULTS_DIR = TESTGEN_AUTOMATION_ROOT / "analysis_results"
ANALYSIS_RESULTS_FILE = ANALYSIS_RESULTS_DIR / "spring_boot_targets.json"

# Max retries for test generation + fixing
MAX_TEST_GENERATION_RETRIES = 5 

def get_test_paths(relative_filepath_from_processed_output: str, project_root: Path):
 
   
    relative_path_obj = Path(relative_filepath_from_processed_output)
    package_path = relative_path_obj.parent
    original_filename_base = relative_path_obj.stem 
    original_java_filename = f"{original_filename_base}.java" 
    test_class_name = f"{original_filename_base}Test.java"
    test_output_dir = project_root / "src" / "test" / "java" / package_path
    test_output_file_path = test_output_dir / test_class_name

    return {
        "original_java_filename": original_java_filename,
        "test_output_dir": test_output_dir,
        "test_output_file_path": test_output_file_path
    }


class TestCaseGenerator:
    def __init__(self, collection_name: str = "code_chunks_collection", build_tool: str = "maven"): # Added build_tool
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

        self.java_test_runner = JavaTestRunner(project_root=SPRING_BOOT_PROJECT_ROOT, build_tool=build_tool) # Initialize the test runner
        self.last_test_run_results = None # Initialize to store feedback
        self.logger = logging.getLogger(__name__) # Added logger instance

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

    def _get_base_prompt_template(self, target_class_name: str, target_package_name: str, custom_imports: List[str], additional_query_instructions: str) -> str:
        """Constructs the base prompt template for test generation."""
        formatted_custom_imports = "\n".join(custom_imports)
        
        return f"""
As an expert Java developer and Spring Boot testing specialist, your task is to generate a comprehensive JUnit 5 test class for the `{target_class_name}` class.
**Crucially, follow these rules for the test class structure and Mockito setup:**
1.  Use `@ExtendWith(MockitoExtension.class)` for JUnit 5.
2.  Declare dependencies that need to be mocked with `@Mock`.
3.  Declare the class under test with `@InjectMocks`.
4.  **VERY IMPORTANT for internal method stubbing on the class under test:** If `{target_class_name}` has methods that call other methods *within itself* that need to be stubbed for testing (e.g., to prevent real side effects or complex logic during a test of an outer method), then also annotate the `@InjectMocks` field with `@Spy`.
5.  When stubbing methods on a `@Spy` object (the `@InjectMocks` instance), **ALWAYS use `doReturn(value).when(spyObject).method()` or `doNothing().when(spyObject).voidMethod()` syntax.**
    * **DO NOT use `when(spyObject.method()).thenReturn(value)` for `@Spy` objects. This is the common cause of `MissingMethodInvocation` errors.**
6.  **Mockito Limitations - CRITICAL:** Be aware that Mockito cannot directly stub `private`, `static`, or `final` methods on a `@Spy` object without advanced (and often discouraged) configurations like PowerMock.
    * If a method is `private`, `static`, or `final`, **DO NOT attempt to stub it directly using Mockito.**
    * Instead, focus on testing the *public* methods that call these unmockable internal methods. Verify the *overall behavior* or *side effects* (e.g., interactions with other mocks, return values from the public method) rather than trying to mock the internal call itself.
    * **Any attempt to stub a private, static, or final method will result in a `MissingMethodInvocation` error.**
7.  Ensure tests are **deterministic**, cover all public methods, and aim for **100% JaCoCo coverage**.
8.  Use **Assertions** to validate outcomes, including for void functions (e.g., `verify` interactions).
9.  Exclude `DataTypeConverters`.
10. Ensure all necessary imports are included, especially from the target class's package and the `com.iemr` internal dependencies.

Here are the relevant imports from the original source file that you may need to consider:
{formatted_custom_imports}

Please import the target class using: `import {target_package_name}.{target_class_name};`
{additional_query_instructions}

Provide ONLY the complete Java code block for the test class. Do NOT include any conversational text, explanations, or extraneous characters outside the code block.

Here is the relevant code context from the project, retrieved from the vector database:

```java
{{context}}
// Begin generated test code
"""
    
    def generate_test_case(self, 
                           target_class_name: str, 
                           target_package_name: str, 
                           custom_imports: List[str],
                           relevant_java_files_for_context: List[str],
                           test_output_file_path: Path, # Added to save generated code
                           focused_methods: List[Dict[str, Any]] | None = None, # New parameter
                           additional_query_instructions: str = "") -> str:
        """
        Generates a JUnit 5 test case by querying the RetrievalQA chain,
        with a dynamically constructed prompt, and includes a feedback loop for corrections.
        """
        self._update_retriever_filter(relevant_java_files_for_context)

        current_additional_instructions = additional_query_instructions

        if focused_methods:
            # Assuming class_name in focused_methods might be fully qualified or simple.
            # target_class_name is simple.
            # A more robust check would be to compare against fqcn if available in focused_methods.
            relevant_focused_methods_for_this_class = [
                fm for fm in focused_methods
                if fm.get("class_name","").endswith(target_class_name)
            ]
            if relevant_focused_methods_for_this_class:
                instruction_intro = (
                    "\n\n--- FOCUSED COVERAGE IMPROVEMENT ---\n"
                    "The following methods in this class have been identified as needing better test coverage. "
                    "Please prioritize generating tests that specifically exercise these methods and improve their line coverage. "
                    "Pay close attention to their logic, branches, and edge cases:\n"
                )
                methods_str_parts = []
                # Ensure TARGET_COVERAGE is accessible, e.g. via os.getenv or passed in
                # For simplicity here, directly using a default or assuming it's part of the broader context.
                # If TestCaseGenerator is meant to be more isolated, this might need to be passed.
                target_coverage_str = os.getenv('TARGET_COVERAGE', '0.9')
                try:
                    target_coverage_val = float(target_coverage_str)
                except ValueError:
                    target_coverage_val = 0.9 # Default if env var is invalid

                for fm_info in relevant_focused_methods_for_this_class:
                    methods_str_parts.append(
                        f"- Method: `{fm_info.get('method_name', 'N/A')}` (Signature: `{fm_info.get('method_signature', 'N/A')}` "
                        f"Current Coverage: {fm_info.get('line_coverage', 0.0)*100:.1f}%, "
                        f"Target: >{target_coverage_val*100:.0f}%)"
                    )
                focused_instruction = instruction_intro + "\n".join(methods_str_parts) + "\n--- END FOCUSED COVERAGE IMPROVEMENT ---\n"

                if current_additional_instructions:
                    current_additional_instructions += "\n" + focused_instruction
                else:
                    current_additional_instructions = focused_instruction

                self.logger.info(f"Added focused coverage instructions for {len(relevant_focused_methods_for_this_class)} methods in {target_class_name}.")

        base_template = self._get_base_prompt_template(
            target_class_name, target_package_name, custom_imports, current_additional_instructions
        )

        generated_code = ""
        for retry_attempt in range(MAX_TEST_GENERATION_RETRIES):
            print(f"\nAttempt {retry_attempt + 1}/{MAX_TEST_GENERATION_RETRIES} for test generation for {target_class_name}...")
 
            current_prompt_query = f"Generate tests for {target_class_name}."
            if retry_attempt > 0 and self.last_test_run_results:
    
                error_feedback_message = (
                    f"\n\n--- PREVIOUS ATTEMPT FEEDBACK ---\n"
                    f"The previously generated test for `{target_class_name}` encountered the following issue:\n"
                    f"Status: {self.last_test_run_results['status']}\n"
                    f"Message: {self.last_test_run_results['message']}\n"
                    f"STDOUT:\n{self.last_test_run_results['stdout']}\n"
                    f"STDERR:\n{self.last_test_run_results['stderr']}\n"
                    f"Please analyze the errors/failures and revise the test code to fix them."
                    f"Ensure you still adhere to all original instructions (Mockito usage, coverage, imports, etc.)."
                    f"\n--- END PREVIOUS ATTEMPT FEEDBACK ---\n"
                )
     
                template_with_feedback = base_template + error_feedback_message 
                QA_CHAIN_PROMPT = PromptTemplate.from_template(template_with_feedback)
            else:
                QA_CHAIN_PROMPT = PromptTemplate.from_template(base_template)

            if self.qa_chain: 
                self.qa_chain.combine_documents_chain.llm_chain.prompt = QA_CHAIN_PROMPT
            else: 
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
                    continue 

                generated_code = temp_generated_code 

               
                os.makedirs(test_output_file_path.parent, exist_ok=True)
                with open(test_output_file_path, 'w', encoding='utf-8') as f:
                    f.write(generated_code)
                print(f"Temporarily saved test file for execution: {test_output_file_path}")

                test_run_results = self.java_test_runner.run_test(test_output_file_path)
                self.last_test_run_results = test_run_results # Store for potential next iteration

                if test_run_results["status"] == "SUCCESS":
                    print(f"Test for {target_class_name} PASSED on attempt {retry_attempt + 1}.")
                    return generated_code # Test passed, return the code
                elif test_run_results["status"] == "FAILED" or test_run_results["status"] == "ERROR":
                    print(f"Test for {target_class_name} FAILED/ERRORED on attempt {retry_attempt + 1}. Message: {test_run_results['message']}")
                    print("Feeding error feedback to LLM for next attempt...")
     
                else: # UNKNOWN status
                    print(f"Test for {target_class_name} returned UNKNOWN status on attempt {retry_attempt + 1}. Message: {test_run_results['message']}")
                    print("Feeding unknown status feedback to LLM for next attempt...")
    

            except Exception as e:
                print(f"An error occurred during LLM call or test execution setup: {e}. Retrying...")
                self.last_test_run_results = {"status": "ERROR", "message": f"Internal generation/execution error: {e}", "stdout": "", "stderr": str(e)}
 
        print(f"Failed to generate a passing test for {target_class_name} after {MAX_TEST_GENERATION_RETRIES} attempts.")
        return generated_code

if __name__ == "__main__":
    try:
     
        BUILD_TOOL = os.getenv("BUILD_TOOL", "maven").lower() 
        test_generator = TestCaseGenerator(collection_name="code_chunks_collection", build_tool=BUILD_TOOL) 
        

        if not ANALYSIS_RESULTS_FILE.exists():
            print(f"ERROR: Analysis results file not found: {ANALYSIS_RESULTS_FILE}")
            print("Please run 'python3 src/analyzer/code_analyzer.py' first to generate analysis data.")
            sys.exit(1)

        print(f"\nLoading discovered targets from: {ANALYSIS_RESULTS_FILE}")
        with open(ANALYSIS_RESULTS_FILE, 'r', encoding='utf-8') as f:
            discovered_targets_metadata = json.load(f)
        
        print(f"Loaded {len(discovered_targets_metadata)} Spring Boot Service/Controller targets for test generation.")

        if not discovered_targets_metadata:
            print("No Spring Boot Service or Controller targets found in analysis results. Exiting test generation.")
            sys.exit(0) 

        for target_info in discovered_targets_metadata:
           
            java_file_path_abs = Path(target_info['java_file_path_abs'])
            relative_processed_txt_path = Path(target_info['relative_processed_txt_path']) 
            target_class_name = target_info['class_name']
            target_package_name = target_info['package_name']
            identified_dependencies_filenames = target_info['dependent_filenames']
            custom_imports_list = target_info['custom_imports'] 

            relevant_java_files_for_context = [java_file_path_abs.name] + identified_dependencies_filenames
            relevant_java_files_for_context = list(set(relevant_java_files_for_context)) # Ensure uniqueness
            
     
            paths = get_test_paths(str(relative_processed_txt_path), SPRING_BOOT_PROJECT_ROOT)
            test_output_dir = paths["test_output_dir"]
            test_output_file_path = paths["test_output_file_path"]

            print("\n" + "="*80)
            print(f"GENERATING TEST FOR: {target_class_name} ({target_package_name})")
            print(f"SOURCE FILE: {java_file_path_abs}")
            print(f"FILES TO RETRIEVE CHUNKS FROM (for context): {relevant_java_files_for_context}")
            print(f"EXTRACTED CUSTOM IMPORTS (for prompt): {custom_imports_list}")
            print(f"EXPECTED TEST OUTPUT PATH: '{test_output_file_path}'")
            print("="*80)

            # --- Generate the test case with feedback loop ---
            generated_test_code = test_generator.generate_test_case(
                target_class_name=target_class_name,
                target_package_name=target_package_name,
                custom_imports=custom_imports_list,
                relevant_java_files_for_context=relevant_java_files_for_context,
                test_output_file_path=test_output_file_path, # Pass the output path
                additional_query_instructions="and make sure there are no errors, and you don't cause mismatch in return types and stuff."
            )
            
            os.makedirs(test_output_dir, exist_ok=True)

            try:
                with open(test_output_file_path, 'w', encoding='utf-8') as f:
                    f.write(generated_test_code)
                print(f"\n[FINAL SUCCESS] Generated test case saved to: '{test_output_file_path}'")
            except Exception as e:
                print(f"\n[FINAL ERROR] Could not save test case to '{test_output_file_path}': {e}")

            print("\n--- FINAL GENERATED TEST CASE (Printed to Console for review) ---")
            print(generated_test_code)
            print("\n" + "="*80 + "\n")
            
    except ValueError as ve:
        print(f"Configuration Error: {ve}")
        print("Please ensure GOOGLE_API_KEY is set and other configurations are correct, especially file paths.")
    except Exception as e:
        print(f"An unexpected error occurred during main execution: {e}")
        print("Verify your ChromaDB setup and network connection for Google Generative AI API, and that file paths are correct.")


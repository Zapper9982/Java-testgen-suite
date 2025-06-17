import sys
from pathlib import Path
import os
import json 
from typing import List, Dict, Any, Union


TESTGEN_AUTOMATION_ROOT = Path(__file__).parent.parent.parent
SPRING_BOOT_PROJECT_ROOT_ENV = os.getenv("SPRING_BOOT_PROJECT_PATH")
if not SPRING_BOOT_PROJECT_ROOT_ENV:
    print("ERROR: SPRING_BOOT_PROJECT_PATH environment variable is not set.")
    print("Please set it in your run.sh script or before running test_case_generator.py.")
    sys.exit(1)
SPRING_BOOT_PROJECT_ROOT = Path(SPRING_BOOT_PROJECT_ROOT_ENV)
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


GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") 
if not GOOGLE_API_KEY:
    print("WARNING: GOOGLE_API_KEY environment variable not set. Please set it for Gemini API calls.")
    print("Example: export GOOGLE_API_KEY='your_google_api_key_here'")


LLM_MODEL_NAME_GEMINI = "gemini-1.5-flash" 

EMBEDDING_MODEL_NAME_BGE = "BAAI/bge-small-en-v1.5"
if torch.backends.mps.is_available():
    DEVICE_FOR_EMBEDDINGS = "mps"
    print("Detected Apple Silicon (M1/M2/M3). Using 'mps' device for embeddings.")
else:
    DEVICE_FOR_EMBEDDINGS = "cpu"
    print("Apple Silicon MPS not available or detected. Falling back to 'cpu' for embeddings.")

ANALYSIS_RESULTS_DIR = TESTGEN_AUTOMATION_ROOT / "analysis_results"
ANALYSIS_RESULTS_FILE = ANALYSIS_RESULTS_DIR / "spring_boot_targets.json"


MAX_TEST_GENERATION_RETRIES = 6

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
      
        self.retriever = self.vectorstore.as_retriever(
            search_type="mmr", 
            search_kwargs={"k": 15},
        )
        print("ChromaDB retriever instantiated (default settings).")

        self.llm = self._instantiate_llm()
        print("Google Gemini LLM instantiated for LangChain.")

        self.qa_chain = None 

        project_root_from_env = os.getenv("SPRING_BOOT_PROJECT_PATH")
        if not project_root_from_env:
            raise ValueError("SPRING_BOOT_PROJECT_PATH environment variable is not set. Cannot initialize JavaTestRunner.")
        self.java_test_runner = JavaTestRunner(project_root=Path(project_root_from_env), build_tool=build_tool)
        self.last_test_run_results = None 

    def _instantiate_llm(self) -> ChatGoogleGenerativeAI:
        if not GOOGLE_API_KEY:
            raise ValueError("GOOGLE_API_KEY environment variable is not set. Cannot initialize Gemini LLM.")
        
        print(f"Using Google Gemini LLM: {LLM_MODEL_NAME_GEMINI}...")
        return ChatGoogleGenerativeAI(model=LLM_MODEL_NAME_GEMINI, temperature=0.7)

    def _update_retriever_filter(self, filenames: Union[str, List[str]]):

        if isinstance(filenames, str):
            filter_filenames = [filenames]
        else:
            filter_filenames = filenames 

        self.retriever = self.vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": 15, 
                "filter": {"filename": {"$in": filter_filenames}}
            },
        )

        if self.qa_chain:
            self.qa_chain.retriever = self.retriever
        print(f"Retriever filter updated to target filenames: '{filter_filenames}'")

    def _get_base_prompt_template(self, target_class_name: str, target_package_name: str, custom_imports: List[str], additional_query_instructions: str) -> str:

        formatted_custom_imports = "\n".join(custom_imports)

        template = """
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
""".format(
            target_class_name=target_class_name,
            target_package_name=target_package_name,
            formatted_custom_imports=formatted_custom_imports,
            additional_query_instructions=additional_query_instructions
        )
        return template
    
    def generate_test_case(self, 
                           target_class_name: str, 
                           target_package_name: str, 
                           custom_imports: List[str],
                           relevant_java_files_for_context: List[str],
                           test_output_file_path: Path, # Added to save generated code
                           additional_query_instructions: str = "") -> str:
        """
        Generates a JUnit 5 test case by querying the RetrievalQA chain,
        with a dynamically constructed prompt, and includes a feedback loop for corrections.
        """

        self._update_retriever_filter(relevant_java_files_for_context)

        base_template = self._get_base_prompt_template(
            target_class_name, target_package_name, custom_imports, additional_query_instructions
        )

        generated_code = ""
        for retry_attempt in range(MAX_TEST_GENERATION_RETRIES):
            print(f"\nAttempt {retry_attempt + 1}/{MAX_TEST_GENERATION_RETRIES} for test generation for {target_class_name}...")
            

            current_prompt_query = f"Generate tests for {target_class_name}."
            if retry_attempt > 0 and self.last_test_run_results:

                stdout_content = self.last_test_run_results.get('stdout', '').replace('{', '{{').replace('}', '}}')
                stderr_content = self.last_test_run_results.get('stderr', '').replace('{', '{{').replace('}', '}}')

                error_feedback_message = (
                    "\n\n--- PREVIOUS ATTEMPT FEEDBACK ---\n"
                    "The previously generated test for `{target_class_name}` encountered the following issue:\n"
                    "Status: {status}\n"
                    "Message: {message}\n"
                    "STDOUT:\n{stdout_content}\n"
                    "STDERR:\n{stderr_content}\n"
                    "Please analyze the errors/failures and revise the test code to fix them."
                    "Ensure you still adhere to all original instructions (Mockito usage, coverage, imports, etc.)."
                    "\n--- END PREVIOUS ATTEMPT FEEDBACK ---\n"
                ).format(
                    target_class_name=target_class_name,
                    status=self.last_test_run_results.get('status', 'N/A'),
                    message=self.last_test_run_results.get('message', 'N/A'),
                    stdout_content=stdout_content,
                    stderr_content=stderr_content
                )

                template_with_feedback = base_template + error_feedback_message 
                QA_CHAIN_PROMPT = PromptTemplate.from_template(template_with_feedback)
            else:
                QA_CHAIN_PROMPT = PromptTemplate.from_template(base_template)

            if self.qa_chain:
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
               
                else: 
                    print(f"Test for {target_class_name} returned UNKNOWN status on attempt {retry_attempt + 1}. Message: {test_run_results['message']}")
                    print("Feeding unknown status feedback to LLM for next RegEx...")
                 

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
            relevant_java_files_for_context = list(set(relevant_java_files_for_context)) 

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

        print("\nInitiating full project test verification (mvn clean verify / gradle clean test)...")
        full_project_test_results = test_generator.java_test_runner.run_project_tests(is_full_verify=True)
        
        print("\n--- Full Project Test Verification Results ---")
        print(f"Status: {full_project_test_results['status']}")
        print(f"Message: {full_project_test_results['message']}")
        if full_project_test_results.get('summary'): 
            print(f"Summary: {full_project_test_results['summary']}")
        else:
            print("No detailed test summary available from build tool output.")
        
        if full_project_test_results['status'] != "SUCCESS":
            print("\nWARNING: Full project verification FAILED or had ERRORS. Check logs above.")
  
        print("\n--- Full Project Test Verification Completed ---")


    except ValueError as ve:
        print(f"Configuration Error: {ve}")
        print("Please ensure GOOGLE_API_KEY is set and other configurations are correct, especially file paths.")
    except Exception as e:
        print(f"An unexpected error occurred during main execution: {e}")
        print("Verify your ChromaDB setup and network connection for Google Generative AI API, and that file paths are correct.")

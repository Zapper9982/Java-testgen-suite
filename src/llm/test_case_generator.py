import sys
from pathlib import Path
import os
import json 
from typing import List, Dict, Any, Union
import re
from dotenv import load_dotenv
load_dotenv()
import javalang
import traceback
import subprocess




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
from analyzer.code_analysis_utils import extract_custom_imports_from_chunk_file, resolve_transitive_dependencies, resolve_dependency_path, build_global_class_map
from test_runner.java_test_runner import JavaTestRunner 

from chroma_db.chroma_client import get_chroma_client, get_or_create_collection
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI

import torch 
from chroma_db.chroma_client import get_chroma_client as get_chroma_client_examples, get_or_create_collection as get_or_create_collection_examples
# Import new batch prompt templates
from llm.prompt_templates import (
    get_controller_batch_prompt,
    get_service_batch_prompt,
    get_merge_prompt
)


# --- Google API Configuration ---
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") 
if not GOOGLE_API_KEY:
    print("WARNING: GOOGLE_API_KEY environment variable not set. Please set it for Gemini API calls.")
    print("Example: export GOOGLE_API_KEY='your_google_api_key_here'")


LLM_MODEL_NAME_GEMINI = "gemini-2.5-pro" 

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

# Control incremental merging behavior
ENABLE_INCREMENTAL_MERGE = os.getenv("ENABLE_INCREMENTAL_MERGE", "true").strip().lower() in {"1", "true", "yes", "y"}
KEEP_BATCH_FILES_ON_MERGE_FAILURE = os.getenv("KEEP_BATCH_FILES_ON_MERGE_FAILURE", "true").strip().lower() in {"1", "true", "yes", "y"}

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

        # --- RAG: Setup for test example retrieval ---
        self.test_examples_collection_name = "test_examples_collection"
        self.test_examples_chroma_client = get_chroma_client_examples()
        self.test_examples_vectorstore = Chroma(
            client=self.test_examples_chroma_client,
            collection_name=self.test_examples_collection_name,
            embedding_function=self.embeddings
        )

    def _instantiate_llm(self) -> ChatGoogleGenerativeAI:
        if not GOOGLE_API_KEY:
            raise ValueError("GOOGLE_API_KEY environment variable is not set. Cannot initialize Gemini LLM.")
        
        print(f"Using Google Gemini LLM: {LLM_MODEL_NAME_GEMINI}...")
        return ChatGoogleGenerativeAI(model=LLM_MODEL_NAME_GEMINI, temperature=0.1)

    def _update_retriever_filter(self, main_class_filename: str, dependency_filenames: List[str], utility_filenames: List[str] = None, k_override: int = None):
        """
        Updates the retriever's filter to target only the main class, its direct dependencies, and utility files if actually imported.
        Orders context as: main class, dependencies, utility files.
        Sets a lower default k for focused retrieval.
        """
        filter_filenames = [main_class_filename] + dependency_filenames
        if utility_filenames:
            filter_filenames += utility_filenames
        # Remove duplicates while preserving order
        seen = set()
        filter_filenames = [x for x in filter_filenames if not (x in seen or seen.add(x))]
        # Set a lower k for focused context
        k = min(5, len(filter_filenames))
        if k_override is not None:
            k = max(MIN_K, k_override)
        self.retriever = self.vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": k,
                "filter": {"filename": {"$in": filter_filenames}}
            },
        )
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

    



    def _get_direct_local_imports(self, class_code: str, project_root: Path) -> list:
        """
        Parse import statements from the class code and return a list of local Java filenames (e.g., 'UserAgentUtil.java') that are directly imported and start with 'com.iemr.'
        """
        import_pattern = re.compile(r'^import\s+(com\.iemr\.[\w\.]+);', re.MULTILINE)
        local_files = set()
        for match in import_pattern.finditer(class_code):
            fqcn = match.group(1)
            class_name = fqcn.split('.')[-1]
            local_files.add(f'{class_name}.java')
        return list(local_files)

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
        if test_output_file_path.exists():
            print(f"[SKIP] Test file already exists (inside generate_test_case): '{test_output_file_path}'")
            print(f"Skipping test generation for {target_class_name} (generate_test_case)")
            return ""

        main_class_filename = Path(target_info['java_file_path_abs']).name if target_info else None
        with open(target_info['java_file_path_abs'], 'r', encoding='utf-8') as f:
            main_class_code = f.read()
        main_class_code = strip_java_comments_and_trailing_whitespace(main_class_code)
        direct_local_deps = self._get_direct_local_imports(main_class_code, SPRING_BOOT_MAIN_JAVA_DIR)
        all_deps = direct_local_deps
        print(f"[STRICT DEBUG] Main class: {main_class_filename}, Direct local deps: {all_deps}")
        print(f"FILES TO RETRIEVE CHUNKS FROM (for context): {[main_class_filename] + all_deps}")
        files_for_context = [main_class_filename] + all_deps
        print(f"[DEBUG] Actually retrieving code for these files: {files_for_context}")
        context = f"--- BEGIN MAIN CLASS UNDER TEST ---\n"
        main_code = self._get_full_code_from_chromadb(main_class_filename)
        main_code = strip_java_comments_and_trailing_whitespace(main_code)
        context += main_code + "\n--- END MAIN CLASS UNDER TEST ---\n\n"
        for dep in all_deps:
            if dep == main_class_filename:
                continue
            dep_code = self._get_full_code_from_chromadb(dep)
            dep_code = strip_java_comments_and_trailing_whitespace(dep_code)
            context += f"--- BEGIN DEPENDENCY: {dep} ---\n{dep_code}\n--- END DEPENDENCY: {dep} ---\n\n"
        # Now, use 'context' in the prompt
        public_methods = extract_public_methods(target_info['java_file_path_abs'])
        test_type = self._detect_test_type(target_info) if target_info else None
        if test_type is None:
            test_type = 'service'  # fallback
        if test_type == 'controller':
            prompt_template = get_controller_test_prompt_template(
                target_class_name, target_package_name, custom_imports, additional_query_instructions
            )
            # Add strict instruction to not generate code for dependencies
            prompt_template += "\nSTRICT: Do NOT generate any code for dependencies. Only generate the test class for the controller. Assume all dependencies exist and are available for mocking.\n"
        else:
            prompt_template = get_service_test_prompt_template(
                target_class_name, target_package_name, custom_imports, additional_query_instructions
            )
        max_retries = MAX_TEST_GENERATION_RETRIES
        error_feedback = None
        last_valid_code = None
        for attempt in range(max_retries):
            prompt = prompt_template.replace('{context}', context)
            if error_feedback and attempt > 0:
                prompt += f"\n--- ERROR FEEDBACK FROM PREVIOUS ATTEMPT ---\n{error_feedback}\n--- END ERROR FEEDBACK ---\n"
            result = self.llm.invoke(prompt)
            if hasattr(result, "content"):
                result = result.content
            code = result.strip()
            code = re.sub(r'^```[a-zA-Z]*\n', '', code)
            code = re.sub(r'^```', '', code)
            code = re.sub(r'```$', '', code)
            code = code.strip()
            # Write to file
            test_output_file_path.parent.mkdir(parents=True, exist_ok=True)
            with open(test_output_file_path, 'w', encoding='utf-8') as f:
                f.write(code)
      
            test_run_results = self.java_test_runner.run_test(test_output_file_path)
            compilation_errors = test_run_results['detailed_errors'].get('compilation_errors', [])
            test_failures = test_run_results['detailed_errors'].get('test_failures', [])
            tests_run_zero = False
            stdout = test_run_results.get('stdout', '')
            test_summary_match = re.search(r"Tests run: (\d+), Failures: (\d+), Errors: (\d+), Skipped: (\d+)", stdout)
            if test_summary_match:
                total = int(test_summary_match.group(1))
                if total == 0:
                    tests_run_zero = True
                    print("[WARNING] No tests were run (Tests run: 0). Treating as failure.")
            if not compilation_errors and not test_failures and not tests_run_zero:
                print(f"[SUCCESS] No compilation or test errors after {attempt+1} attempt(s).")
                last_valid_code = code
                break
            else:
                error_msgs = []
                if compilation_errors:
                    print(f"[COMPILATION ERROR] Detected after attempt {attempt+1}:")
                    for err in compilation_errors:
                        print(f"  - {err['message']} (at {err['location']})")
                        error_msgs.append(f"COMPILATION: {err['message']} (at {err['location']})")
                if test_failures:
                    print(f"[TEST FAILURE] Detected after attempt {attempt+1}:")
                    for err in test_failures:
                        print(f"  - {err['message']} (in {err['location']})")
                        error_msgs.append(f"TEST: {err['message']} (in {err['location']})")
                if tests_run_zero:
                    error_msgs.append("NO TESTS RUN: The generated test class did not contain any executable tests. Ensure at least one @Test method is present and not ignored/skipped.")
                full_compilation_error = self.extract_full_compilation_error(stdout)
                print("\n[DEBUG] FULL COMPILATION ERROR BLOCK EXTRACTED:\n" + (full_compilation_error or '[EMPTY]') + "\n[END DEBUG FULL COMPILATION ERROR BLOCK]\n")
                error_feedback = '\n'.join(error_msgs)
                if full_compilation_error:
                    error_feedback += '\n\n--- FULL COMPILATION ERROR OUTPUT ---\n' + full_compilation_error + '\n--- END FULL COMPILATION ERROR OUTPUT ---\n'
                retries += 1
        if last_valid_code:
            print(f"[FINAL SUCCESS] Generated test case saved to: '{test_output_file_path}'")
            return last_valid_code
        else:
            print(f"[ERROR] Test generation failed for {target_class_name} after {max_retries} attempts. See logs for details.")
            return "// Test not generated: failed after feedback loop."

    def generate_test_case_in_batches(
        self,
        target_class_name: str,
        target_package_name: str,
        custom_imports: List[str],
        relevant_java_files_for_context: List[str],
        test_output_file_path: Path,
        additional_query_instructions: str,
        requires_db_test: bool,
        dependency_signatures: Dict[str, str] = None,
        target_info: Dict[str, Any] = None
    ) -> str:
        """
        Batch mode: For every batch, send the full code of the main class and only directly imported local dependencies (from ChromaDB) in the prompt context.
        Also print the full prompt before each LLM call for debugging/verification.
        """
        if test_output_file_path.exists():
            print(f"[SKIP] Test file already exists (inside generate_test_case_in_batches): '{test_output_file_path}'")
            print(f"Skipping test generation for {target_class_name} (generate_test_case_in_batches)")
            return ""

        main_class_filename = Path(target_info['java_file_path_abs']).name if target_info else None
        with open(target_info['java_file_path_abs'], 'r', encoding='utf-8') as f:
            main_class_code = f.read()
        main_class_code = strip_java_comments_and_trailing_whitespace(main_class_code)
        direct_local_deps = self._get_direct_local_imports(main_class_code, SPRING_BOOT_MAIN_JAVA_DIR)
        all_deps = direct_local_deps
        print(f"[STRICT DEBUG] Main class: {main_class_filename}, Direct local deps: {all_deps}")
        print(f"FILES TO RETRIEVE CHUNKS FROM (for context): {[main_class_filename] + all_deps}")
        files_for_context = [main_class_filename] + all_deps
        print(f"[DEBUG] Actually retrieving code for these files: {files_for_context}")
        test_type = self._detect_test_type(target_info) if target_info else None
        if test_type is None:
            test_type = 'service'  # fallback
        full_context = f"--- BEGIN MAIN CLASS UNDER TEST ---\n"
        # Use the full .java file for the main class code
        with open(target_info['java_file_path_abs'], 'r', encoding='utf-8') as f:
            main_code = f.read()
        main_code = strip_java_comments_and_trailing_whitespace(main_code)
        full_context += main_code + "\n--- END MAIN CLASS UNDER TEST ---\n\n"
        if test_type != "controller":
            for dep in all_deps:
                if dep == main_class_filename:
                    continue
                dep_path = resolve_dependency_path(dep, main_code, SPRING_BOOT_MAIN_JAVA_DIR)
                print(f"[DEBUG] Dependency: {dep}, Resolved path: {dep_path}")
                if dep_path and os.path.exists(dep_path):
                    dep_signatures = extract_class_signatures(dep_path)
                else:
                    print(f"[WARNING] Could not find dependency file: {dep}")
                    dep_signatures = f"// Dependency not found: {dep}"
                dep_signatures = strip_java_comments_and_trailing_whitespace(dep_signatures)
                full_context += f"--- BEGIN DEPENDENCY SIGNATURES: {dep} ---\n{dep_signatures}\n--- END DEPENDENCY SIGNATURES: {dep} ---\n\n"
        # Extract public methods and endpoint map from the main class code only (not from full_context)
        with open(target_info['java_file_path_abs'], 'r', encoding='utf-8') as f:
            main_class_code = f.read()
        # Use javalang to extract both public methods and endpoint paths
        public_methods = set()
        endpoint_map = {}
        try:
            tree = javalang.parse.parse(main_class_code)
            main_class = None
            for type_decl in tree.types:
                if isinstance(type_decl, javalang.tree.ClassDeclaration):
                    main_class = type_decl
                    break
            if main_class:
                for method in main_class.methods:
                    if 'public' in method.modifiers:
                        public_methods.add(method.name)
                        # Look for REST endpoint annotations
                        endpoint = None
                        if method.annotations:
                            for ann in method.annotations:
                                ann_name = ann.name.lower()
                                if ann_name in [
                                    'getmapping', 'postmapping', 'putmapping', 'deletemapping', 'patchmapping', 'requestmapping'
                                ]:
                                    # Try to extract the path value
                                    if ann.element:
                                        # Handles @GetMapping(path = "/foo") or @GetMapping("/foo")
                                        if hasattr(ann.element, 'value') and ann.element.value:
                                            endpoint = ann.element.value.value if hasattr(ann.element.value, 'value') else str(ann.element.value)
                                        elif hasattr(ann.element, 'pairs') and ann.element.pairs:
                                            for pair in ann.element.pairs:
                                                if pair.name == 'path' or pair.name == 'value':
                                                    endpoint = pair.value.value if hasattr(pair.value, 'value') else str(pair.value)
                                    elif ann.element is not None:
                                        endpoint = str(ann.element)
                        if endpoint:
                            endpoint_map[method.name] = endpoint
        except Exception as e:
            print(f"[extract_public_methods] javalang parse error: {e}")
        public_methods = list(public_methods)
        if not public_methods:
            print("[ERROR] No public methods found for batch mode generation.")
            return "// No public methods found to test."
        batch_size = 3
        batches = [public_methods[i:i+batch_size] for i in range(0, len(public_methods), batch_size)]
        
        # Initialize incremental merging variables
        merged_test_class = None
        merged_test_file_path = test_output_file_path  # The final merged file
        
        final_code = None
        for i, batch in enumerate(batches):
            # --- ADDED: Check for batch method name mismatches ---
            batch_set = set(batch)
            public_methods_set = set(public_methods)
            if not batch_set.issubset(public_methods_set):
                missing = batch_set - public_methods_set
                print(f"[WARNING] Batch {i+1} contains method names not in extracted public methods: {missing}")
            print(f"\n[INFO] Generating test case for batch {i+1}/{len(batches)} with these public methods: {batch}\n")
            # Build endpoints list for this batch (force-apply fix)
            endpoints_list = []
            for m in batch:
                if m in endpoint_map:
                    endpoints_list.append(f"- {m}: {endpoint_map[m]}")
            endpoints_list_str = '\n'.join(endpoints_list)
            print(f"[BATCH MODE] Generating tests for methods: {batch}")
            method_list_str = '\n'.join(f'- {m}' for m in batch)
            retries = 0  # Always define retries at the start of the batch
            success = False
            error_feedback = None
            # --- NEW: Use minimal class code for batch ---
            with open(target_info['java_file_path_abs'], 'r', encoding='utf-8') as f:
                main_code = f.read()
            main_code = strip_java_comments_and_trailing_whitespace(main_code)
            # Extract import statements
            import_lines = [line for line in main_code.splitlines() if line.strip().startswith('import ')]
            imports_section = ''
            if import_lines:
                imports_section = '--- BEGIN IMPORTS ---\n' + '\n'.join(import_lines) + '\n--- END IMPORTS ---\n\n'
            minimal_class_code = None
            try:
                minimal_class_code = extract_minimal_class_for_methods(target_info['java_file_path_abs'], batch)
            except Exception as e:
                print(f"[ERROR] Failed to extract minimal class code for batch {i+1}: {e}")
                traceback.print_exc()
                minimal_class_code = main_code  # fallback
            # --- PATCH: Only include dependency signatures for types referenced in minimal class ---
            referenced_types = extract_referenced_types(target_info['java_file_path_abs'], batch)
            dep_signatures = []
            included_deps = []
            
            # Extract types that are actually used in the minimal class code
            minimal_class_types = set()
            if minimal_class_code:
                # Look for field declarations and method calls in the minimal class
                lines = minimal_class_code.split('\n')
                for line in lines:
                    line = line.strip()
                    # Look for field declarations like: private SomeType fieldName;
                    if line.startswith('private ') or line.startswith('@Autowired private ') or line.startswith('@PersistenceContext private '):
                        parts = line.split()
                        if len(parts) >= 3:
                            # Handle @Autowired private SomeType fieldName;
                            if line.startswith('@Autowired private '):
                                type_name = parts[2]  # The type name after @Autowired private
                            else:
                                type_name = parts[1]  # The type name after private
                            if type_name and not type_name.startswith('static') and not type_name.startswith('final'):
                                minimal_class_types.add(type_name)
                    # Look for method calls like: someType.method()
                    elif '.' in line and '(' in line:
                        parts = line.split('.')
                        if len(parts) >= 2:
                            field_name = parts[0].strip()
                            if field_name and not field_name.startswith('//'):
                                # Try to find the type of this field
                                for ref_type in referenced_types:
                                    if ref_type.lower().endswith(field_name.lower()) or field_name.lower() in ref_type.lower():
                                        minimal_class_types.add(ref_type)
                                        break
            
            # Only include dependency signatures for types actually used in minimal class
            for dep_type in referenced_types:
                if dep_type in minimal_class_types:
                    dep_path = resolve_dependency_path(dep_type + '.java', main_code, SPRING_BOOT_MAIN_JAVA_DIR)
                    if dep_path and os.path.exists(dep_path):
                        dep_sign = extract_class_signatures(dep_path)
                        dep_signatures.append(f"--- BEGIN DEPENDENCY SIGNATURES: {dep_type} ---\n{dep_sign}\n--- END DEPENDENCY SIGNATURES: {dep_type} ---\n")
                        included_deps.append(dep_type)
            
            if included_deps:
                print(f"[BATCH {i+1}] Including dependency signatures in prompt: {included_deps}")
            else:
                print(f"[BATCH {i+1}] No dependency signatures included (no local dependencies found)")
            minimal_context = imports_section
            minimal_context += f"--- BEGIN MAIN CLASS UNDER TEST (MINIMAL) ---\n{minimal_class_code}\n--- END MAIN CLASS UNDER TEST (MINIMAL) ---\n\n"
            minimal_context += "\n".join(dep_signatures)
            strict_no_comments = '\nSTRICT: Do NOT add any comments to the generated code. and MAKE SURE that the code is JUNIT5 + MOCKITO STYLE FOR SERVICES\n'
            minimal_context += strict_no_comments
            
            # Update package name to reflect the new folder structure
            batch_package_name = target_package_name  # Use original package name, not with Test suffix
            
            # Define the batch test file path in the same directory as the final test file
            batch_test_output_path = test_output_file_path.parent / f"{target_class_name}Test_Batch{i+1}.java"
            
            print(f"[DEBUG] test_type: {test_type}, batch: {i+1}/{len(batches)}, retry: {retries}")
            while retries < 15 and not success:
                context = minimal_context  # Use minimal context for every batch
                previous_test_code = None
                # Always read the current batch test file for every retry (not just for new batches)
                try:
                    with open(batch_test_output_path, 'r', encoding='utf-8') as f:
                        previous_test_code = f.read()
                except Exception:
                    previous_test_code = None
                
                # Generate prompt using template functions
                if test_type == "controller":
                    prompt = get_controller_batch_prompt(
                        target_class_name=target_class_name,
                        target_package_name=batch_package_name,
                        batch_number=i+1,
                        method_list_str=method_list_str,
                        context=context,
                        previous_test_code=previous_test_code,
                        error_feedback=error_feedback if retries > 0 else None
                    )
                else:
                    prompt = get_service_batch_prompt(
                        target_class_name=target_class_name,
                        target_package_name=batch_package_name,
                        batch_number=i+1,
                        method_list_str=method_list_str,
                        context=context,
                        previous_test_code=previous_test_code,
                        error_feedback=error_feedback if retries > 0 else None
                    )
                
                # Print the prompt for every batch attempt
                print(f"\n[BATCH {i+1} RETRY {retries}] PROMPT SENT TO LLM:\n" + prompt + "\n[END PROMPT]\n")
                # Debug print: show minimal class code for this batch
                print(f"[DEBUG] Minimal class code for batch {i+1}, retry {retries} (methods: {batch}):\n--- BEGIN MINIMAL CLASS CODE ---\n{minimal_class_code}\n--- END MINIMAL CLASS CODE ---\n")
                result = self.llm.invoke(prompt)
                if hasattr(result, "content"):
                    result = result.content
                code = result.strip()
                code = re.sub(r'^```[a-zA-Z]*\n', '', code)
                code = re.sub(r'^```', '', code)
                code = re.sub(r'```$', '', code)
                code = code.strip()
                
                # Implement incremental merging logic
                if ENABLE_INCREMENTAL_MERGE:
                    if i == 0:
                        # First batch: initialize merged test class with correct class name
                        merged_test_class = self._fix_class_name_for_merged_file(code, target_class_name)
                        print(f"[MERGE] Initialized merged test class for {target_class_name} with correct class name")
                    else:
                        # Subsequent batches: LLM merge
                        print(f"[MERGE] Merging batch {i+1} into existing test class...")
                        try:
                            merged_test_class = self.merge_batch_with_existing_test_class(
                                merged_test_class, code, target_class_name, target_package_name, test_type
                            )
                            print(f"[MERGE] Successfully merged batch {i+1}")
                        except Exception as merge_error:
                            print(f"[MERGE][ERROR] Failed to merge batch {i+1}: {merge_error}")
                            # Fallback: keep batch separate
                            merged_test_class = code
                    
                    # Ensure directories exist before writing files
                    merged_test_file_path.parent.mkdir(parents=True, exist_ok=True)
                    batch_test_output_path.parent.mkdir(parents=True, exist_ok=True)
                    
                    # Write merged class to final file
                    with open(merged_test_file_path, 'w', encoding='utf-8') as f:
                        f.write(merged_test_class)
                    
                    # Also write to batch file for debugging
                    with open(batch_test_output_path, 'w', encoding='utf-8') as f:
                        f.write(code)
                    
                    # Run tests on merged file (not batch file)
                    test_run_results = self.java_test_runner.run_test(merged_test_file_path)
                else:
                    # Original behavior: write to batch file only
                    # Ensure directory exists before writing file
                    batch_test_output_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(batch_test_output_path, 'w', encoding='utf-8') as f:
                        f.write(code)
                    # Run the batch test file
                    test_run_results = self.java_test_runner.run_test(batch_test_output_path)
                
                compilation_errors = test_run_results['detailed_errors'].get('compilation_errors', [])
                test_failures = test_run_results['detailed_errors'].get('test_failures', [])
                tests_run_zero = False
                stdout = test_run_results.get('stdout', '')
                test_summary_match = re.search(r"Tests run: (\d+), Failures: (\d+), Errors: (\d+), Skipped: (\d+)", stdout)
                if test_summary_match:
                    total = int(test_summary_match.group(1))
                    if total == 0:
                        tests_run_zero = True
                        print("[WARNING] No tests were run (Tests run: 0). Treating as failure.")
                if not compilation_errors and not test_failures and not tests_run_zero:
                    print(f"[SUCCESS] Batch {i+1}/{len(batches)}: No compilation or test errors.")
                    success = True
                    
                    # Clean up batch file after successful generation and merge
                    try:
                        if batch_test_output_path.exists():
                            batch_test_output_path.unlink()  # Delete the batch file
                            print(f"[CLEANUP] Deleted temporary batch file: {batch_test_output_path.name}")
                    except Exception as cleanup_error:
                        print(f"[CLEANUP][WARNING] Failed to delete batch file {batch_test_output_path}: {cleanup_error}")
                else:
                    error_msgs = []
                    if compilation_errors:
                        print(f"[COMPILATION ERROR] Detected in batch {i+1}:")
                        for err in compilation_errors:
                            print(f"  - {err['message']} (at {err['location']})")
                            error_msgs.append(f"COMPILATION: {err['message']} (at {err['location']})")
                    if test_failures:
                        print(f"[TEST FAILURE] Detected in batch {i+1}:")
                        for err in test_failures:
                            print(f"  - {err['message']} (in {err['location']})")
                            error_msgs.append(f"TEST: {err['message']} (in {err['location']})")
                    if tests_run_zero:
                        error_msgs.append("NO TESTS RUN: The generated test class did not contain any executable tests. Ensure at least one @Test method is present and not ignored/skipped.")
                    full_compilation_error = self.extract_full_compilation_error(stdout)
                    print("\n[DEBUG] FULL COMPILATION ERROR BLOCK EXTRACTED (BATCH):\n" + (full_compilation_error or '[EMPTY]') + "\n[END DEBUG FULL COMPILATION ERROR BLOCK]\n")
                    error_feedback = '\n'.join(error_msgs)
                    error_block = extract_error_block_after_results(stdout)
                    if error_block:
                        error_feedback += '\n\n--- ERROR BLOCK AFTER [INFO] Results: ---\n' + error_block + '\n--- END ERROR BLOCK ---\n'
                    retries += 1
            # Update final_code based on merging mode
            if ENABLE_INCREMENTAL_MERGE and merged_test_class:
                final_code = merged_test_class
            else:
                # Fallback to reading the last batch file
                with open(batch_test_output_path, 'r', encoding='utf-8') as f:
                    final_code = f.read()
        # After all batches:
        return final_code

    def merge_batch_with_existing_test_class(
        self, 
        existing_test_class: str, 
        new_batch_code: str, 
        target_class_name: str,
        target_package_name: str,
        test_type: str
    ) -> str:
        """
        LLM-powered merge of new batch into existing test class.
        Returns the merged test class code.
        """
        merge_prompt = get_merge_prompt(
            existing_test_class=existing_test_class,
            new_batch_code=new_batch_code,
            target_class_name=target_class_name,
            target_package_name=target_package_name,
            test_type=test_type
        )
        
        # Add retry logic for API failures
        max_merge_retries = 3
        for attempt in range(max_merge_retries):
            try:
                print(f"[MERGE] Attempt {attempt + 1}/{max_merge_retries} to merge batch...")
                result = self.llm.invoke(merge_prompt)
                if hasattr(result, "content"):
                    result = result.content
                
                # Debug: Show raw LLM response before cleaning
                print(f"[MERGE][DEBUG] Raw LLM response (first 200 chars): {result[:200]}")
                if result.startswith('```'):
                    print("[MERGE][DEBUG] WARNING: LLM response starts with backticks - will clean them up")
                
                # Clean up markdown code block backticks (same as in batch generation)
                merged_code = result.strip()
                merged_code = re.sub(r'^```[a-zA-Z]*\n', '', merged_code)
                merged_code = re.sub(r'^```', '', merged_code)
                merged_code = re.sub(r'```$', '', merged_code)
                merged_code = merged_code.strip()
                
                # Debug: Show cleaned response
                print(f"[MERGE][DEBUG] Cleaned response (first 200 chars): {merged_code[:200]}")
                
                # Check if we got a valid response
                if merged_code and len(merged_code) > 50:  # Basic sanity check
                    print(f"[MERGE] Successfully merged on attempt {attempt + 1}")
                    return merged_code
                else:
                    print(f"[MERGE][WARNING] Got empty or too short response on attempt {attempt + 1}")
                    if attempt == max_merge_retries - 1:
                        print("[MERGE][ERROR] All merge attempts failed - returning original code")
                        return existing_test_class  # Fallback to existing code
                    
            except Exception as e:
                print(f"[MERGE][ERROR] Attempt {attempt + 1} failed with error: {e}")
                if "InternalServerError" in str(e) or "500" in str(e):
                    print("[MERGE] Google API internal server error - will retry...")
                    if attempt < max_merge_retries - 1:
                        import time
                        wait_time = (attempt + 1) * 2  # Exponential backoff
                        print(f"[MERGE] Waiting {wait_time} seconds before retry...")
                        time.sleep(wait_time)
                        continue
                    else:
                        print("[MERGE][ERROR] Max retries reached - returning existing test class")
                        return existing_test_class
                else:
                    # Non-retryable error
                    print(f"[MERGE][ERROR] Non-retryable error: {e}")
                    return existing_test_class
        
        # Should not reach here, but just in case
        return existing_test_class

    def _fix_class_name_for_merged_file(self, batch_code: str, target_class_name: str) -> str:
        """
        Fix the class name in batch code to be the final merged class name (without Batch suffix).
        Changes UserServiceTest_Batch1 to UserServiceTest.
        """
        import re
        
        # Pattern to match class declaration with Batch suffix
        class_pattern = rf'class\s+{re.escape(target_class_name)}Test_Batch\d+'
        
        # Find the original class name for logging
        original_match = re.search(class_pattern, batch_code)
        if original_match:
            original_class_name = original_match.group()
            print(f"[CLASS_NAME_FIX] Changing class name from '{original_class_name}' to '{target_class_name}Test'")
        else:
            print(f"[CLASS_NAME_FIX] No batch class name found to replace in code")
        
        # Replace with the final class name
        fixed_code = re.sub(class_pattern, f'class {target_class_name}Test', batch_code)
        
        return fixed_code

    def extract_public_methods(self, class_code: str) -> set:
        """
        Use javalang to robustly extract all public method names from the main class only.
        """
        try:
            tree = javalang.parse.parse(class_code)
        except Exception as e:
            print(f"[extract_public_methods] javalang parse error: {e}")
            return set()
        # Find the main class (the first class declaration)
        main_class = None
        for type_decl in tree.types:
            if isinstance(type_decl, javalang.tree.ClassDeclaration):
                main_class = type_decl
                break
        if not main_class:
            print("[extract_public_methods] Could not find main class declaration.")
            return set()
        methods = set()
        for method in main_class.methods:
            if 'public' in method.modifiers:
                methods.add(method.name)
        return methods



    def extract_full_compilation_error(self, stdout: str) -> str:
        """
        Extract the full compilation error block from Maven/Gradle output.
        Returns the block from '[ERROR] COMPILATION ERROR' to the next '[INFO]' or end of error section.
        If that block is too short, fallback to collecting all lines starting with [ERROR] and the next 2 lines after each for context.
        """
        error_block = []
        in_error = False
        for line in stdout.splitlines():
            if '[ERROR] COMPILATION ERROR' in line:
                in_error = True
            if in_error:
                error_block.append(line)
                # End block at next [INFO] (but not the starting line)
                if '[INFO]' in line and len(error_block) > 1:
                    break
        # If the block is too short (just header), fallback to all [ERROR] lines and their context
        if len(error_block) <= 2:
            error_block = []
            lines = stdout.splitlines()
            i = 0
            while i < len(lines):
                if lines[i].startswith('[ERROR]'):
                    # Add this line and the next 2 lines for context
                    error_block.append(lines[i])
                    if i+1 < len(lines):
                        error_block.append(lines[i+1])
                    if i+2 < len(lines):
                        error_block.append(lines[i+2])
                i += 1
        return '\n'.join(error_block) if error_block else stdout

# --- Helper: Extract class signatures (public/protected methods, fields, constructors) ---
def extract_class_signatures(java_code: str, file_path: str = None) -> str:
    """
    Given full Java class code, return a string with only the class declaration and public/protected method/field/constructor signatures.
    """
    try:
        tree = javalang.parse.parse(java_code)
    except Exception as e:
        print(f"[extract_class_signatures] javalang parse error in {file_path or 'unknown file'}: {e}")
        traceback.print_exc()
        return java_code  # fallback: return original code
    output = []
    for type_decl in tree.types:
        if isinstance(type_decl, javalang.tree.ClassDeclaration):
            class_decl = f"public class {type_decl.name}"
            if type_decl.extends:
                class_decl += f" extends {type_decl.extends.name}"
            if type_decl.implements:
                impls = ', '.join(i.name for i in type_decl.implements)
                class_decl += f" implements {impls}"
            class_decl += " {"
            output.append(class_decl)
            # Fields
            for field in type_decl.fields:
                mods = ' '.join(sorted(field.modifiers & {'public', 'protected'}))
                if not mods:
                    continue
                decl = f"    {mods} {field.type.name if hasattr(field.type, 'name') else str(field.type)}"
                decl += ' ' + ', '.join(d.name for d in field.declarators) + ';'
                output.append(decl)
            # Constructors
            for ctor in type_decl.constructors:
                mods = ' '.join(sorted(ctor.modifiers & {'public', 'protected'}))
                if not mods:
                    continue
                params = ', '.join(
                    (p.type.name if hasattr(p.type, 'name') else str(p.type)) + ("[]" if p.type.dimensions else "") + " " + p.name
                    for p in ctor.parameters
                )
                output.append(f"    {mods} {ctor.name}({params});")
            # Methods
            for method in type_decl.methods:
                mods = ' '.join(sorted(method.modifiers & {'public', 'protected'}))
                if not mods:
                    continue
                params = ', '.join(
                    (p.type.name if hasattr(p.type, 'name') else str(p.type)) + ("[]" if p.type.dimensions else "") + " " + p.name
                    for p in method.parameters
                )
                ret_type = method.return_type.name if method.return_type and hasattr(method.return_type, 'name') else (str(method.return_type) if method.return_type else 'void')
                output.append(f"    {mods} {ret_type} {method.name}({params});")
            output.append("}")
            break  # Only first class
    return '\n'.join(output) if output else java_code

def extract_minimal_class_for_methods(java_code: str, method_names: list, file_path: str = None) -> str:
    import re
    try:
        tree = javalang.parse.parse(java_code)
    except Exception as e:
        print(f"[extract_minimal_class_for_methods] javalang parse error in {file_path or 'unknown file'}: {e}")
        traceback.print_exc()
        raise RuntimeError(f"Failed to parse Java code for minimal class extraction: {e}")
    # Find the main class
    main_class = None
    for type_decl in tree.types:
        if isinstance(type_decl, javalang.tree.ClassDeclaration):
            main_class = type_decl
            break
    if not main_class:
        print("[extract_minimal_class_for_methods] Could not find main class declaration.")
        return 'public class Dummy {}'
    # Collect class-level annotations
    class_annos = []
    lines = java_code.splitlines()
    if hasattr(main_class, 'annotations') and main_class.annotations:
        for anno in main_class.annotations:
            if hasattr(anno, 'position') and anno.position:
                start = anno.position[0] - 1
                class_annos.append(lines[start])
            else:
                class_annos.append(f"@{anno.name}")
    # Class declaration line and its index
    class_decl_line = None
    class_decl_idx = None
    for i, line in enumerate(lines):
        if re.match(r'.*class\s+' + re.escape(main_class.name) + r'\b', line):
            class_decl_line = line
            class_decl_idx = i
            break
    if not class_decl_line:
        class_decl_line = f"public class {main_class.name} {{"
        class_decl_idx = 0
    # Find the end of the class (matching closing brace)
    brace_count = 0
    class_end_idx = None
    for i in range(class_decl_idx, len(lines)):
        brace_count += lines[i].count('{')
        brace_count -= lines[i].count('}')
        if brace_count == 0 and i > class_decl_idx:
            class_end_idx = i
            break
    if class_end_idx is None:
        class_end_idx = len(lines) - 1
    # Extract all lines inside the class body (excluding methods/constructors)
    field_lines = []
    inside_method = False
    method_or_ctor_pattern = re.compile(r'\s*(public|private|protected)?\s*[\w<>\[\]]+\s+\w+\s*\([^)]*\)\s*(throws [\w, ]+)?\s*\{')
    for i in range(class_decl_idx + 1, class_end_idx):
        line = lines[i]
        # Detect method or constructor start
        if method_or_ctor_pattern.match(line):
            inside_method = True
        if inside_method:
            # Look for end of method/constructor
            if '{' in line or '}' in line:
                # Count braces to find method end
                method_brace_count = line.count('{') - line.count('}')
                while method_brace_count != 0 and i < class_end_idx:
                    i += 1
                    next_line = lines[i]
                    method_brace_count += next_line.count('{') - next_line.count('}')
                inside_method = False
            continue
        # Otherwise, this is a field or annotation or blank/comment
        if line.strip() and not line.strip().startswith('//'):
            field_lines.append(line)
    # Map method name to method node
    method_map = {m.name: m for m in main_class.methods}
    # Helper: get method source by line numbers, including annotations
    def get_method_src_with_annotations(method):
        if hasattr(method, 'position') and method.position:
            start = method.position[0] - 1
            # Look upwards for annotations
            anno_lines = []
            idx = start - 1
            while idx >= 0 and lines[idx].strip().startswith('@'):
                anno_lines.insert(0, lines[idx])
                idx -= 1
            # Now, get the method body as before
            brace_count = 0
            method_lines = []
            started = False
            for l in lines[start:]:
                if '{' in l:
                    brace_count += l.count('{')
                    started = True
                if '}' in l:
                    brace_count -= l.count('}')
                method_lines.append(l)
                if started and brace_count == 0:
                    break
            return '\n'.join(anno_lines + method_lines)
        # fallback: reconstruct signature and body
        mods = ' '.join(method.modifiers)
        ret_type = method.return_type.name if method.return_type and hasattr(method.return_type, 'name') else (str(method.return_type) if method.return_type else 'void')
        params = ', '.join(
            (p.type.name if hasattr(p.type, 'name') else str(p.type)) + ("[]" if p.type.dimensions else "") + " " + p.name
            for p in method.parameters
        )
        header = f"    {mods} {ret_type} {method.name}({params}) "
        return header + "{ ... }"
    # Find all methods to include: batch + directly called helpers (recursively)
    to_include = set(method_names)
    included = set()
    def add_called_helpers(method):
        if method.name in included:
            return
        included.add(method.name)
        # Find all method invocations in the method body
        if hasattr(method, 'body') and method.body:
            for path, node in method:
                if isinstance(node, javalang.tree.MethodInvocation):
                    if node.member in method_map and node.member not in to_include:
                        to_include.add(node.member)
                        add_called_helpers(method_map[node.member])
    for m in method_names:
        if m in method_map:
            add_called_helpers(method_map[m])
    # Collect method sources (with annotations)
    method_lines = []
    referenced_inner_classes = set()
    for m in main_class.methods:
        if m.name in to_include:
            src = get_method_src_with_annotations(m)
            method_lines.append(src)
    # Find all inner class names in the main class
    inner_class_nodes = [node for node in main_class.body if isinstance(node, javalang.tree.ClassDeclaration)]
    inner_class_names = {ic.name for ic in inner_class_nodes}
    # Find referenced inner classes in all included methods
    for src in method_lines:
        for cname in inner_class_names:
            if re.search(r'\b' + re.escape(cname) + r'\b', src):
                referenced_inner_classes.add(cname)
    # Collect inner class code
    inner_class_code_blocks = []
    for ic in inner_class_nodes:
        if ic.name in referenced_inner_classes:
            if hasattr(ic, 'position') and ic.position:
                start = ic.position[0] - 1
                lines_ic = lines[start:]
                brace_count = 0
                class_lines = []
                started = False
                for l in lines_ic:
                    if '{' in l:
                        brace_count += l.count('{')
                        started = True
                    if '}' in l:
                        brace_count -= l.count('}')
                    class_lines.append(l)
                    if started and brace_count == 0:
                        break
                inner_class_code_blocks.append('\n'.join(class_lines))
            else:
                inner_class_code_blocks.append(f"class {ic.name} {{ ... }}")
    # Assemble minimal class
    result = ''
    if class_annos:
        result += '\n'.join(class_annos) + '\n'
    result += class_decl_line + '\n'
    for f in field_lines:
        result += f + '\n'
    for m in method_lines:
        result += m + '\n'
    for icb in inner_class_code_blocks:
        result += icb + '\n'
    result += '}'
    return result

# --- Helper: Extract error block after [INFO] Results: ---
def extract_error_block_after_results(stdout: str) -> str:
    """
    Extracts the error block that appears after '[INFO] Results:' and before the line containing '[ERROR] There are test failures' (inclusive).
    """
    lines = stdout.splitlines()
    in_error_block = False
    error_block = []
    for line in lines:
        if '[INFO] Results:' in line:
            in_error_block = True
            continue  # Skip the '[INFO] Results:' line itself
        if in_error_block:
            error_block.append(line)
            if '[ERROR] There are test failures' in line:
                break  # Stop after including this line
    return '\n'.join(error_block).strip()

# --- Helper: Strip comments from Java code ---
def strip_java_comments_and_trailing_whitespace(code: str) -> str:
    import re
    # Remove all /* ... */ comments (including multiline)
    code = re.sub(r'/\*.*?\*/', '', code, flags=re.DOTALL)
    # Remove all // ... comments (whole line or trailing)
    code = re.sub(r'//.*', '', code)
    # Remove all # ... comments (for Python-style, just in case)
    code = re.sub(r'#.*', '', code)
    # Remove lines that are now empty or only whitespace
    code = '\n'.join([line.rstrip() for line in code.splitlines() if line.strip()])
    return code

# --- JavaParser subprocess bridge integration ---
# Requires: javabridge/JavaParserBridge.class and lib/javaparser-core-3.25.4.jar
import subprocess
JAVAPARSER_JAR = os.path.join('lib', 'javaparser-core-3.25.4.jar')
JAVAPARSER_BRIDGE_CLASS = 'javabridge.JavaParserBridge'

def run_javaparser_bridge(command, java_file_path, method_names=None):
    # Always include '.' in the classpath for correct package resolution
    args = ['java', '-cp', f'{JAVAPARSER_JAR}:javabridge:.', JAVAPARSER_BRIDGE_CLASS, command, java_file_path]
    if method_names:
        args.append(','.join(method_names))
    result = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        print(f"[JavaParserBridge ERROR] {result.stderr}")
        return ""
    return result.stdout.strip()

def extract_public_methods(java_file_path):
    output = run_javaparser_bridge('extract_methods', java_file_path)
    return set(output.split(',')) if output else set()

def extract_class_signatures(java_file_path):
    return run_javaparser_bridge('extract_signatures', java_file_path)

def extract_minimal_class_for_methods(java_file_path, method_names):
    return run_javaparser_bridge('extract_minimal_class', java_file_path, method_names)

def extract_referenced_types(java_file_path, method_names):
    output = run_javaparser_bridge('extract_referenced_types', java_file_path, method_names)
    if not output:
        return set()
    return set([t.strip() for t in output.split(',') if t.strip()])

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
            relevant_java_files_for_context = list(set(relevant_java_files_for_context))
            paths = get_test_paths(str(relative_processed_txt_path), SPRING_BOOT_PROJECT_ROOT)
            test_output_dir = paths["test_output_dir"]
            test_output_file_path = paths["test_output_file_path"]
            if test_output_file_path.exists():
                print(f"\n[SKIP] Test file already exists: '{test_output_file_path}'")
                print(f"Skipping test generation for {target_class_name}")
                continue
            print("\n" + "="*80)
            print(f"GENERATING UNIT TEST FOR: {target_class_name} ({target_package_name})")
            print(f"SOURCE FILE: {java_file_path_abs}")
            print(f"FILES TO RETRIEVE CHUNKS FROM (for context): {relevant_java_files_for_context}")
            print(f"EXTRACTED CUSTOM IMPORTS (for prompt): {custom_imports_list}")
            print(f"DATABASE MOCKING REQUIRED: {requires_db_test}")
            print(f"EXPECTED TEST OUTPUT PATH: '{test_output_file_path}'")
            print("="*80)
            # --- Always use batch mode for test generation ---
            generated_test_code = test_generator.generate_test_case_in_batches(
                target_class_name=target_class_name,
                target_package_name=target_package_name,
                custom_imports=custom_imports_list,
                relevant_java_files_for_context=relevant_java_files_for_context,
                test_output_file_path=test_output_file_path,
                additional_query_instructions="and make sure there are no errors, and you don't cause mismatch in return types and stuff.",
                requires_db_test=requires_db_test,
                dependency_signatures=None,
                target_info=target_info
            )
            print(f"\n[FINAL SUCCESS] Generated test case saved to: '{test_output_file_path}'")
            print("\n--- FINAL GENERATED TEST CASE (Printed to Console for review) ---")
            print(generated_test_code)
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
        traceback.print_exc()
        print("Verify your ChromaDB setup and network connection for Google Generative AI API, and that file paths are correct.")


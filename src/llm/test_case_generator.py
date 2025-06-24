import sys
from pathlib import Path
import os
import json 
from typing import List, Dict, Any, Union
import re

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
LLM_MODEL_NAME_GEMINI = "gemini-2.0-flash" 

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

    def generate_test_plan(self, target_class_name: str, class_code: str, real_methods: list = None) -> List[Dict[str, str]]:
        """
        Step 1: Ask the LLM to generate a test plan (method names + descriptions) for the class.
        Returns a list of dicts: [{"method_name": ..., "description": ...}, ...]
        """
        # If real_methods is provided, use it for the prompt (for stricter retries)
        if real_methods is not None:
            method_list_str = "\n".join(f'- {m}' for m in real_methods)
            plan_prompt = f"""
You are an expert Java test designer. The following class has these public methods:
{method_list_str}

ONLY generate a test plan for these methods, in this exact order, using the exact names. Do NOT invent or omit any methods. If you do, you will be penalized.

Output a JSON array like:
[
  {{"method": "setSessionObject", "description": "..."}},
  {{"method": "updateCacheObj", "description": "..."}},
  ...
]

--- BEGIN CLASS UNDER TEST ---
{class_code}
--- END CLASS UNDER TEST ---
"""
        else:
            plan_prompt = f"""
You are an expert Java test designer. Given the following class, list all public methods and, for each, describe the test cases you would write. For each method, provide:
- The method name (as in the class)
- A one-sentence description of what you would test for this method (including edge cases, exceptions, etc.)

Do NOT invent or omit any methods. Output a JSON array like:
[
  {{"method": "setSessionObject", "description": "..."}},
  ...
]

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
        # Try to parse as JSON
        try:
            test_cases = json.loads(result)
            # Normalize to expected format
            parsed = []
            for entry in test_cases:
                if isinstance(entry, dict) and 'method' in entry:
                    parsed.append({"method_name": entry['method'], "description": entry.get('description', '')})
            return parsed
        except Exception as e:
            print(f"[WARNING] Could not parse LLM output as JSON: {e}")
            # Fallback: try to parse as before (legacy)
            test_cases = []
            for block in result.split("Method:"):
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

    def extract_all_test_methods(self, code: str) -> list:
        """
        Extract all valid @Test methods from LLM output, even if wrapped in a class or with imports.
        Returns a list of cleaned method strings.
        """
        # Remove code blocks, imports, package, class/interface/enum wrappers
        code = re.sub(r"```[\s\S]*?```", "", code)
        code = re.sub(r"^\s*import\s+.*;\s*", "", code, flags=re.MULTILINE)
        code = re.sub(r"^\s*package\s+.*;\s*", "", code, flags=re.MULTILINE)
        code = re.sub(r"^\s*(public\s+)?(class|interface|enum)\s+\w+\s*\{", "", code, flags=re.MULTILINE)
        code = re.sub(r"^\s*}\s*$", "", code, flags=re.MULTILINE)
        # Find all @Test methods
        method_pattern = re.compile(r"@Test\s+void\s+\w+\s*\([^)]*\)\s*\{[\s\S]*?\n\}", re.MULTILINE)
        methods = [m.group(0).strip() for m in method_pattern.finditer(code)]
        return methods

    def generate_test_method(self, target_class_name: str, class_code: str, test_case: Dict[str, str], custom_imports: List[str], test_type: str, error_feedback: str = None) -> list:
        """
        For each test case, generate the test method code using a focused prompt.
        Returns a list of all valid @Test methods found in the LLM output.
        """
        imports_section = '\n'.join(custom_imports) if custom_imports else ''
        method_prompt = f"""
You are an expert Java test writer. Write a JUnit5+Mockito test method for the following scenario.

Class under test:
{class_code}

Test method name: {test_case['method_name']}
Description: {test_case['description']}

STRICT REQUIREMENTS:
- Output ONLY the Java test method code, no class definition, no imports, no explanations, no markdown.
- Use Mockito for mocking dependencies.
- Use JUnit5 assertions.
- Do NOT repeat the class under test.
- Use the following imports if needed (but do NOT output import statements):
{imports_section}
"""
        max_retries = 5
        for attempt in range(max_retries):
            prompt = method_prompt
            if error_feedback and attempt > 0:
                prompt += f"\n--- ERROR FEEDBACK FROM PREVIOUS ATTEMPT ---\n{error_feedback}\n--- END ERROR FEEDBACK ---\n"
            result = self.llm.invoke(prompt)
            if hasattr(result, "content"):
                result = result.content
            code = result.strip()
            print(f"\n--- RAW LLM OUTPUT (Attempt {attempt+1}) ---\n{code}\n--- END RAW LLM OUTPUT ---\n")
            methods = self.extract_all_test_methods(code)
            if methods:
                return methods
            # Prepare explicit feedback for next retry
            error_feedback = (
                "Your previous output was invalid. STRICTLY output ONLY valid Java test method(s) with @Test annotation, "
                "no import statements, no class/interface/enum/package definitions, no markdown, and no explanations. "
                "Do NOT output anything except the method(s)."
            )
        # If all retries fail, return a comment as a single method
        return [f"// [ERROR] LLM failed to generate a valid test method for {test_case['method_name']} after {max_retries} attempts."]

    def _clean_and_validate_method_code(self, code: str) -> Union[str, None]:
        """
        Clean and validate LLM output for a Java test method. If missing @Test but otherwise valid, add it. Returns cleaned code if valid, else None.
        """
        # Remove code blocks (markdown)
        code = re.sub(r"```[\s\S]*?```", "", code)
        # Remove import statements
        code = re.sub(r"^\s*import\s+.*;\s*", "", code, flags=re.MULTILINE)
        # Remove class/interface/enum definitions
        code = re.sub(r"^\s*(public\s+)?(class|interface|enum)\s+\w+\s*\{[\s\S]*?\}", "", code, flags=re.MULTILINE)
        # Remove package statements
        code = re.sub(r"^\s*package\s+.*;\s*", "", code, flags=re.MULTILINE)
        # Remove any remaining @ExtendWith or similar annotations outside methods
        code = re.sub(r"^\s*@\w+\(.*\)\s*", "", code, flags=re.MULTILINE)
        # Remove stray semicolons at the start of lines
        code = re.sub(r"^\s*;\s*$", "", code, flags=re.MULTILINE)
        # Remove empty lines
        code = "\n".join([line for line in code.splitlines() if line.strip()])
        # Accept if it contains a method signature (void ...) and no import/class/interface/enum/package
        if re.search(r"void\s+\w+\s*\(", code):
            # If missing @Test, add it at the top
            if not re.search(r"@Test", code):
                code = "@Test\n" + code
            # Should not contain any import/class/interface/enum/package after cleaning
            if re.search(r"import\s|class\s|interface\s|enum\s|package\s", code):
                return None
            return code.strip()
        # Otherwise, reject
        return None

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
        # --- NEW: Union dependency-based and import-based retrieval ---
        all_imports = target_info.get('all_imports', []) if target_info else []
        project_base_package = "com.iemr"  # Change if your base package is different
        imported_project_files = []
        import_re = re.compile(r'import\s+(' + re.escape(project_base_package) + r'\.[\w\.]+)\.(\w+);')
        for imp in all_imports:
            m = import_re.match(imp)
            if m:
                class_name = m.group(2)
                imported_project_files.append(f"{class_name}.java")
        # Union with dependency-based retrieval
        relevant_java_files_for_context = list(set(relevant_java_files_for_context) | set(imported_project_files))
        if test_type in ("service", "controller", "repository"):
            self._update_retriever_filter(relevant_java_files_for_context, k_override=10)
            retrieved_docs = self.retriever.get_relevant_documents(f"Full code for {target_class_name}")
            class_code = "\n\n".join([doc.page_content for doc in retrieved_docs if target_class_name in doc.page_content])
            if not class_code:
                class_code = "\n\n".join([doc.page_content for doc in retrieved_docs])

            # --- Test Plan Generation with Validation, Retry, and Auto-correction ---
            max_test_plan_retries = 3
            real_methods = sorted(self.extract_public_methods(class_code))
            test_plan = []
            for plan_attempt in range(max_test_plan_retries):
                test_plan = self.generate_test_plan(target_class_name, class_code, real_methods=real_methods)
                if not test_plan:
                    print("[ERROR] LLM did not return a valid test plan. Aborting test generation.")
                    return ""
                plan_methods = self.extract_test_plan_methods(test_plan)
                missing_methods = set(real_methods) - plan_methods
                hallucinated_methods = plan_methods - set(real_methods)
                # Auto-correct: add missing, remove hallucinated
                if missing_methods or hallucinated_methods:
                    print(f"[VALIDATION] Test plan mismatch. Missing methods: {missing_methods}, Hallucinated methods: {hallucinated_methods}")
                    # Remove hallucinated
                    test_plan = [tc for tc in test_plan if tc['method_name'] in real_methods]
                    # Add missing with placeholder
                    for m in missing_methods:
                        test_plan.append({"method_name": m, "description": "No description provided (auto-added)."})
                    # Sort to match real_methods order
                    test_plan = sorted(test_plan, key=lambda x: real_methods.index(x['method_name']))
                if not missing_methods and not hallucinated_methods:
                    break  # Test plan is valid
                print("[VALIDATION] Auto-corrected test plan. Retrying if needed...")
                # If last retry, just use the auto-corrected plan
                if plan_attempt == max_test_plan_retries - 1:
                    print("[VALIDATION] Using auto-corrected test plan after retries.")
            # --- If LLM still fails, fallback to per-method prompting ---
            if not test_plan or len(test_plan) != len(real_methods):
                print("[FALLBACK] LLM failed to generate a valid test plan. Using per-method prompting.")
                test_plan = []
                for m in real_methods:
                    per_method_prompt = f"""
You are an expert Java test designer. For the method '{m}' in the following class, provide a one-sentence description of what you would test (including edge cases, exceptions, etc.).

--- BEGIN CLASS UNDER TEST ---
{class_code}
--- END CLASS UNDER TEST ---
Output only the description, no code or method name.
"""
                    result = self.llm.invoke(per_method_prompt)
                    if hasattr(result, "content"):
                        result = result.content
                    description = result.strip().splitlines()[0] if result.strip() else "No description provided."
                    test_plan.append({"method_name": m, "description": description})
            # --- Continue as before with validated test_plan ---
            test_methods = []
            method_errors = {}
            for test_case in test_plan:
                error_feedback = None
                for attempt in range(MAX_TEST_GENERATION_RETRIES):
                    method_codes = self.generate_test_method(target_class_name, class_code, test_case, custom_imports, test_type, error_feedback=error_feedback)
                    # method_codes is now a list of methods
                    # Assemble a temporary test class with all previous methods + these
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
{chr(10).join(test_methods + method_codes)}
}}
"""
                    # Write to file and try to compile/run
                    os.makedirs(test_output_file_path.parent, exist_ok=True)
                    with open(test_output_file_path, 'w', encoding='utf-8') as f:
                        f.write(temp_test_class_code)
                    test_run_results = self.java_test_runner.run_test(test_output_file_path)
                    if test_run_results["status"] == "SUCCESS":
                        test_methods.extend(method_codes)
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
                            test_methods.extend(method_codes)  # Save the last attempt anyway
            # Step 3: Assemble the final test class
            test_class_code = self.assemble_test_class(target_package_name, custom_imports, target_class_name, test_methods, class_code)
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

    def extract_public_methods(self, class_code: str) -> set:
        """
        Extracts all public method names from the given Java class code using regex.
        """
        # This regex matches public methods (not constructors, not static blocks)
        method_pattern = re.compile(r'public\s+[\w<>\[\]]+\s+(\w+)\s*\(')
        return set(method_pattern.findall(class_code))

    def extract_test_plan_methods(self, test_plan: list) -> set:
        """
        Extracts method names from the LLM-generated test plan.
        """
        return set(tc['method_name'] for tc in test_plan if 'method_name' in tc)

    def assemble_test_class(self, package_name: str, imports: List[str], class_name: str, test_methods: List[str], class_code: str = "") -> str:
        """
        Assemble the final test class file with package, imports, annotations, class definition, and all test methods.
        Auto-detects and adds missing imports for all classes used in the test methods.
        """
        # Auto-detect used class names and add missing imports
        used_class_names = self._extract_used_class_names(test_methods)
        auto_imports = self._map_class_names_to_imports(used_class_names, class_code, extra_known_imports=imports)
        # Compose package statement
        package_stmt = f"package {package_name};\n" if package_name else ""
        # Compose imports (unique, sorted)
        import_lines = sorted(set(imports + auto_imports + [
            "org.junit.jupiter.api.Test",
            "org.junit.jupiter.api.extension.ExtendWith",
            "org.mockito.InjectMocks",
            "org.mockito.Mock",
            "org.mockito.junit.jupiter.MockitoExtension",
            "static org.mockito.Mockito.*",
            "static org.junit.jupiter.api.Assertions.*"
        ]))
        import_section = "\n".join(f"import {imp};" for imp in import_lines if not imp.startswith("static "))
        static_import_section = "\n".join(f"import {imp};" for imp in import_lines if imp.startswith("static "))
        # Compose class annotation and definition
        class_anno = "@ExtendWith(MockitoExtension.class)"
        class_def = f"class {class_name}Test {{"
        # Join all test methods
        methods_section = "\n\n".join(m for m in test_methods if m.strip())
        # Final assembly
        return f"{package_stmt}\n{import_section}\n{static_import_section}\n\n{class_anno}\n{class_def}\n\n{methods_section}\n\n}}"

    def _extract_used_class_names(self, test_methods: List[str]) -> set:
        """
        Extracts all class names used in the test methods (simple heuristic: capitalized words, not primitives).
        """
        java_keywords = {"int", "long", "float", "double", "boolean", "char", "byte", "short", "void", "String", "assert", "return", "if", "else", "for", "while", "switch", "case", "break", "continue", "new", "public", "private", "protected", "static", "final", "class", "interface", "enum", "try", "catch", "finally", "throw", "throws", "extends", "implements", "import", "package", "this", "super", "synchronized", "volatile", "transient", "abstract", "native", "strictfp", "default", "instanceof", "true", "false", "null"}
        class_names = set()
        for method in test_methods:
            # Find all capitalized words (potential class names)
            for match in re.findall(r'\b([A-Z][A-Za-z0-9_]*)\b', method):
                if match not in java_keywords:
                    class_names.add(match)
        return class_names

    def _map_class_names_to_imports(self, class_names: set, class_code: str, extra_known_imports: List[str] = None) -> List[str]:
        """
        Map class names to import statements using a known mapping, the class under test's imports, and any extra known imports.
        """
        # Known mapping for common Java/Mockito/JUnit/Servlet classes
        known_imports = {
            "Optional": "java.util.Optional",
            "List": "java.util.List",
            "ArrayList": "java.util.ArrayList",
            "Map": "java.util.Map",
            "HashMap": "java.util.HashMap",
            "Set": "java.util.Set",
            "HashSet": "java.util.HashSet",
            "Cookie": "jakarta.servlet.http.Cookie",
            "HttpServletRequest": "jakarta.servlet.http.HttpServletRequest",
            "HttpServletResponse": "jakarta.servlet.http.HttpServletResponse",
            "MockedStatic": "org.mockito.MockedStatic",
            "ArgumentCaptor": "org.mockito.ArgumentCaptor",
            "InjectMocks": "org.mockito.InjectMocks",
            "Mock": "org.mockito.Mock",
            "ExtendWith": "org.junit.jupiter.api.extension.ExtendWith",
            "Test": "org.junit.jupiter.api.Test",
            "BeforeEach": "org.junit.jupiter.api.BeforeEach",
            "AfterEach": "org.junit.jupiter.api.AfterEach",
            "DisplayName": "org.junit.jupiter.api.DisplayName",
            "Assertions": "org.junit.jupiter.api.Assertions",
            "Mockito": "org.mockito.Mockito",
            "Service": "org.springframework.stereotype.Service",
            "Autowired": "org.springframework.beans.factory.annotation.Autowired",
            "Captor": "org.mockito.Captor",
            "Disabled": "org.junit.jupiter.api.Disabled",
            "ParameterizedTest": "org.junit.jupiter.params.ParameterizedTest",
            "ValueSource": "org.junit.jupiter.params.provider.ValueSource",
            "CsvSource": "org.junit.jupiter.params.provider.CsvSource",
            "SpringBootTest": "org.springframework.boot.test.context.SpringBootTest",
            "Transactional": "org.springframework.transaction.annotation.Transactional",
            "EntityManager": "jakarta.persistence.EntityManager",
            "Base64": "java.util.Base64",
            "UUID": "java.util.UUID",
            "Stream": "java.util.stream.Stream",
            "Collectors": "java.util.stream.Collectors",
        }
        # Add extra known imports if provided
        if extra_known_imports:
            for imp in extra_known_imports:
                # Try to extract class name from import
                m = re.match(r'(?:static\s+)?([\w\.]+)\.([A-Z][A-Za-z0-9_]*)', imp)
                if m:
                    known_imports[m.group(2)] = m.group(1) + '.' + m.group(2)
        # Extract imports from class_code
        for imp in re.findall(r'import\s+([\w\.]+);', class_code):
            class_name = imp.split('.')[-1]
            known_imports[class_name] = imp
        # Map class names to imports
        imports = []
        for name in class_names:
            if name in known_imports:
                imports.append(known_imports[name])
        return imports

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


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
from chroma_db.chroma_client import get_chroma_client as get_chroma_client_examples, get_or_create_collection as get_or_create_collection_examples


# --- Google API Configuration ---
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") 
if not GOOGLE_API_KEY:
    print("WARNING: GOOGLE_API_KEY environment variable not set. Please set it for Gemini API calls.")
    print("Example: export GOOGLE_API_KEY='your_google_api_key_here'")

# LLM model definition - Using Gemini 1.5 Flash for potentially better rate limits and speed
LLM_MODEL_NAME_GEMINI = "gemini-2.5-flash" 

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
        # --- PATCH: Strip code block markers before JSON parsing ---
        result = result.strip()
        # Remove triple backticks and optional language label (e.g., ```json or ```java)
        result = re.sub(r'^```[a-zA-Z]*\n', '', result)
        result = re.sub(r'^```', '', result)
        result = re.sub(r'```$', '', result)
        result = result.strip()
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
        # First, try to extract the entire class content if it's wrapped in a class
        class_match = re.search(r'class\s+\w+Test\s*\{([\s\S]*)\}', code)
        if class_match:
            # Extract content inside the class
            class_content = class_match.group(1)
        else:
            class_content = code
        
        # Remove code blocks (markdown)
        class_content = re.sub(r'```[\s\S]*?```', '', class_content)
        
        # Remove package and import statements
        class_content = re.sub(r'^\s*package\s+.*;\s*', '', class_content, flags=re.MULTILINE)
        class_content = re.sub(r'^\s*import\s+.*;\s*', '', class_content, flags=re.MULTILINE)
        
        # Find all @Test methods with more flexible patterns
        test_methods = []
        
        # Pattern 1: @Test void methodName() { ... }
        pattern1 = re.compile(r'@Test\s+void\s+(\w+)\s*\([^)]*\)\s*\{[\s\S]*?\n\s*\}', re.MULTILINE)
        for match in pattern1.finditer(class_content):
            test_methods.append(match.group(0).strip())
        
        # Pattern 2: @Test public void methodName() { ... }
        pattern2 = re.compile(r'@Test\s+public\s+void\s+(\w+)\s*\([^)]*\)\s*\{[\s\S]*?\n\s*\}', re.MULTILINE)
        for match in pattern2.finditer(class_content):
            test_methods.append(match.group(0).strip())
        
        # Pattern 3: @Test private void methodName() { ... }
        pattern3 = re.compile(r'@Test\s+private\s+void\s+(\w+)\s*\([^)]*\)\s*\{[\s\S]*?\n\s*\}', re.MULTILINE)
        for match in pattern3.finditer(class_content):
            test_methods.append(match.group(0).strip())
        
        # Pattern 4: More flexible - any method with @Test annotation
        pattern4 = re.compile(r'@Test[^{]*\{[\s\S]*?\n\s*\}', re.MULTILINE)
        for match in pattern4.finditer(class_content):
            method_content = match.group(0).strip()
            # Only add if it's not already captured by other patterns
            if not any(method_content in existing for existing in test_methods):
                test_methods.append(method_content)
        
        # If no methods found with @Test, look for any void methods that might be tests
        if not test_methods:
            void_method_pattern = re.compile(r'void\s+(\w+)\s*\([^)]*\)\s*\{[\s\S]*?\n\s*\}', re.MULTILINE)
            for match in void_method_pattern.finditer(class_content):
                method_content = match.group(0).strip()
                # Add @Test annotation if missing
                if not method_content.startswith('@Test'):
                    method_content = '@Test\n' + method_content
                test_methods.append(method_content)
        
        return test_methods

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

    def generate_whole_class_test_with_context(self, target_class_name, target_package_name, class_code, public_methods, custom_imports, imports_context, formatted_examples=None):
        method_list_str = '\n'.join(f'- {m}' for m in public_methods)
        imports_section = '\n'.join(custom_imports) if custom_imports else ''
        strictness = (
            "IMPORTANT: Only use methods, fields, and constructors that are present in the code blocks below. "
            "Do NOT invent or assume any methods, fields, or classes. "
            "If a method or field is not present in the provided code, do NOT use it in your test. "
            "If you are unsure, leave it out. "
            "If you hallucinate any code, you will be penalized and re-prompted. "
            "You must generate a compiling, passing, and style-compliant test class."
        )
        prompt = ''
        if formatted_examples:
            prompt += f"// The following are real test examples from your codebase.\n{formatted_examples}\n\n"
        prompt += f"""
You are an expert Java developer. You are to generate a complete JUnit 5 + Mockito test class for the MAIN CLASS below. The other classes are provided as context only (do NOT generate tests for them).

{imports_context}

--- BEGIN MAIN CLASS UNDER TEST ---\n{class_code}\n--- END MAIN CLASS UNDER TEST ---

Instructions:
- Only generate tests for the MAIN CLASS.
- Include all necessary imports and annotations.
- Name the test class {target_class_name}Test and use the package {target_package_name}.
- Cover ALL of these public methods (do not skip any):\n{method_list_str}
- For each method, create at least one @Test method that tests its functionality.
- Avoid unnecessary stubbing or mocking. Only mock what is required for compilation or to isolate the class under test. Do NOT mock dependencies that are not used in the test method. Do NOT mock simple POJOs or value objects.
- Do NOT output explanations, markdown, or comments outside the code.
- Output ONLY the Java code for the test class, nothing else.
- Make sure to include @Test annotations on all test methods.
- Use these imports if needed (but do NOT output import statements if not needed):\n{imports_section}

{strictness}
"""
        print("\n--- WHOLE-CLASS TEST GENERATION PROMPT (FULL) ---\n" + prompt + "\n--- END WHOLE-CLASS TEST GENERATION PROMPT ---\n")
        result = self.llm.invoke(prompt)
        if hasattr(result, "content"):
            result = result.content
        print("\n--- RAW LLM WHOLE-CLASS OUTPUT (FULL) ---\n" + result + "\n--- END RAW LLM WHOLE-CLASS OUTPUT ---\n")
        return result.strip()

    def generate_whole_class_test_with_feedback(self, target_class_name, target_package_name, class_code, public_methods, custom_imports, imports_context, feedback_msg, formatted_examples=None):
        method_list_str = '\n'.join(f'- {m}' for m in public_methods)
        imports_section = '\n'.join(custom_imports) if custom_imports else ''
        strictness = (
            "IMPORTANT: Only use methods, fields, and constructors that are present in the code blocks below. "
            "Do NOT invent or assume any methods, fields, or classes. "
            "If a method or field is not present in the provided code, do NOT use it in your test. "
            "If you are unsure, leave it out. "
            "If you hallucinate any code, you will be penalized and re-prompted. "
            "You must generate a compiling, passing, and style-compliant test class."
        )
        prompt = ''
        if formatted_examples:
            prompt += f"// The following are real test examples from your codebase.\n{formatted_examples}\n\n"
        prompt += f"""
{feedback_msg}

You are an expert Java developer. You are to generate a complete JUnit 5 + Mockito test class for the MAIN CLASS below. The other classes are provided as context only (do NOT generate tests for them).

{imports_context}

--- BEGIN MAIN CLASS UNDER TEST ---\n{class_code}\n--- END MAIN CLASS UNDER TEST ---

Instructions:
- Only generate tests for the MAIN CLASS.
- Include all necessary imports and annotations.
- Name the test class {target_class_name}Test and use the package {target_package_name}.
- Cover ALL of these public methods (do not skip any):\n{method_list_str}
- For each method, create at least one @Test method that tests its functionality.
- Avoid unnecessary stubbing or mocking. Only mock what is required for compilation or to isolate the class under test. Do NOT mock dependencies that are not used in the test method. Do NOT mock simple POJOs or value objects.
- Do NOT output explanations, markdown, or comments outside the code.
- Output ONLY the Java code for the test class, nothing else.
- Make sure to include @Test annotations on all test methods.
- Use these imports if needed (but do NOT output import statements if not needed):\n{imports_section}

{strictness}
"""
        print("\n--- WHOLE-CLASS TEST GENERATION PROMPT (WITH FEEDBACK, FULL) ---\n" + prompt + "\n--- END WHOLE-CLASS TEST GENERATION PROMPT ---\n")
        result = self.llm.invoke(prompt)
        if hasattr(result, "content"):
            result = result.content
        print("\n--- RAW LLM WHOLE-CLASS OUTPUT (WITH FEEDBACK, FULL) ---\n" + result + "\n--- END RAW LLM WHOLE-CLASS OUTPUT ---\n")
        return result.strip()

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
        self._update_retriever_filter(relevant_java_files_for_context, k_override=10)
        retrieved_docs = self.retriever.get_relevant_documents(f"Full code for {target_class_name}")
        class_code = "\n\n".join([doc.page_content for doc in retrieved_docs if target_class_name in doc.page_content])
        if not class_code:
            class_code = "\n\n".join([doc.page_content for doc in retrieved_docs])
        public_methods = sorted(self.extract_public_methods(class_code))
        
        # --- Set retry limits ---
        MAX_RETRIES = 15
        max_feedback_retries = MAX_RETRIES
        max_retries = MAX_RETRIES

        print(f"[DEBUG] Starting test generation for: {target_class_name}")
        print(f"[DEBUG] Output test file will be: {test_output_file_path}")
        print(f"[DEBUG] Custom imports: {custom_imports}")
        print(f"[DEBUG] Additional instructions: {additional_query_instructions}")
        print(f"[DEBUG] Target info: {target_info}")

        # --- Retrieve the full class code for the main class under test ---
        print(f"[DEBUG] Fetching main class code for: {target_class_name}")
        class_code = None
        main_java_filename = None
        if target_info and 'java_file_path_abs' in target_info:
            main_java_filename = Path(target_info['java_file_path_abs']).name
        if main_java_filename:
            print(f"[DEBUG] Using ChromaDB retriever for file: {main_java_filename}")
            self._update_retriever_filter([main_java_filename], k_override=20)
            retrieved_docs = self.retriever.get_relevant_documents(f"Full code for {target_class_name}")
            print(f"[DEBUG] Retrieved {len(retrieved_docs)} docs from ChromaDB for main class.")
            for i, doc in enumerate(retrieved_docs):
                print(f"[DEBUG] Main class chunk {i+1}:\n{doc.page_content[:500]}{'...' if len(doc.page_content) > 500 else ''}\n---")
            class_code = "\n\n".join([doc.page_content for doc in retrieved_docs if target_class_name in doc.page_content])
            if not class_code:
                class_code = "\n\n".join([doc.page_content for doc in retrieved_docs])
            print(f"[DEBUG] Loaded class code for {target_class_name} from ChromaDB chunks.")
        else:
            class_code = ""
        print("\n[DEBUG] CLASS CODE SENT TO LLM (FULL):\n" + class_code + "\n[END DEBUG CLASS CODE]\n")

        # --- Extract project-local imports and load their code as context from ChromaDB ---
        print(f"[DEBUG] Extracting and fetching imported class context for: {target_class_name}")
        import_pattern = re.compile(r'^import\s+(com\\.iemr\\.[\\w\\.]+)\\.([A-Z][A-Za-z0-9_]*)\;', re.MULTILINE)
        imported_classes = []
        for match in import_pattern.finditer(class_code):
            class_name = match.group(2)
            dep_java_filename = f"{class_name}.java"
            print(f"[DEBUG] Fetching import: {dep_java_filename}")
            self._update_retriever_filter([dep_java_filename], k_override=10)
            dep_docs = self.retriever.get_relevant_documents(f"Full code for {class_name}")
            print(f"[DEBUG] Retrieved {len(dep_docs)} docs from ChromaDB for import {class_name}.")
            for i, doc in enumerate(dep_docs):
                print(f"[DEBUG] Import chunk {i+1} for {class_name}:\n{doc.page_content[:500]}{'...' if len(doc.page_content) > 500 else ''}\n---")
            dep_code = "\n\n".join([doc.page_content for doc in dep_docs if class_name in doc.page_content])
            if not dep_code:
                dep_code = "\n\n".join([doc.page_content for doc in dep_docs])
            if dep_code:
                imported_classes.append((class_name, dep_code))
                print(f"[DEBUG] Added imported class context for {class_name} from ChromaDB.")
            else:
                print(f"[WARNING] Imported class code not found in ChromaDB: {dep_java_filename}")
        imports_context = ""
        for class_name, dep_code in imported_classes:
            print(f"[DEBUG] IMPORTED CLASS CONTEXT for {class_name}:\n{dep_code}\n--- END IMPORTED CLASS ---")
            imports_context += f"\n--- BEGIN IMPORTED CLASS: {class_name} ---\n{dep_code}\n--- END IMPORTED CLASS: {class_name} ---\n"

        # --- Ensure imported_class_members is defined before feedback loop ---
        def extract_class_members(java_code):
            method_pattern = re.compile(r'(public|protected|private)?\s+[\w<>,\[\]]+\s+(\w+)\s*\([^)]*\)')
            field_pattern = re.compile(r'(private|protected|public)\s+[\w<>,\[\]]+\s+(\w+)\s*[;=]')
            members = set()
            fields = []
            for f in field_pattern.finditer(java_code):
                field_name = f.group(2)
                fields.append(field_name)
                members.add(field_name)
            for m in method_pattern.finditer(java_code):
                members.add(m.group(2))
            if ('@Data' in java_code) or ('@Getter' in java_code) or ('@Setter' in java_code):
                for field in fields:
                    if not field:
                        continue
                    cap = field[0].upper() + field[1:] if len(field) > 1 else field.upper()
                    members.add(f'get{cap}')
                    members.add(f'set{cap}')
            return members
        imported_class_members = {}
        for class_name, dep_code in imported_classes:
            imported_class_members[class_name] = extract_class_members(dep_code)

        # --- RAG: Retrieve real test examples ---
        print(f"[RAG] Retrieving real test examples for: {target_class_name}")
        method_sigs = list(self.extract_public_methods(class_code))
        test_examples = self.retrieve_similar_test_examples(class_code, method_sigs, top_n=3)
        formatted_examples = self.format_test_examples_for_prompt(test_examples) if test_examples else ''
        if formatted_examples:
            print(f"[RAG] Retrieved {len(test_examples)} real test examples. Including in prompt.")
        else:
            print(f"[RAG] No real test examples found for this class.")

        # --- Automated feedback loop for hallucinated members ---
        feedback_attempt = 0
        hallucinated = set()
        while feedback_attempt <= max_feedback_retries:
            print(f"[DEBUG] Hallucination feedback loop attempt {feedback_attempt+1}")
            if feedback_attempt == 0:
                test_class_code = self.generate_whole_class_test_with_context(target_class_name, target_package_name, class_code, public_methods, custom_imports, imports_context, formatted_examples)
            else:
                test_class_code = self.generate_whole_class_test_with_feedback(target_class_name, target_package_name, class_code, public_methods, custom_imports, imports_context, feedback_msg, formatted_examples)
            print(f"[DEBUG] LLM OUTPUT (hallucination check):\n{test_class_code}\n--- END LLM OUTPUT ---")
            hallucinated = self.find_hallucinated_members(test_class_code, imported_class_members)
            print(f"[DEBUG] Hallucinated members found: {hallucinated}")
            if not hallucinated:
                break
            print(f"[FEEDBACK LOOP] Hallucinated members found: {hallucinated}. Re-prompting LLM with feedback.")
            feedback_msg = ("WARNING: In your previous output, you used the following members which do NOT exist in the provided code: "
                            f"{', '.join(hallucinated)}. Do NOT use these or any other non-existent members. Only use what is present in the code blocks below. If you hallucinate again, you will be penalized.")
            feedback_attempt += 1
        test_class_code = test_class_code.strip()
        test_class_code = re.sub(r'^```[a-zA-Z]*\n', '', test_class_code)
        test_class_code = re.sub(r'^```', '', test_class_code)
        test_class_code = re.sub(r'```$', '', test_class_code)
        test_class_code = test_class_code.strip()
        all_test_class_outputs = [test_class_code]

        if not public_methods:
            print(f"[WARNING] No public methods found in {target_class_name}. Skipping test generation.")
            return ""

        print(f"[DEBUG] Validating test plan for {target_class_name}")
        test_plan = self.generate_test_plan(target_class_name, class_code)
        print(f"[DEBUG] LLM-generated test plan: {test_plan}")
        plan_methods = [tc['method_name'] for tc in test_plan]
        real_methods = sorted(public_methods)
        print(f"[DEBUG] Methods from static analysis: {real_methods}")
        retries = 0
        while set(plan_methods) != set(real_methods) and retries < MAX_RETRIES:
            print(f"[TEST PLAN VALIDATION] Mismatch between LLM plan and real methods. Retrying with strict feedback.")
            print(f"  LLM plan methods: {plan_methods}")
            print(f"  Static analysis methods: {real_methods}")
            test_plan = self.generate_test_plan(target_class_name, class_code, real_methods)
            print(f"[DEBUG] LLM-generated test plan (retry {retries+1}): {test_plan}")
            plan_methods = [tc['method_name'] for tc in test_plan]
            retries += 1
        if set(plan_methods) != set(real_methods):
            print(f"[TEST PLAN VALIDATION] LLM failed to match real method list after retries. Using static analysis list.")
            print(f"  LLM plan methods: {plan_methods}")
            print(f"  Static analysis methods: {real_methods}")
            plan_methods = real_methods
            test_plan = [{"method_name": m, "description": f"Test for {m}"} for m in real_methods]
        max_methods_per_group = 5
        method_groups = [plan_methods[i:i+max_methods_per_group] for i in range(0, len(plan_methods), max_methods_per_group)]
        for group in method_groups:
            print(f"[DEBUG] Generating test group for methods: {group}")
            group_methods = [tc for tc in test_plan if tc['method_name'] in group]
            anti_hallucination = "Do NOT invent methods, do NOT mock POJOs/value objects, only mock what is required for compilation or isolation. BAD: Mocks unused dependencies. GOOD: Only mocks required dependencies. If unsure, prefer not to mock. INCLUDE ALL PUBLIC METHODS, INCLUDING STATIC METHODS."
            positive_example = "// GOOD: Only mocks required dependencies, covers all public and static methods."
            negative_example = "// BAD: Mocks unused dependencies, skips methods, causes build to fail."
            prompt_instructions = f"{additional_query_instructions}\n{anti_hallucination}\n{positive_example}\n{negative_example}"
            group_test_code = self.generate_whole_class_test_with_context(target_class_name, target_package_name, class_code, group, custom_imports, imports_context, formatted_examples)
            print(f"[DEBUG] RAW LLM OUTPUT for group {group}:\n{group_test_code}\n--- END RAW LLM OUTPUT ---")
            group_test_code = group_test_code.strip()
            group_test_code = re.sub(r'^```[a-zA-Z]*\n', '', group_test_code)
            group_test_code = re.sub(r'^```', '', group_test_code)
            group_test_code = re.sub(r'```$', '', group_test_code)
            group_test_code = group_test_code.strip()
            error_feedback = None
            last_valid_code = None
            for attempt in range(MAX_RETRIES):
                print(f"[DEBUG] Test generation retry {attempt+1} for group: {group}")
                if error_feedback and attempt > 0:
                    print(f"[LLM ERROR FEEDBACK] Feeding the following error message to LLM (attempt {attempt+1}):\n{error_feedback}\n--- END ERROR FEEDBACK ---")
                    group_test_code = self.generate_whole_class_test_with_feedback(target_class_name, target_package_name, class_code, group, custom_imports, imports_context, error_feedback, formatted_examples)
                    print(f"[DEBUG] RAW LLM OUTPUT (after feedback) for group {group}:\n{group_test_code}\n--- END RAW LLM OUTPUT ---")
                    group_test_code = group_test_code.strip()
                    group_test_code = re.sub(r'^```[a-zA-Z]*\n', '', group_test_code)
                    group_test_code = re.sub(r'^```', '', group_test_code)
                    group_test_code = re.sub(r'```$', '', group_test_code)
                    group_test_code = group_test_code.strip()
                abs_path = os.path.abspath(test_output_file_path)
                print(f"[DEBUG] About to write GROUP test to: {abs_path}")
                try:
                    with open(test_output_file_path, 'w', encoding='utf-8') as f:
                        f.write(group_test_code)
                    print(f"[DEBUG] Successfully wrote GROUP test to {abs_path} (length: {len(group_test_code)})")
                except Exception as e:
                    print(f"[ERROR] Failed to write GROUP test to {abs_path}: {e}")
                test_run_results = self.java_test_runner.run_test(test_output_file_path)
                print(f"[DEBUG] Test run results: {test_run_results}")
                compilation_errors = test_run_results['detailed_errors'].get('compilation_errors', [])
                test_failures = test_run_results['detailed_errors'].get('test_failures', [])
                static_analysis = self.run_static_analysis(test_output_file_path)
                static_errors = static_analysis['errors']
                print(f"[DEBUG] Compilation errors: {compilation_errors}")
                print(f"[DEBUG] Test failures: {test_failures}")
                print(f"[DEBUG] Static analysis errors: {static_errors}")
                if not compilation_errors and not test_failures and not static_errors:
                    print(f"[SUCCESS] No compilation, test, or static analysis errors after {attempt+1} attempt(s).")
                    last_valid_code = group_test_code
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
                    if static_errors:
                        print(f"[STATIC ANALYSIS ERROR] Detected after attempt {attempt+1}:")
                        for err in static_errors:
                            print(f"  - {err}")
                            error_msgs.append(f"STATIC: {err}")
                    error_feedback = '\n'.join(error_msgs)
            if last_valid_code:
                all_test_class_outputs.append(last_valid_code)
            else:
                print(f"[ERROR] Test generation failed for methods {group} after {MAX_RETRIES} attempts. See logs for details.")
        all_methods = []
        covered_methods = set()
        for code in all_test_class_outputs:
            extracted_methods = self.extract_all_test_methods(code)
            print(f"[DEBUG] Extracted test methods: {extracted_methods}")
            if extracted_methods:
                all_methods.extend(extracted_methods)
                for m in extracted_methods:
                    match = re.search(r'void\s+(test_?([A-Za-z0-9_]+))\s*\(', m)
                    if match:
                        covered_methods.add(match.group(2).lower())
            else:
                if 'class' in code and 'Test' in code and ('@Test' in code or 'void' in code):
                    print(f"[FALLBACK] Using LLM output directly as test class for {target_class_name}")
                    cleaned_code = code.strip()
                    cleaned_code = re.sub(r'^```[a-zA-Z]*\n', '', cleaned_code)
                    cleaned_code = re.sub(r'^```', '', cleaned_code)
                    cleaned_code = re.sub(r'```$', '', cleaned_code)
                    cleaned_code = cleaned_code.strip()
                    print(f"[DEBUG] FINAL TEST CLASS CODE (FALLBACK):\n{cleaned_code}\n--- END FINAL TEST CLASS CODE ---")
                    return cleaned_code
        for real_method in real_methods:
            if not any(real_method.lower() in m.lower() for m in covered_methods):
                print(f"[LLM COVERAGE] Prompting LLM to generate test for uncovered method: {real_method}")
                single_method_prompt = f"""
You are an expert Java test writer. Write a JUnit 5 + Mockito test method for the following public method in the class {target_class_name}:

Method signature:
{real_method}

Class code:
{class_code}

STRICT REQUIREMENTS:
- Output ONLY the Java test method code, no class definition, no imports, no explanations, no markdown.
- Use Mockito for mocking dependencies.
- Use JUnit5 assertions.
- Do NOT repeat the class under test.
- The test method name should clearly reference the method being tested.
- Use the following imports if needed (but do NOT output import statements):
{chr(10).join(custom_imports)}
- Do NOT invent or use any methods, fields, or classes not present in the provided code. If you hallucinate, you will be penalized.
"""
                for attempt in range(MAX_RETRIES):
                    print(f"[LLM COVERAGE] Retry {attempt+1} for method: {real_method}")
                    print(f"[LLM COVERAGE] Prompt sent to LLM:\n{single_method_prompt}\n--- END PROMPT ---")
                    result = self.llm.invoke(single_method_prompt)
                    if hasattr(result, "content"):
                        result = result.content
                    print(f"[LLM COVERAGE] RAW LLM OUTPUT for method {real_method}:\n{result}\n--- END RAW LLM OUTPUT ---")
                    code = result.strip()
                    methods = self.extract_all_test_methods(code)
                    print(f"[LLM COVERAGE] Extracted methods: {methods}")
                    if methods:
                        all_methods.extend(methods)
                        covered_methods.add(real_method.lower())
                        break
        if not all_methods:
            print(f"[ERROR] No test methods extracted for {target_class_name}. Skipping test class.")
            return ""
        final_test_class_code = self.assemble_test_class(target_package_name, custom_imports, target_class_name, all_methods, class_code)
        final_test_class_code = final_test_class_code.strip()
        final_test_class_code = re.sub(r'^```[a-zA-Z]*\n', '', final_test_class_code)
        final_test_class_code = re.sub(r'^```', '', final_test_class_code)
        final_test_class_code = re.sub(r'```$', '', final_test_class_code)
        final_test_class_code = final_test_class_code.strip()
        print(f"[DEBUG] FINAL GENERATED TEST CLASS CODE (before write):\n{final_test_class_code}\n--- END FINAL TEST CLASS CODE ---")
        try:
            with open(test_output_file_path, 'w', encoding='utf-8') as f:
                f.write(final_test_class_code)
            print(f"[FINAL SUCCESS] Generated test case saved to: '{test_output_file_path}'")
        except Exception as e:
            print(f"[FINAL ERROR] Could not save test case to '{test_output_file_path}': {e}")
        print("\n--- FINAL GENERATED TEST CASE (Printed to Console for review) ---")
        print(final_test_class_code)
        print("\n" + "="*80 + "\n")
        return final_test_class_code

    def extract_public_methods(self, class_code: str) -> set:
        """
        Extracts all public method names from the given Java class code using regex.
        Includes constructors and static methods.
        """
        methods = set()
        
        # Match public constructors (same name as class)
        constructor_pattern = re.compile(r'public\s+(\w+)\s*\([^)]*\)\s*\{')
        for match in constructor_pattern.finditer(class_code):
            methods.add(match.group(1))
        
        # Match public methods (including static, final, etc.)
        method_pattern = re.compile(r'public\s+(?:static\s+)?(?:final\s+)?(?:abstract\s+)?(?:synchronized\s+)?(?:native\s+)?(?:strictfp\s+)?(?:<[^>]+>\s+)?[\w<>\[\]]+\s+(\w+)\s*\(')
        for match in method_pattern.finditer(class_code):
            methods.add(match.group(1))
        
        # Match public static methods that might have different patterns
        static_method_pattern = re.compile(r'public\s+static\s+(?:final\s+)?(?:<[^>]+>\s+)?[\w<>\[\]]+\s+(\w+)\s*\(')
        for match in static_method_pattern.finditer(class_code):
            methods.add(match.group(1))
        
        return methods

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
        
        # Clean and prepare imports
        all_imports = []
        
        # Process custom imports (remove 'import ' prefix if present)
        for imp in imports:
            if imp.startswith('import '):
                all_imports.append(imp)
            else:
                all_imports.append(f"import {imp};")
        
        # Process auto-detected imports
        for imp in auto_imports:
            if imp.startswith('import '):
                all_imports.append(imp)
            else:
                all_imports.append(f"import {imp};")
        
        # Add standard test imports
        standard_imports = [
            "import org.junit.jupiter.api.Test;",
            "import org.junit.jupiter.api.extension.ExtendWith;",
            "import org.mockito.InjectMocks;",
            "import org.mockito.Mock;",
            "import org.mockito.junit.jupiter.MockitoExtension;",
            "import static org.mockito.Mockito.*;",
            "import static org.junit.jupiter.api.Assertions.*;"
        ]
        all_imports.extend(standard_imports)
        
        # Remove duplicates and sort
        unique_imports = sorted(set(all_imports))
        
        # Separate static and non-static imports
        static_imports = [imp for imp in unique_imports if "static" in imp]
        regular_imports = [imp for imp in unique_imports if "static" not in imp]
        
        # Compose import sections
        import_section = "\n".join(regular_imports)
        static_import_section = "\n".join(static_imports)
        
        # Compose class annotation and definition
        class_anno = "@ExtendWith(MockitoExtension.class)"
        class_def = f"class {class_name}Test {{"
        
        # Join all test methods
        methods_section = "\n\n".join(m for m in test_methods if m.strip())
        
        # Final assembly
        result = f"{package_stmt}\n{import_section}\n"
        if static_import_section:
            result += f"{static_import_section}\n"
        result += f"\n{class_anno}\n{class_def}\n\n{methods_section}\n\n}}"
        
        return result

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

    def retrieve_similar_test_examples(self, class_code: str, method_signatures: list, top_n: int = 3):
        """
        Retrieve top-N similar real test examples from ChromaDB for the given class code and method signatures.
        Returns a list of dicts with 'page_content' and 'metadata'.
        """
        query = class_code + '\n' + '\n'.join(method_signatures)
        query_embedding = self.embeddings.embed_documents([query])[0]
        results = self.test_examples_vectorstore.similarity_search_by_vector(
            query_embedding, k=top_n, filter={'type': 'test_example'}
        )
        return results

    def format_test_examples_for_prompt(self, examples: list) -> str:
        """
        Format retrieved test/source examples for inclusion in the LLM prompt.
        """
        formatted = []
        for ex in examples:
            # Split page_content into source and test
            content = getattr(ex, 'page_content', '')
            parts = re.split(r'---?\s*BEGIN TEST\s*---?', content)
            if len(parts) == 2:
                source_code = parts[0].replace('SOURCE:', '').strip()
                test_code = parts[1].replace('TEST:', '').strip()
            else:
                # Fallback: try to split by 'TEST:'
                if 'TEST:' in content:
                    source_code, test_code = content.split('TEST:', 1)
                    source_code = source_code.replace('SOURCE:', '').strip()
                    test_code = test_code.strip()
                else:
                    source_code = content.strip()
                    test_code = ''
            formatted.append(f"--- BEGIN EXAMPLE SOURCE ---\n{source_code}\n--- END EXAMPLE SOURCE ---\n\n--- BEGIN EXAMPLE TEST ---\n{test_code}\n--- END EXAMPLE TEST ---\n")
        return '\n\n'.join(formatted)

    def find_hallucinated_members(self, test_code, imported_class_members):
        hallucinated = set()
        for class_name, members in imported_class_members.items():
            usage_pattern = re.compile(rf'{class_name}\.([A-Za-z0-9_]+)')
            for match in usage_pattern.finditer(test_code):
                member = match.group(1)
                if member not in members:
                    hallucinated.add(f'{class_name}.{member}')
            new_pattern = re.compile(rf'new\s+{class_name}\(\)\.([A-Za-z0-9_]+)')
            for match in new_pattern.finditer(test_code):
                member = match.group(1)
                if member not in members:
                    hallucinated.add(f'{class_name}.{member}')
        return hallucinated

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

            # --- NEW: Check if test file already exists and skip if it does ---
            if test_output_file_path.exists():
                print(f"\n[SKIP] Test file already exists: '{test_output_file_path}'")
                print(f"Skipping test generation for {target_class_name}")
                continue

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


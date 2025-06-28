# NO-OP: Trigger reload for extract_public_methods visibility
import sys
from pathlib import Path
import os
import json 
import tempfile
from typing import List, Dict, Any, Union
import re
import time

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
        
        try:
            print(f"Loading embedding model: {EMBEDDING_MODEL_NAME_BGE} on {DEVICE_FOR_EMBEDDINGS}...")
            self.embeddings = HuggingFaceEmbeddings(
                model_name=EMBEDDING_MODEL_NAME_BGE,
                encode_kwargs={'normalize_embeddings': True}
            )
            print("Embedding model loaded.")
        except Exception as e:
            print(f"[ERROR] Failed to load embedding model: {e}")
            raise
          
        try:
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
        except Exception as e:
            print(f"[ERROR] Failed to connect to ChromaDB: {e}")
            raise

        try:
            self.llm = self._instantiate_llm()
            print("Google Gemini LLM instantiated for LangChain.")
        except Exception as e:
            print(f"[ERROR] Failed to initialize LLM: {e}")
            raise

        # QA chain will be initialized/updated dynamically in generate_test_case
        self.qa_chain = None 

        try:
            project_root_from_env = os.getenv("SPRING_BOOT_PROJECT_PATH")
            if not project_root_from_env:
                raise ValueError("SPRING_BOOT_PROJECT_PATH environment variable is not set. Cannot initialize JavaTestRunner.")
            self.java_test_runner = JavaTestRunner(project_root=Path(project_root_from_env), build_tool=build_tool)
            self.last_test_run_results = None # Initialize to store feedback
        except Exception as e:
            print(f"[ERROR] Failed to initialize JavaTestRunner: {e}")
            raise

        # --- RAG: Setup for test example retrieval ---
        try:
            self.test_examples_collection_name = "test_examples_collection"
            self.test_examples_chroma_client = get_chroma_client_examples()
            self.test_examples_vectorstore = Chroma(
                client=self.test_examples_chroma_client,
                collection_name=self.test_examples_collection_name,
                embedding_function=self.embeddings
            )
        except Exception as e:
            print(f"[WARNING] Failed to setup test examples collection: {e}")
            self.test_examples_vectorstore = None

    def _instantiate_llm(self) -> ChatGoogleGenerativeAI:
        if not GOOGLE_API_KEY:
            raise ValueError("GOOGLE_API_KEY environment variable is not set. Cannot initialize Gemini LLM.")
        
        print(f"Using Google Gemini LLM: {LLM_MODEL_NAME_GEMINI}...")
        # Use low temperature (0.1) for consistent, deterministic test generation
        # Higher temperatures (0.6+) lead to more creative but less reliable output
        return ChatGoogleGenerativeAI(
            model=LLM_MODEL_NAME_GEMINI, 
            temperature=0.2,
            request_timeout=120  # 2 minute timeout to prevent hanging
        )

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

    def extract_test_class_components(self, code: str) -> tuple:
        """
        Extract both field declarations (@Mock, @InjectMocks) and test methods from LLM output.
        Returns a tuple of (field_declarations, test_methods).
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
        
        # Extract field declarations (@Mock, @InjectMocks)
        field_declarations = []
        
        # Pattern for @Mock field declarations
        mock_pattern = re.compile(r'@Mock\s+[^;]+;', re.MULTILINE)
        for match in mock_pattern.finditer(class_content):
            field_declarations.append(match.group(0).strip())
        
        # Pattern for @InjectMocks field declarations
        inject_mocks_pattern = re.compile(r'@InjectMocks\s+[^;]+;', re.MULTILINE)
        for match in inject_mocks_pattern.finditer(class_content):
            field_declarations.append(match.group(0).strip())
        
        # Extract test methods (same logic as extract_all_test_methods)
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
        
        return field_declarations, test_methods

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
        print(f"[DEBUG] generate_whole_class_test_with_context called for {target_class_name}")
        method_list_str = '\n'.join(f'- {m}' for m in public_methods)
        imports_section = '\n'.join(custom_imports) if custom_imports else ''
        strictness = (
            "IMPORTANT: Only use methods, fields, and constructors that are present in the code blocks below. "
            "Do NOT invent or assume any methods, fields, or classes. "
            "If a method or field is not present in the provided code, do NOT use it in your test. "
            "If you are unsure, leave it out. "
            "If you hallucinate any code, you will be penalized and re-prompted. "
            "You must generate a compiling, passing, and style-compliant test class. "
            "If the class or its methods use a logger (e.g., org.slf4j.Logger, log.info, log.error, etc.), write tests that verify logging behavior where appropriate. "
            "Use Mockito or other suitable techniques to verify that logging statements are called as expected, especially for error or important info logs. "
            "If you are unsure how to verify logging, add a comment in the test indicating what should be checked. "
            "Generate tests for exception/negative paths (e.g., when repo throws). "
            "Cover edge cases and all branches (e.g., with/without working location). "
            "Do NOT define or create dummy DTOs, entities, or repository interfaces inside the test class. Use only the real classes provided in the context. If a class is missing, do NOT invent it—report an error instead."
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
        print(f"[DEBUG] About to call LLM with prompt length: {len(prompt)} characters")
        result = self.llm.invoke(prompt)
        print(f"[DEBUG] LLM call completed")
        if hasattr(result, "content"):
            result = result.content
        print(f"[DEBUG] Returning result with length: {len(result)} characters")
        return result.strip()

    def generate_whole_class_test_with_feedback(self, target_class_name, target_package_name, class_code, public_methods, custom_imports, imports_context, feedback_msg, formatted_examples=None, previous_test_class_code=None):
        method_list_str = '\n'.join(f'- {m}' for m in public_methods)
        imports_section = '\n'.join(custom_imports) if custom_imports else ''
        strictness = (
            "IMPORTANT: Only use methods, fields, and constructors that are present in the code blocks below. "
            "Do NOT invent or assume any methods, fields, or classes. "
            "If a method or field is not present in the provided code, do NOT use it in your test. "
            "If you are unsure, leave it out. "
            "If you hallucinate any code, you will be penalized and re-prompted. "
            "You must generate a compiling, passing, and style-compliant test class. "
            "If the class or its methods use a logger (e.g., org.slf4j.Logger, log.info, log.error, etc.), write tests that verify logging behavior where appropriate. "
            "Use Mockito or other suitable techniques to verify that logging statements are called as expected, especially for error or important info logs. "
            "If you are unsure how to verify logging, add a comment in the test indicating what should be checked. "
            "Generate tests for exception/negative paths (e.g., when repo throws). "
            "Cover edge cases and all branches (e.g., with/without working location). "
            "Do NOT define or create dummy DTOs, entities, or repository interfaces inside the test class. Use only the real classes provided in the context. If a class is missing, do NOT invent it—report an error instead."
        )
        prompt = ''
        if feedback_msg:
            prompt += f"{feedback_msg}\n\n"
        if previous_test_class_code:
            prompt += f"PREVIOUS TEST CASE OUTPUT:\n--- BEGIN PREVIOUS OUTPUT ---\n{previous_test_class_code}\n--- END PREVIOUS OUTPUT ---\n\n"
        if formatted_examples:
            prompt += f"// The following are real test examples from your codebase.\n{formatted_examples}\n\n"
        prompt += f"""
You are an expert Java developer. You are to fix and regenerate a complete JUnit 5 + Mockito test class for the MAIN CLASS below. The other classes are provided as context only (do NOT generate tests for them).

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
- If error feedback includes a FAILED TEST FILE CONTENT section, analyze that failed code and fix the specific issues mentioned in the error messages.
- Pay special attention to compilation errors and fix them precisely.

{strictness}
"""
        result = self.llm.invoke(prompt)
        if hasattr(result, "content"):
            result = result.content
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
        # --- NEW: Early exit if test file already exists ---
        if test_output_file_path.exists():
            print(f"[SKIP] Test file already exists (inside generate_test_case): '{test_output_file_path}'")
            print(f"Skipping test generation for {target_class_name} (generate_test_case)")
            return ""

        # --- Retrieve the full class code for the main class under test (single assignment, guaranteed) ---
        main_class_filename = Path(target_info['java_file_path_abs']).name if target_info else None
        # [FIX] Handle relevant_java_files_for_context as a list of file paths (strings)
        dependency_filenames = [Path(f).name for f in relevant_java_files_for_context if Path(f).name != main_class_filename]
        utility_filenames = [Path(f).name for f in relevant_java_files_for_context if Path(f).name not in dependency_filenames and Path(f).name != main_class_filename]
        self._update_retriever_filter(main_class_filename, dependency_filenames, utility_filenames, k_override=5)
        retrieved_docs = self.retriever.get_relevant_documents(f"Full code for {target_class_name}")
        class_code = "\n\n".join([doc.page_content for doc in retrieved_docs if target_class_name in doc.page_content])
        if not class_code:
            class_code = "\n\n".join([doc.page_content for doc in retrieved_docs])
        if not class_code:
            class_code = "// ERROR: Could not retrieve main class code from ChromaDB."
        print("\n[DEBUG] CLASS CODE SENT TO LLM (FULL):\n" + class_code + "\n[END DEBUG CLASS CODE]\n")

        # [FIX] Extract public methods from class_code
        public_methods = self.extract_public_methods(class_code)

        # --- Extract project-local imports and load their code as context from ChromaDB ---
        print(f"[DEBUG] Extracting and fetching imported class context for: {target_class_name}")
        
        # Debug: Print all import lines in the class code
        print(f"[DEBUG] All import lines in {target_class_name}:")
        for line in class_code.split('\n'):
            if line.strip().startswith('import '):
                print(f"  Found import: {line.strip()}")
        
        # Fixed pattern to capture all com.iemr imports with multiple dots
        import_pattern = re.compile(r'^import\s+(com\.iemr\.[\w\.]+)\.([A-Z][A-Za-z0-9_]*)\;', re.MULTILINE)
        imported_classes = []
        missing_dependencies = []
        
        # --- NEW: Print all imports found in the class code ---
        all_imports_found = []
        for match in import_pattern.finditer(class_code):
            full_import = match.group(0)
            class_name = match.group(2)
            all_imports_found.append((full_import, class_name))
        
        print(f"[IMPORTS DEBUG] Found {len(all_imports_found)} local imports in {target_class_name}:")
        for full_import, class_name in all_imports_found:
            print(f"  - {full_import} (class: {class_name})")
        
        for match in import_pattern.finditer(class_code):
            class_name = match.group(2)
            dep_java_filename = f"{class_name}.java"
            print(f"[DEBUG] Fetching import: {dep_java_filename}")
            # Fix: Pass the filename as main_class_filename and empty list as dependency_filenames
            self._update_retriever_filter(dep_java_filename, [], k_override=10)
            dep_docs = self.retriever.get_relevant_documents(f"Full code for {class_name}")
            print(f"[DEBUG] Retrieved {len(dep_docs)} docs from ChromaDB for import {class_name}.")
            dep_code = "\n\n".join([doc.page_content for doc in dep_docs if class_name in doc.page_content])
            if not dep_code:
                dep_code = "\n\n".join([doc.page_content for doc in dep_docs])
            if dep_code:
                imported_classes.append((class_name, dep_code))
                print(f"[DEBUG] Added imported class context for {class_name} from ChromaDB.")
            else:
                print(f"[WARNING] Imported class code not found in ChromaDB: {dep_java_filename}")
                missing_dependencies.append(class_name)
        if missing_dependencies:
            print(f"[ERROR] Missing dependencies for {target_class_name}: {missing_dependencies}. Skipping test generation.")
            return f"// Test not generated: missing dependency class(es): {', '.join(missing_dependencies)}"
        
        # --- NEW: Print summary of imported classes being fed to LLM ---
        print(f"\n[LLM CONTEXT SUMMARY] The following {len(imported_classes)} imported classes will be fed to the LLM as context:")
        for i, (class_name, dep_code) in enumerate(imported_classes, 1):
            # Get a preview of the class (first few lines)
            preview_lines = dep_code.split('\n')[:5]
            preview = '\n'.join(preview_lines)
            print(f"  {i}. {class_name}")
            print(f"     Preview: {preview}")
            if len(dep_code.split('\n')) > 5:
                remaining_lines = len(dep_code.split('\n')) - 5
                print(f"     ... (and {remaining_lines} more lines)")
            print()
        
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
        
        # Add option to disable RAG if it's causing issues
        DISABLE_RAG = os.getenv("DISABLE_RAG", "false").lower() == "true"
        if DISABLE_RAG or self.test_examples_vectorstore is None:
            if DISABLE_RAG:
                print(f"[RAG] RAG disabled via environment variable. Skipping test example retrieval.")
            else:
                print(f"[RAG] Test examples collection not available. Skipping test example retrieval.")
            results = []
        else:
            # Add timeout protection and debugging for RAG
            try:
                print(f"[RAG DEBUG] Method signatures: {method_sigs}")
                print(f"[RAG DEBUG] Class code length: {len(class_code)} characters")
                
                # Limit the query size to prevent hanging
                max_query_length = 10000  # Limit to 10K characters
                query = class_code + '\n' + '\n'.join(method_sigs)
                if len(query) > max_query_length:
                    print(f"[RAG WARNING] Query too long ({len(query)} chars), truncating to {max_query_length}")
                    query = query[:max_query_length]
                
                print(f"[RAG DEBUG] Query length: {len(query)} characters")
                print(f"[RAG DEBUG] Starting embedding generation...")
                
                query_embedding = self.embeddings.embed_documents([query])[0]
                print(f"[RAG DEBUG] Embedding generated successfully, length: {len(query_embedding)}")
                
                print(f"[RAG DEBUG] Starting similarity search...")
                # Fix: Define top_n here
                top_n = 3
                results = self.test_examples_vectorstore.similarity_search_by_vector(
                    query_embedding, k=top_n, filter={'type': 'test_example'}
                )
                print(f"[RAG DEBUG] Similarity search completed, found {len(results)} results")
                
            except Exception as e:
                print(f"[RAG ERROR] Failed to retrieve test examples: {e}")
                print(f"[RAG] Continuing without test examples...")
                results = []
        
        formatted_examples = self.format_test_examples_for_prompt(results) if results else ''
        if formatted_examples:
            print(f"[RAG] Retrieved {len(results)} real test examples. Including in prompt.")
        else:
            print(f"[RAG] No real test examples found for this class.")

        print(f"[DEBUG] RAG step completed. Moving to test generation...")

        # --- Automated feedback loop for hallucinated members ---
        feedback_attempt = 0
        hallucinated = set()
        previous_test_class_code = None
        print(f"[DEBUG] Starting automated feedback loop...")
        max_hallucination_retries = 3  # Limit to 3 attempts to prevent infinite loop
        while feedback_attempt <= max_hallucination_retries:
            print(f"[DEBUG] Feedback attempt {feedback_attempt + 1}/{max_hallucination_retries + 1}")
            if feedback_attempt == 0:
                print(f"[DEBUG] Generating initial test class...")
                test_class_code = self.generate_whole_class_test_with_context(target_class_name, target_package_name, class_code, public_methods, custom_imports, imports_context, formatted_examples)
                print(f"[DEBUG] Initial test class generation completed.")
            else:
                print(f"[DEBUG] Generating test class with feedback...")
                feedback_msg = (
                    "ERRORS DETECTED IN PREVIOUS TEST CASE:\n"
                    f"{feedback_msg}\n\n"
                    "INSTRUCTIONS:\n"
                    "- Only fix the errors listed above.\n"
                    "- Do NOT change any code that is not related to these errors.\n"
                    "- If you are unsure, add a comment in the code explaining your reasoning.\n"
                    "- Do NOT invent new methods, fields, or classes.\n"
                    "- Output ONLY the corrected test class code, no explanations or markdown.\n"
                    "IMPORTANT: If you change working code, you will be penalized."
                )
                test_class_code = self.generate_whole_class_test_with_feedback(target_class_name, target_package_name, class_code, public_methods, custom_imports, imports_context, feedback_msg, formatted_examples, previous_test_class_code)
                print(f"[DEBUG] Test class generation with feedback completed.")
            previous_test_class_code = test_class_code
            print(f"[DEBUG] Checking for hallucinated members...")
            hallucinated = self.find_hallucinated_members(test_class_code, imported_class_members)
            print(f"[DEBUG] Found {len(hallucinated)} hallucinated members: {hallucinated}")
            
            # Only break if no hallucinated members OR if we've reached max retries
            if not hallucinated or feedback_attempt >= max_hallucination_retries:
                if feedback_attempt >= max_hallucination_retries:
                    print(f"[DEBUG] Reached max hallucination retries ({max_hallucination_retries}). Continuing with current test class.")
                else:
                    print(f"[DEBUG] No hallucinated members found. Breaking feedback loop.")
                break
                
            # --- Stronger feedback for hallucination ---
            feedback_msg = ("ERROR: In your previous output, you used the following members which do NOT exist in the provided code: "
                            f"{', '.join(hallucinated)}. STRICTLY do NOT use these or any other non-existent members. Only use what is present in the code blocks below. If you hallucinate again, your output will be rejected and you will be penalized. If you are unsure, leave it out or add a comment.")
            feedback_attempt += 1
            print(f"[DEBUG] Incrementing feedback attempt to {feedback_attempt}")
        
        print(f"[DEBUG] Feedback loop completed after {feedback_attempt} attempts.")
        test_class_code = test_class_code.strip()
        test_class_code = re.sub(r'^```[a-zA-Z]*\n', '', test_class_code)
        test_class_code = re.sub(r'^```', '', test_class_code)
        test_class_code = re.sub(r'```$', '', test_class_code)
        test_class_code = test_class_code.strip()
        print(f"[DEBUG] Test class code cleaned up. Length: {len(test_class_code)} characters")
        
        # REMOVED: Don't write to file here - wait for final merged result
        
        # [FIX] Only accumulate successful outputs for each group
        all_test_class_outputs = []
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
        while set(plan_methods) != set(real_methods) and retries < MAX_TEST_GENERATION_RETRIES:
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
        print(f"[DEBUG] Processing {len(method_groups)} method groups: {method_groups}")
        
        # Add timeout protection for group processing
        group_processing_start_time = time.time()
        max_group_processing_time = 600  # 10 minutes max for all groups
        
        # Initialize accumulation of passing test methods and field declarations
        accumulated_test_methods = []
        accumulated_field_declarations = []  # NEW: Accumulate field declarations
        successful_groups = 0
        
        for group_index, group in enumerate(method_groups):
            # Check if we've exceeded the maximum processing time
            if time.time() - group_processing_start_time > max_group_processing_time:
                print(f"[WARNING] Group processing exceeded {max_group_processing_time} seconds. Using accumulated tests.")
                break
            
            print(f"[DEBUG] Processing group {group_index + 1}/{len(method_groups)}: {group}")
            group_methods = [tc for tc in test_plan if tc['method_name'] in group]
            anti_hallucination = "Do NOT invent methods, do NOT mock POJOs/value objects, only mock what is required for compilation or isolation. BAD: Mocks unused dependencies. GOOD: Only mocks required dependencies. If unsure, prefer not to mock. INCLUDE ALL PUBLIC METHODS, INCLUDING STATIC METHODS."
            positive_example = "// GOOD: Only mocks required dependencies, covers all public and static methods."
            negative_example = "// BAD: Mocks unused dependencies, skips methods, causes build to fail."
            prompt_instructions = f"{additional_query_instructions}\n{anti_hallucination}\n{positive_example}\n{negative_example}"
            
            print(f"[DEBUG] Calling generate_whole_class_test_with_context for group {group_index + 1}...")
            try:
                group_test_code = self.generate_whole_class_test_with_context(target_class_name, target_package_name, class_code, group, custom_imports, imports_context, formatted_examples)
                print(f"[DEBUG] generate_whole_class_test_with_context completed for group {group_index + 1}")
            except Exception as e:
                print(f"[ERROR] LLM call failed for group {group_index + 1}: {e}")
                print(f"[WARNING] Skipping group {group_index + 1} due to LLM failure.")
                continue
            
            print(f"[DEBUG] RAW LLM OUTPUT for group {group}:\n{group_test_code}\n--- END RAW LLM OUTPUT ---")
            group_test_code = group_test_code.strip()
            group_test_code = re.sub(r'^```[a-zA-Z]*\n', '', group_test_code)
            group_test_code = re.sub(r'^```', '', group_test_code)
            group_test_code = re.sub(r'```$', '', group_test_code)
            group_test_code = group_test_code.strip()
            
            # Try to get valid test methods from this group
            group_valid_methods = []
            error_feedback = None
            
            for attempt in range(MAX_TEST_GENERATION_RETRIES):
                print(f"[DEBUG] Test generation retry {attempt+1} for group {group_index + 1}: {group}")
                
                if error_feedback and attempt > 0:
                    feedback_msg = (
                        "ERRORS DETECTED IN PREVIOUS TEST CASE:\n"
                        f"{error_feedback}\n\n"
                        "INSTRUCTIONS:\n"
                        "- Only fix the errors listed above.\n"
                        "- Do NOT change any code that is not related to these errors.\n"
                        "- If you are unsure, add a comment in the code explaining your reasoning.\n"
                        "- Do NOT invent new methods, fields, or classes.\n"
                        "- Output ONLY the corrected test class code, no explanations or markdown.\n"
                        "IMPORTANT: If you change working code, you will be penalized."
                    )
                    print(f"[LLM ERROR FEEDBACK] Feeding the following error message to LLM (attempt {attempt+1}):\n{feedback_msg}\n--- END ERROR FEEDBACK ---")
                    group_test_code = self.generate_whole_class_test_with_feedback(
                        target_class_name, target_package_name, class_code, group, custom_imports, imports_context, feedback_msg, formatted_examples, group_test_code
                    )
                    print(f"[DEBUG] RAW LLM OUTPUT (after feedback) for group {group}:\n{group_test_code}\n--- END RAW LLM OUTPUT ---")
                    group_test_code = group_test_code.strip()
                    group_test_code = re.sub(r'^```[a-zA-Z]*\n', '', group_test_code)
                    group_test_code = re.sub(r'^```', '', group_test_code)
                    group_test_code = re.sub(r'```$', '', group_test_code)
                    group_test_code = group_test_code.strip()
                
                # Extract test methods and field declarations from this group's code
                field_declarations, extracted_methods = self.extract_test_class_components(group_test_code)
                print(f"[DEBUG] Extracted {len(extracted_methods)} test methods and {len(field_declarations)} field declarations from group {group_index + 1}")
                
                if extracted_methods:
                    # Create a test class with just these methods to test compilation
                    temp_test_class = self.assemble_test_class(target_package_name, custom_imports, target_class_name, extracted_methods, class_code, field_declarations)
                    
                    # Use temporary file for compilation testing (FIXED)
                    with tempfile.NamedTemporaryFile(mode='w', suffix='.java', delete=False, encoding='utf-8') as temp_file:
                        temp_file.write(temp_test_class)
                        temp_file_path = temp_file.name
                    
                    try:
                        print(f"[DEBUG] Testing temporary test file for group {group_index + 1}...")
                        test_run_results = self.java_test_runner.run_test(Path(temp_file_path))
                        print(f"[DEBUG] Test run completed for group {group_index + 1}")
                        
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
                            print(f"[SUCCESS] Group {group_index + 1} tests compile and run successfully!")
                            group_valid_methods = extracted_methods
                            successful_groups += 1
                            break  # FIXED: Break out of attempt loop on success
                        else:
                            error_msgs = []
                            if compilation_errors:
                                print(f"[COMPILATION ERROR] Detected after attempt {attempt+1} for group {group_index + 1}:")
                                for err in compilation_errors:
                                    print(f"  - {err['message']} (at {err['location']})")
                                    error_msgs.append(f"COMPILATION: {err['message']} (at {err['location']})")
                            if test_failures:
                                print(f"[TEST FAILURE] Detected after attempt {attempt+1} for group {group_index + 1}:")
                                for err in test_failures:
                                    print(f"  - {err['message']} (in {err['location']})")
                                    error_msgs.append(f"TEST: {err['message']} (in {err['location']})")
                            if tests_run_zero:
                                error_msgs.append("NO TESTS RUN: The generated test class did not contain any executable tests. Ensure at least one @Test method is present and not ignored/skipped.")
                            
                            # Read the failed test file content for better error feedback
                            failed_test_content = ""
                            try:
                                with open(temp_file_path, 'r', encoding='utf-8') as f:
                                    failed_test_content = f.read()
                                print(f"[DEBUG] Read failed test file content for error feedback (length: {len(failed_test_content)} chars)")
                            except Exception as e:
                                print(f"[WARNING] Could not read failed test file for error feedback: {e}")
                            
                            error_feedback = '\n'.join(error_msgs)
                            
                            # Include the failed test content in error feedback for next attempt
                            if failed_test_content:
                                error_feedback += f"\n\nFAILED TEST FILE CONTENT:\n--- BEGIN FAILED TEST ---\n{failed_test_content}\n--- END FAILED TEST ---\n"
                                print(f"[DEBUG] Added failed test file content to error feedback for next LLM attempt")
                            else:
                                print(f"[DEBUG] No failed test file content available for error feedback")
                    finally:
                        # Clean up temporary file
                        if os.path.exists(temp_file_path):
                            os.unlink(temp_file_path)
                else:
                    print(f"[WARNING] No test methods extracted from group {group_index + 1}")
                    error_feedback = "No valid test methods found. Generate proper @Test methods."
            
            # FIXED: Add successful test methods and field declarations to accumulation (moved outside the attempt loop)
            if group_valid_methods:
                accumulated_test_methods.extend(group_valid_methods)
                # NEW: Also accumulate field declarations from this group
                if field_declarations:
                    # Only add field declarations that aren't already accumulated (avoid duplicates)
                    for field_decl in field_declarations:
                        if field_decl not in accumulated_field_declarations:
                            accumulated_field_declarations.append(field_decl)
                print(f"[SUCCESS] Added {len(group_valid_methods)} test methods and {len(field_declarations)} field declarations from group {group_index + 1} to accumulation")
            else:
                print(f"[ERROR] Failed to generate valid tests for group {group_index + 1} after {MAX_TEST_GENERATION_RETRIES} attempts")
        
        print(f"[DEBUG] Group processing completed. {successful_groups}/{len(method_groups)} groups successful.")
        print(f"[DEBUG] Total accumulated test methods: {len(accumulated_test_methods)}")
        print(f"[DEBUG] Total accumulated field declarations: {len(accumulated_field_declarations)}")
        for i, field_decl in enumerate(accumulated_field_declarations, 1):
            print(f"  {i}. {field_decl}")
        
        # Remove duplicate test methods by method name
        unique_methods = {}
        for method in accumulated_test_methods:
            match = re.search(r'void\s+(\w+)\s*\(', method)
            if match:
                method_name = match.group(1)
                if method_name not in unique_methods:
                    unique_methods[method_name] = method
        
        print(f"[DEBUG] After deduplication: {len(unique_methods)} unique test methods")
        
        # Generate final test class with all accumulated methods and field declarations
        if unique_methods:
            final_test_class_code = self.assemble_test_class(target_package_name, custom_imports, target_class_name, list(unique_methods.values()), class_code, accumulated_field_declarations)
            final_test_class_code = final_test_class_code.strip()
            final_test_class_code = re.sub(r'^```[a-zA-Z]*\n', '', final_test_class_code)
            final_test_class_code = re.sub(r'^```', '', final_test_class_code)
            final_test_class_code = re.sub(r'```$', '', final_test_class_code)
            final_test_class_code = final_test_class_code.strip()
            
            # FIXED: Final validation of the complete test class
            print(f"[DEBUG] Performing final validation of complete test class...")
            with tempfile.NamedTemporaryFile(mode='w', suffix='.java', delete=False, encoding='utf-8') as temp_file:
                temp_file.write(final_test_class_code)
                temp_file_path = temp_file.name
            
            try:
                final_test_results = self.java_test_runner.run_test(Path(temp_file_path))
                final_compilation_errors = final_test_results['detailed_errors'].get('compilation_errors', [])
                final_test_failures = final_test_results['detailed_errors'].get('test_failures', [])
                final_tests_run_zero = False
                final_stdout = final_test_results.get('stdout', '')
                final_test_summary_match = re.search(r"Tests run: (\d+), Failures: (\d+), Errors: (\d+), Skipped: (\d+)", final_stdout)
                if final_test_summary_match:
                    final_total = int(final_test_summary_match.group(1))
                    if final_total == 0:
                        final_tests_run_zero = True
                
                if not final_compilation_errors and not final_test_failures and not final_tests_run_zero:
                    print(f"[SUCCESS] Final test class validation passed! Returning complete test class.")
                    print(f"[DEBUG] FINAL MERGED TEST CLASS CODE:\n{final_test_class_code}\n--- END FINAL TEST CLASS CODE ---")
                    return final_test_class_code
                else:
                    print(f"[WARNING] Final test class has issues. Falling back to simple method.")
                    print(f"  Compilation errors: {len(final_compilation_errors)}")
                    print(f"  Test failures: {len(final_test_failures)}")
                    print(f"  Tests run zero: {final_tests_run_zero}")
                    return self.generate_simple_test_case(target_class_name, target_package_name, class_code, custom_imports, imports_context)
            finally:
                # Clean up temporary file
                if os.path.exists(temp_file_path):
                    os.unlink(temp_file_path)
        else:
            print(f"[WARNING] No valid test methods accumulated. Falling back to simple method.")
            return self.generate_simple_test_case(target_class_name, target_package_name, class_code, custom_imports, imports_context)

    def extract_public_methods(self, class_code: str) -> set:
        """
        Extracts all public method names from the given Java class code using regex.
        Includes constructors and static methods. Handles annotations, generics, and line breaks.
        """
        methods = set()
        
        # Match public constructors (same name as class)
        constructor_pattern = re.compile(r'public\s+(\w+)\s*\([^)]*\)\s*\{')
        for match in constructor_pattern.finditer(class_code):
            methods.add(match.group(1))
        
        # Improved pattern for Spring controller methods (handles @ResponseBody and other annotations)
        method_pattern = re.compile(
            r'public\s+'  # public modifier
            r'(?:@[\w.]+\s*)*'  # Annotations after public (like @ResponseBody)
            r'(?:[\w<>\[\],\s]+\s+)*'  # Return type
            r'(\w+)\s*\(',  # Method name
            re.MULTILINE
        )
        for match in method_pattern.finditer(class_code):
            methods.add(match.group(1))
        
        # Match public static methods that might have different patterns
        static_method_pattern = re.compile(r'public\s+static\s+(?:final\s+)?(?:<[^>]+>\s+)?[\w<>,\[\]]+\s+(\w+)\s*\(')
        for match in static_method_pattern.finditer(class_code):
            methods.add(match.group(1))
        
        print(f"[DEBUG] extract_public_methods found: {methods}")
        return methods

    def extract_test_plan_methods(self, test_plan: list) -> set:
        """
        Extracts method names from the LLM-generated test plan.
        """
        return set(tc['method_name'] for tc in test_plan if 'method_name' in tc)

    def assemble_test_class(self, package_name: str, imports: List[str], class_name: str, test_methods: List[str], class_code: str = "", field_declarations: List[str] = None) -> str:
        """
        Assemble the final test class file with package, imports, annotations, class definition, field declarations, and all test methods.
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
        
        # Compose field declarations section
        field_section = ""
        if field_declarations:
            field_section = "\n\n".join(field_declarations)
        
        # Join all test methods
        methods_section = "\n\n".join(m for m in test_methods if m.strip())
        
        # Final assembly
        result = f"{package_stmt}\n{import_section}\n"
        if static_import_section:
            result += f"{static_import_section}\n"
        result += f"\n{class_anno}\n{class_def}\n"
        
        if field_section:
            result += f"\n{field_section}\n"
        
        if methods_section:
            result += f"\n{methods_section}\n"
        
        result += "\n}"
        
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
            # Only check for method/field usage, not class references
            usage_pattern = re.compile(rf'{re.escape(class_name)}\.([A-Za-z0-9_]+)')
            for match in usage_pattern.finditer(test_code):
                member = match.group(1)
                if member not in members:
                    # Skip common class references that are likely valid
                    if member.lower() in ['class', 'type', 'getclass']:
                        continue
                    hallucinated.add(f'{class_name}.{member}')
            
            # Check for new instance creation followed by method calls
            new_pattern = re.compile(rf'new\s+{re.escape(class_name)}\(\)\.([A-Za-z0-9_]+)')
            for match in new_pattern.finditer(test_code):
                member = match.group(1)
                if member not in members:
                    # Skip common class references that are likely valid
                    if member.lower() in ['class', 'type', 'getclass']:
                        continue
                    hallucinated.add(f'{class_name}.{member}')
        return hallucinated

    def generate_simple_test_case(self, target_class_name: str, target_package_name: str, class_code: str, custom_imports: List[str], imports_context: str = None) -> str:
        """
        Simplified test generation with error feedback loop.
        Generates tests in a single pass with focused prompts and fixes compilation errors.
        Optionally includes imported class context for better dependency handling.
        """
        # Extract public methods using static analysis
        public_methods = self.extract_public_methods(class_code)
        if not public_methods:
            return "// No public methods found to test"
        
        method_list = "\n".join(f"- {method}" for method in sorted(public_methods))
        
        # Enhanced prompt with optional context
        context_section = ""
        if imports_context:
            context_section = f"""
IMPORTED CLASSES CONTEXT (for reference only - do NOT generate tests for these):
{imports_context}

"""
        
        # Improved prompt with clearer structure requirements
        base_prompt = f"""
You are an expert Java test developer. Generate a complete, properly structured JUnit 5 + Mockito test class for the following Java class.

{context_section}CLASS UNDER TEST:
{class_code}

REQUIREMENTS:
- Test class name: {target_class_name}Test
- Package: {target_package_name}
- Test ALL these public methods: {method_list}
- Use @ExtendWith(MockitoExtension.class)
- Include @Mock and @InjectMocks annotations properly
- Mock only dependencies that are actually used in each test method
- Include ALL necessary imports (Arrays, Collections, etc.)
- Test both success and failure scenarios
- Verify logging behavior if present (use ArgumentCaptor for log messages)
- Test edge cases and exception handling
- Use descriptive test method names (e.g., testMethodName_WhenCondition_ShouldReturnExpected)
- Follow AAA pattern (Arrange, Act, Assert)
- Use meaningful assertions with clear error messages
- Ensure ALL test methods are complete and properly closed

IMPORTS TO USE:
{chr(10).join(custom_imports)}

IMPORTANT:
- Do NOT invent methods, fields, or classes not present in the source code
- Do NOT mock simple POJOs or value objects
- Only mock dependencies that are actually injected or used
- If unsure about a dependency, prefer not to mock it
- Ensure all tests compile and run without errors
- Generate COMPLETE test methods - do not truncate or leave incomplete
- Include proper class structure with all annotations
- Use the imported classes context above for proper dependency handling

Output ONLY the complete test class code, no explanations or markdown.
"""
        
        # Error feedback loop
        max_attempts = 5
        previous_test_code = None
        
        for attempt in range(max_attempts):
            try:
                if attempt == 0:
                    # First attempt - use base prompt
                    prompt = base_prompt
                else:
                    # Subsequent attempts - add error feedback
                    prompt = f"""
ERRORS DETECTED IN PREVIOUS TEST CASE:
{error_feedback}

INSTRUCTIONS:
- Only fix the errors listed above.
- Do NOT change any code that is not related to these errors.
- If you are unsure, add a comment in the code explaining your reasoning.
- Do NOT invent new methods, fields, or classes.
- Output ONLY the corrected test class code, no explanations or markdown.
- Pay special attention to compilation errors and fix them precisely.

{base_prompt}
"""
                
                result = self.llm.invoke(prompt)
                if hasattr(result, "content"):
                    result = result.content
                
                # Clean up the output
                code = result.strip()
                code = re.sub(r'^```[a-zA-Z]*\n', '', code)
                code = re.sub(r'^```', '', code)
                code = re.sub(r'```$', '', code)
                code = code.strip()
                
                # Validate that the output looks like a complete test class
                if not code.startswith('package ') and not code.startswith('import '):
                    print(f"[WARNING] Generated code doesn't look like a proper test class. Regenerating...")
                    # Try one more time with a simpler prompt
                    simple_prompt = f"""
Generate a complete JUnit 5 test class for {target_class_name} with these methods: {method_list}
Include proper imports, @ExtendWith(MockitoExtension.class), @Mock, @InjectMocks, and complete test methods.
Output ONLY the Java code.
"""
                    result = self.llm.invoke(simple_prompt)
                    if hasattr(result, "content"):
                        result = result.content
                    code = result.strip()
                    code = re.sub(r'^```[a-zA-Z]*\n', '', code)
                    code = re.sub(r'^```', '', code)
                    code = re.sub(r'```$', '', code)
                    code = code.strip()
                
                # Test compilation (create temporary file for testing)
                with tempfile.NamedTemporaryFile(mode='w', suffix='.java', delete=False, encoding='utf-8') as temp_file:
                    temp_file.write(code)
                    temp_file_path = temp_file.name
                
                try:
                    # Test the generated code
                    test_run_results = self.java_test_runner.run_test(Path(temp_file_path))
                    
                    compilation_errors = test_run_results['detailed_errors'].get('compilation_errors', [])
                    test_failures = test_run_results['detailed_errors'].get('test_failures', [])
                    tests_run_zero = False
                    stdout = test_run_results.get('stdout', '')
                    test_summary_match = re.search(r"Tests run: (\d+), Failures: (\d+), Errors: (\d+), Skipped: (\d+)", stdout)
                    if test_summary_match:
                        total = int(test_summary_match.group(1))
                        if total == 0:
                            tests_run_zero = True
                    
                    if not compilation_errors and not test_failures and not tests_run_zero:
                        print(f"[SUCCESS] Simple method generated working test after {attempt + 1} attempts!")
                        os.unlink(temp_file_path)  # Clean up temp file
                        return code
                    else:
                        # Collect error messages for next attempt
                        error_msgs = []
                        if compilation_errors:
                            for err in compilation_errors:
                                error_msgs.append(f"COMPILATION: {err['message']} (at {err['location']})")
                        if test_failures:
                            for err in test_failures:
                                error_msgs.append(f"TEST: {err['message']} (in {err['location']})")
                        if tests_run_zero:
                            error_msgs.append("NO TESTS RUN: The generated test class did not contain any executable tests.")
                        
                        error_feedback = '\n'.join(error_msgs)
                        previous_test_code = code
                        print(f"[ATTEMPT {attempt + 1}] Test has errors, retrying with feedback...")
                        
                finally:
                    # Clean up temp file
                    if os.path.exists(temp_file_path):
                        os.unlink(temp_file_path)
                
            except Exception as e:
                print(f"[ERROR] LLM call failed in simple method (attempt {attempt + 1}): {e}")
                if attempt == max_attempts - 1:
                    break
        
        # If all attempts failed, return the last generated code or fallback
        if previous_test_code:
            print(f"[WARNING] Simple method failed after {max_attempts} attempts. Returning last generated code.")
            return previous_test_code
        else:
            print(f"[ERROR] Simple method failed completely. Returning fallback template.")
            # Return a basic template as fallback
            return f"""package {target_package_name};

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.InjectMocks;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;

@ExtendWith(MockitoExtension.class)
class {target_class_name}Test {{
    
    @Mock
    private Object mockDependency;
    
    @InjectMocks
    private {target_class_name} {target_class_name.lower()}Service;
    
    // TODO: Add test methods for: {method_list}
    @Test
    void testPlaceholder() {{
        // Placeholder test - implement actual tests
        assertTrue(true);
    }}
}}"""

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

            # Verify source file exists
            if not java_file_path_abs.exists():
                print(f"\n[ERROR] Source file does not exist: '{java_file_path_abs}'")
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
            # Use simple method for better reliability with error feedback
            USE_SIMPLE_METHOD = False  # Set to True to use simple method with error feedback
            
            if USE_SIMPLE_METHOD:
                # Get class code directly from the main class file
                with open(java_file_path_abs, 'r', encoding='utf-8') as f:
                    class_code = f.read()
                
                # OPTIONAL: Fetch imported class context for better dependency handling
                imports_context = None
                # TEMPORARILY DISABLED: Imported class context fetching to avoid 'self' error
                # try:
                #     # Extract project-local imports and load their code as context
                #     import_pattern = re.compile(r'^import\s+(com\.iemr\.[\w\.]+)\.([A-Z][A-Za-z0-9_]*)\;', re.MULTILINE)
                #     imported_classes = []
                #     
                #     for match in import_pattern.finditer(class_code):
                #         class_name = match.group(2)
                #         dep_java_filename = f"{class_name}.java"
                #         test_generator._update_retriever_filter(dep_java_filename, [], k_override=10)
                #         dep_docs = test_generator.retriever.get_relevant_documents(f"Full code for {class_name}")
                #         dep_code = "\n\n".join([doc.page_content for doc in dep_docs if class_name in doc.page_content])
                #         if not dep_code:
                #             dep_code = "\n\n".join([doc.page_content for doc in dep_docs])
                #         if dep_code:
                #             imported_classes.append((class_name, dep_code))
                #     
                #     if imported_classes:
                #         imports_context = ""
                #         for class_name, dep_code in imported_classes:
                #             imports_context += f"\n--- BEGIN IMPORTED CLASS: {class_name} ---\n{dep_code}\n--- END IMPORTED CLASS: {class_name} ---\n"
                #         print(f"[DEBUG] Simple method will use context for {len(imported_classes)} imported classes")
                # except Exception as e:
                #     print(f"[WARNING] Could not fetch imported class context for simple method: {e}")
                #     imports_context = None
                
                generated_test_code = test_generator.generate_simple_test_case(
                    target_class_name=target_class_name,
                    target_package_name=target_package_name,
                    class_code=class_code,
                    custom_imports=custom_imports_list,
                    imports_context=imports_context
                )
                
                # Write the generated test code to file
                try:
                    test_output_file_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(test_output_file_path, 'w', encoding='utf-8') as f:
                        f.write(generated_test_code)
                    print(f"[SUCCESS] Simple method generated and validated test - no additional validation needed")
                except Exception as e:
                    print(f"[ERROR] Failed to write test file: {e}")
                    print(f"Skipping test generation for {target_class_name}")
                    continue
                
                # REMOVED: Redundant validation since simple method already validates internally
            else:
                # Use the complex method (original) with proper group processing and error feedback
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
                
                # Write the final generated test code to file
                if generated_test_code and generated_test_code.strip():
                    try:
                        test_output_file_path.parent.mkdir(parents=True, exist_ok=True)
                        with open(test_output_file_path, 'w', encoding='utf-8') as f:
                            f.write(generated_test_code)
                        print(f"[DEBUG] Final test class written to file: {test_output_file_path}")
                    except Exception as e:
                        print(f"[ERROR] Failed to write final test file: {e}")
                        print(f"Test code was generated but could not be saved for {target_class_name}")
                else:
                    print(f"[WARNING] No test code generated for {target_class_name}")
            
            print(f"\n[FINAL SUCCESS] Generated test case saved to: '{test_output_file_path}'")

            print("\n--- FINAL GENERATED TEST CASE (Printed to Console for review) ---")
            if 'generated_test_code' in locals():
                print(generated_test_code)
            else:
                print("No test code was generated.")
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


# NO-OP: Trigger reload for extract_public_methods visibility
import sys
from pathlib import Path
import os
import json 
from typing import List, Dict, Any, Union
import re
from dotenv import load_dotenv
load_dotenv()
import javalang




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
from llm.test_prompt_templates import (
    get_controller_test_prompt_template,
    get_service_test_prompt_template,
    TEST_QUALITY_CHECK_PROMPT,
    BEST_PRACTICES_EXAMPLE
)


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

{strictness}
"""
        result = self.llm.invoke(prompt)
        if hasattr(result, "content"):
            result = result.content
        return result.strip()

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
        direct_local_deps = self._get_direct_local_imports(main_class_code, SPRING_BOOT_MAIN_JAVA_DIR)
        all_deps = direct_local_deps
        print(f"[STRICT DEBUG] Main class: {main_class_filename}, Direct local deps: {all_deps}")
        print(f"FILES TO RETRIEVE CHUNKS FROM (for context): {[main_class_filename] + all_deps}")
        files_for_context = [main_class_filename] + all_deps
        print(f"[DEBUG] Actually retrieving code for these files: {files_for_context}")
        context = f"--- BEGIN MAIN CLASS UNDER TEST ---\n"
        main_code = self._get_full_code_from_chromadb(main_class_filename)
        context += main_code + "\n--- END MAIN CLASS UNDER TEST ---\n\n"
        for dep in all_deps:
            if dep == main_class_filename:
                continue
            dep_code = self._get_full_code_from_chromadb(dep)
            context += f"--- BEGIN DEPENDENCY: {dep} ---\n{dep_code}\n--- END DEPENDENCY: {dep} ---\n\n"
        # Now, use 'context' in the prompt
        public_methods = self.extract_public_methods(context)
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
            # Run the test file
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
        full_context += main_code + "\n--- END MAIN CLASS UNDER TEST ---\n\n"
        if test_type != "controller":
            for dep in all_deps:
                if dep == main_class_filename:
                    continue
                dep_code = self._get_full_code_from_chromadb(dep)
                dep_signatures = extract_class_signatures(dep_code)
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
            # Extract import statements
            import_lines = [line for line in main_code.splitlines() if line.strip().startswith('import ')]
            imports_section = ''
            if import_lines:
                imports_section = '--- BEGIN IMPORTS ---\n' + '\n'.join(import_lines) + '\n--- END IMPORTS ---\n\n'
            minimal_class_code = extract_minimal_class_for_methods(main_code, batch)
            # --- PATCH: Use transitive dependency resolution for all referenced types ---
            # Use resolve_transitive_dependencies to get all .java files referenced by the batch methods
            from analyzer.code_analysis_utils import resolve_transitive_dependencies
            transitive_deps = resolve_transitive_dependencies(Path(target_info['java_file_path_abs']), SPRING_BOOT_MAIN_JAVA_DIR)
            # Remove the main class itself from dependencies
            transitive_deps = [dep for dep in transitive_deps if Path(dep).name != main_class_filename]
            dep_signatures = []
            for dep_path in transitive_deps:
                dep_filename = Path(dep_path).name
                dep_code = self._get_full_code_from_chromadb(dep_filename)
                dep_sign = extract_class_signatures(dep_code)
                dep_signatures.append(f"--- BEGIN DEPENDENCY SIGNATURES: {dep_filename} ---\n{dep_sign}\n--- END DEPENDENCY SIGNATURES: {dep_filename} ---\n")
            # Build the minimal context
            minimal_context = imports_section
            minimal_context += f"--- BEGIN MAIN CLASS UNDER TEST (MINIMAL) ---\n{minimal_class_code}\n--- END MAIN CLASS UNDER TEST (MINIMAL) ---\n\n"
            minimal_context += "\n".join(dep_signatures)
            # Add strictness for no comments
            strict_no_comments = '\nSTRICT: Do NOT add any comments to the generated code.\n'
            minimal_context += strict_no_comments
            while retries < 15 and not success:
                context = minimal_context  # Use minimal context for every batch
                if test_type == "controller":
                    good_example = '''
// GOOD EXAMPLE (DO THIS):
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.InjectMocks;
import org.mockito.Mock;
import org.mockito.MockitoAnnotations;
import org.mockito.junit.jupiter.MockitoExtension;
import org.springframework.test.web.servlet.MockMvc;
import org.springframework.test.web.servlet.setup.MockMvcBuilders;

@ExtendWith(MockitoExtension.class)
class MyControllerTest {
    private MockMvc mockMvc;
    @Mock MyService myService;
    @InjectMocks MyController controller;

    @BeforeEach
    void setUp() {
        MockitoAnnotations.openMocks(this);
        mockMvc = MockMvcBuilders.standaloneSetup(controller).build();
    }

    @Test
    void shouldDoSomething() throws Exception {
        // test logic using mockMvc
    }
}
'''
                    forbidden = """
STRICT REQUIREMENTS:
- You MUST use standalone MockMvc with Mockito: @ExtendWith(MockitoExtension.class), @InjectMocks for the controller, @Mock for dependencies, and initialize MockMvc in @BeforeEach using MockMvcBuilders.standaloneSetup(controller).
- Do NOT use @WebMvcTest, @MockBean, @Autowired, or @SpringBootTest for controllers.
- Do NOT use field injection for MockMvc or dependencies.
- Do NOT generate any code for dependencies. Only generate the test class for the controller. Assume all dependencies exist and are available for mocking.
- If you use any forbidden annotation or pattern, you will be penalized and re-prompted.
- Output ONLY compilable Java code, no explanations or markdown.
"""
                    dependencies_context = ''  # No dependencies for controllers in batch mode
                    if i == 0 and retries == 0:
                        prompt = f"""{good_example}
{forbidden}
--- ENDPOINTS UNDER TEST ---
{endpoints_list_str}
--- END ENDPOINTS ---
{context}
You must generate a complete test class named {target_class_name}Test in package {target_package_name}, covering ONLY these methods:
{method_list_str}
"""
                    elif retries > 0:
                        # --- NEW: Structured error feedback prompt ---
                        with open(test_output_file_path, 'r', encoding='utf-8') as f:
                            failed_test_code = f.read()
                        # Parse error_feedback for summary
                        error_summary = ""
                        if error_feedback:
                            # Try to extract type, method, message, location from error_feedback
                            lines = error_feedback.split('\n')
                            for line in lines:
                                if line.startswith("COMPILATION:"):
                                    error_summary += f"Type: COMPILATION ERROR\nMessage: {line[12:].strip()}\n"
                                elif line.startswith("TEST:"):
                                    # Try to extract method and message
                                    msg = line[5:].strip()
                                    m = re.match(r"(.+?) \\(in (.+?)\\)", msg)
                                    if m:
                                        error_summary += f"Type: TEST FAILURE\nTest Method: {m.group(2)}\nMessage: {m.group(1)}\n"
                                    else:
                                        error_summary += f"Type: TEST FAILURE\nMessage: {msg}\n"
                                elif line.startswith("NO TESTS RUN"):
                                    error_summary += f"Type: NO TESTS RUN\nMessage: {line.strip()}\n"
                        if not error_summary:
                            error_summary = error_feedback or "Unknown error."
                        prompt = f"""
--- ERROR SUMMARY ---
{error_summary}

--- INSTRUCTIONS ---
- Fix ONLY the error(s) shown above.
- Do NOT change unrelated methods or code.
- Ensure the test class compiles and all tests pass.
- If you are unsure about a dependency or method, add a TODO comment.

--- FAILED TEST CLASS ---
{failed_test_code}
--- END FAILED TEST CLASS ---

{context}
You must generate a complete test class named {target_class_name}Test in package {target_package_name}, covering ONLY these methods:
{method_list_str}
"""
                    elif i > 0:
                        with open(test_output_file_path, 'r', encoding='utf-8') as f:
                            current_test_code = f.read()
                        prompt = f"""
{good_example}
{forbidden}
--- ENDPOINTS UNDER TEST ---
{endpoints_list_str}
--- END ENDPOINTS ---

You are an expert Java test developer. Here is the current test class for {target_class_name}:
--- BEGIN EXISTING TEST CLASS ---
{current_test_code}
--- END EXISTING TEST CLASS ---

STRICT INSTRUCTIONS:
- Retain all previously generated test methods and class structure.
- Add new test methods for ONLY these methods:
{method_list_str}
- Do NOT remove or modify any existing test methods unless you are fixing errors.
- The final test class must contain all previously generated test methods plus the new ones for this batch.
- Output the complete, compilable test class.

{context}
"""
                        if i == 1 and retries == 0:
                            pass  # Removed debug print of prompt
                    # Safeguard: Remove any Dependency: blocks or lines from the prompt
                    import re
                    prompt = re.sub(r'^Dependency:.*$', '', prompt, flags=re.MULTILINE)
                    prompt = re.sub(r'- public .*$', '', prompt, flags=re.MULTILINE)
                    # Remove all signatures, parameter lists, and annotation lines from main class code for controllers
                    # Remove any excessive blank lines
                    prompt = re.sub(r'\n{3,}', '\n\n', prompt)
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
                if i == 0:
                    # First batch: write the full class
                    test_output_file_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(test_output_file_path, 'w', encoding='utf-8') as f:
                        f.write(code)
                else:
                    # For subsequent batches, overwrite the test file with the new full class
                    with open(test_output_file_path, 'w', encoding='utf-8') as f:
                        f.write(code)
                # Run the test file
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
                    print(f"[SUCCESS] Batch {i+1}/{len(batches)}: No compilation or test errors.")
                    success = True
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
                    if full_compilation_error:
                        error_feedback += '\n\n--- FULL COMPILATION ERROR OUTPUT ---\n' + full_compilation_error + '\n--- END FULL COMPILATION ERROR OUTPUT ---\n'
                    retries += 1
            # Always update final_code with the latest test file after each batch
            with open(test_output_file_path, 'r', encoding='utf-8') as f:
                final_code = f.read()
        # After all batches:
        return final_code

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

    def extract_test_plan_methods(self, test_plan: list) -> set:
        """
        Extracts method names from the LLM-generated test plan.
        """
        return set(tc['method_name'] for tc in test_plan if 'method_name' in tc)

    def assemble_test_class(self, package_name: str, imports: List[str], class_name: str, test_methods: List[str], class_code: str = "", test_type: str = "service") -> str:
        """
        Assemble the final test class file with package, imports, annotations, class definition, and all test methods.
        - For 'controller', use @WebMvcTest, @MockBean, @Autowired MockMvc, and organize with @Nested classes if possible.
        - For 'service'/'repository', use @ExtendWith(MockitoExtension.class), @Mock, @InjectMocks.
        - Only add relevant imports/annotations for the test type.
        - Post-process to check for mixed usage (e.g., both @MockBean and @InjectMocks) and reject/fix if found.
        - All generated test methods are grouped by collecting them into a list, deduplicating by method name, and joining them in the class body.
        """
        # Auto-detect used class names and add missing imports
        used_class_names = self._extract_used_class_names(test_methods)
        auto_imports = self._map_class_names_to_imports(used_class_names, class_code, extra_known_imports=imports)

        package_stmt = f"package {package_name};\n" if package_name else ""
        all_imports = []
        for imp in imports:
            if imp.startswith('import '):
                all_imports.append(imp)
            else:
                all_imports.append(f"import {imp};")
        for imp in auto_imports:
            if imp.startswith('import '):
                all_imports.append(imp)
            else:
                all_imports.append(f"import {imp};")
        unique_imports = sorted(set(all_imports))
        static_imports = [imp for imp in unique_imports if "static" in imp]
        regular_imports = [imp for imp in unique_imports if "static" not in imp]
        import_section = "\n".join(regular_imports)
        static_import_section = "\n".join(static_imports)

        # Controller vs Service/Repository logic
        if test_type == "controller":
            class_anno = f"@WebMvcTest({class_name}.class)"
            class_def = f"class {class_name}Test {{"
            # Add MockMvc and @MockBean fields if not present in methods
            field_section = "    @Autowired\n    private MockMvc mockMvc;\n"
            # Optionally, parse dependencies to add @MockBean fields
            # (This could be improved by using controller analysis output)
            # Group methods by @Nested if possible (not enforced here)
        else:
            class_anno = "@ExtendWith(MockitoExtension.class)"
            class_def = f"class {class_name}Test {{"
            field_section = ""

        # Group and deduplicate test methods
        methods_section = "\n\n".join(m for m in test_methods if m.strip())

        # Post-processing: Check for mixed usage
        if test_type == "controller":
            if "@InjectMocks" in methods_section or "@Mock " in methods_section:
                # Remove or comment out these lines
                methods_section = methods_section.replace("@InjectMocks", "// [REMOVED: @InjectMocks not allowed in controller tests]")
                methods_section = methods_section.replace("@Mock ", "// [REMOVED: @Mock not allowed in controller tests]")
        else:
            if "@WebMvcTest" in methods_section or "MockMvc" in methods_section or "@MockBean" in methods_section:
                # Remove or comment out these lines
                methods_section = methods_section.replace("@WebMvcTest", "// [REMOVED: @WebMvcTest not allowed in service/repo tests]")
                methods_section = methods_section.replace("MockMvc", "// [REMOVED: MockMvc not allowed in service/repo tests]")
                methods_section = methods_section.replace("@MockBean", "// [REMOVED: @MockBean not allowed in service/repo tests]")

        # Final assembly
        result = f"{package_stmt}\n{import_section}\n"
        if static_import_section:
            result += f"{static_import_section}\n"
        result += f"\n{class_anno}\n{class_def}\n\n{field_section}{methods_section}\n\n}}"
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

    def generate_simple_test_case(self, target_class_name: str, target_package_name: str, class_code: str, custom_imports: List[str]) -> str:
        """
        Simplified test generation that avoids the complex multi-step process.
        Generates tests in a single pass with focused prompts.
        """
        # Extract public methods using static analysis
        public_methods = self.extract_public_methods(class_code)
        if not public_methods:
            return "// No public methods found to test"
        
        method_list = "\n".join(f"- {method}" for method in sorted(public_methods))
        
        # Simplified, focused prompt
        prompt = f"""
You are an expert Java test developer. Generate a complete JUnit 5 + Mockito test class for the following Java class.

CLASS UNDER TEST:
{class_code}

REQUIREMENTS:
- Test class name: {target_class_name}Test
- Package: {target_package_name}
- Test ALL these public methods: {method_list}
- Use @ExtendWith(MockitoExtension.class)
- Mock only dependencies that are actually used in each test method
- Include proper imports
- Test both success and failure scenarios
- Verify logging behavior if present (use ArgumentCaptor for log messages)
- Test edge cases and exception handling
- Use descriptive test method names (e.g., testMethodName_WhenCondition_ShouldReturnExpected)
- Follow AAA pattern (Arrange, Act, Assert)
- Use meaningful assertions with clear error messages

IMPORTS TO USE:
{chr(10).join(custom_imports)}

IMPORTANT:
- Do NOT invent methods, fields, or classes not present in the source code
- Do NOT mock simple POJOs or value objects
- Only mock dependencies that are actually injected or used
- If unsure about a dependency, prefer not to mock it
- Ensure all tests compile and run without errors

Output ONLY the complete test class code, no explanations or markdown.
"""
        
        result = self.llm.invoke(prompt)
        if hasattr(result, "content"):
            result = result.content
        
        # Clean up the output
        code = result.strip()
        code = re.sub(r'^```[a-zA-Z]*\n', '', code)
        code = re.sub(r'^```', '', code)
        code = re.sub(r'```$', '', code)
        
        return code.strip()

    def debug_minimal_controller_prompt(self, target_class_name, target_package_name, method_code, custom_imports):
        """
        Debug utility: Send a minimal, focused MockMvc controller prompt to the LLM for a single method.
        Prints the prompt and the LLM's output.
        """
        from llm import test_prompt_templates
        good_example = '''
// GOOD EXAMPLE (DO THIS):
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.autoconfigure.web.servlet.WebMvcTest;
import org.springframework.boot.test.mock.mockito.MockBean;
import org.springframework.test.web.servlet.MockMvc;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.*;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.*;

@WebMvcTest(UserController.class)
class UserControllerTest {
    @Autowired MockMvc mockMvc;
    @MockBean UserService userService;
    @Test void shouldReturnUser_whenUserExists() throws Exception {
        when(userService.findByName("alice")).thenReturn(new User("alice"));
        mockMvc.perform(get("/users/alice"))
            .andExpect(status().isOk())
            .andExpect(jsonPath("$.name").value("alice"));
    }
}
'''
        strict_instructions = """
STRICT: You MUST use MockMvc, @WebMvcTest, @MockBean, and @Autowired MockMvc for all tests. Do NOT use @InjectMocks or @Mock for controllers.
- Output ONLY compilable Java code, no explanations or markdown.
- Use @WebMvcTest({target_class_name}.class) for the test class annotation.
- Use @MockBean for all dependencies (not @Mock).
- Inject MockMvc using @Autowired.
- For this method, generate a test method that:
    - Uses MockMvc to perform HTTP requests (e.g., mockMvc.perform(...)).
    - Asserts HTTP status, response body, and side effects using andExpect and other assertions.
    - Mocks service/repository dependencies using @MockBean and Mockito (when, doReturn, etc.).
- Never instantiate dependencies manually—always use dependency injection and mocking.
- Do NOT hallucinate methods, fields, helpers, or imports—use only what is present in the provided context and imports.
- Do NOT invent or use any methods, fields, classes, or helpers not present in the provided code.
- If you are unsure, leave it out or add a comment.
- If you hallucinate, you will be penalized and re-prompted.
- Always include {custom_imports} in the import section.
- The output must be a single, compilable Java test class, similar in style and completeness to the GOOD EXAMPLE above.
"""
        prompt = f"{good_example}\n{strict_instructions}\n--- BEGIN METHOD UNDER TEST ---\n{method_code}\n--- END METHOD UNDER TEST ---"
        print("\n[DEBUG] Minimal MockMvc Controller Prompt:\n" + prompt + "\n--- END PROMPT ---\n")
        # Use the LLM directly
        from langchain_google_genai import ChatGoogleGenerativeAI
        import os
        GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.1)
        result = llm.invoke(prompt)
        if hasattr(result, "content"):
            result = result.content
        print("\n[DEBUG] LLM Output:\n" + result + "\n--- END LLM OUTPUT ---\n")

    def _sanitize_llm_test_methods(self, llm_output: str) -> str:
        """
        Sanitize LLM output for batch mode: remove imports, package, class headers, and extra braces.
        Only keep @Test methods and their bodies.
        """
        lines = llm_output.splitlines()
        keep = []
        inside_method = False
        brace_count = 0
        for line in lines:
            stripped = line.strip()
            # Skip package/import/class/interface/enum lines and most annotations except @Test
            if (stripped.startswith('package ') or
                stripped.startswith('import ') or
                (stripped.startswith('class ') and 'Test' in stripped) or
                stripped.startswith('@ExtendWith') or
                stripped.startswith('@RunWith') or
                stripped.startswith('@InjectMocks') or
                stripped.startswith('@Mock') or
                stripped.startswith('@WebMvcTest') or
                stripped.startswith('@SpringBootTest') or
                stripped.startswith('@BeforeEach') or
                stripped.startswith('@AfterEach') or
                stripped.startswith('@Autowired') or
                stripped.startswith('@MockBean')):
                continue
            # Only keep @Test annotation and method bodies
            if stripped.startswith('@Test'):
                keep.append(line)
                inside_method = True
                brace_count = 0
                continue
            if inside_method:
                keep.append(line)
                brace_count += line.count('{') - line.count('}')
                # If method is closed, stop
                if brace_count <= 0 and ('}' in line):
                    inside_method = False
        # Fallback: if nothing kept, just return the original (to avoid empty batch)
        if not keep:
            # As a fallback, remove imports, package, and class lines, and braces not part of methods
            filtered = [l for l in lines if not (l.strip().startswith('package ') or l.strip().startswith('import ') or l.strip().startswith('class '))]
            return '\n'.join(filtered)
        return '\n'.join(keep)

    def _extract_field_declarations(self, llm_output: str) -> str:
        """
        Extract field declarations (with @Mock, @InjectMocks, @MockBean, etc.) from LLM output.
        Returns a string with the field declarations to be inserted after the class header.
        """
        lines = llm_output.splitlines()
        keep = []
        inside_field = False
        for line in lines:
            stripped = line.strip()
            # Look for field-level annotations and declarations
            if (stripped.startswith('@Mock') or
                stripped.startswith('@InjectMocks') or
                stripped.startswith('@MockBean') or
                stripped.startswith('@Autowired')):
                keep.append(line)
                inside_field = True
                continue
            # If inside a field declaration, keep the next line (the field itself)
            if inside_field:
                if (stripped and not stripped.startswith('@')):
                    keep.append(line)
                    inside_field = False
        return '\n'.join(keep)

    def _extract_import_statements(self, llm_output: str) -> list:
        """
        Extract import statements from LLM output.
        Returns a list of unique import statements.
        """
        lines = llm_output.splitlines()
        imports = [line.strip() for line in lines if line.strip().startswith('import ')]
        return list(dict.fromkeys(imports))  # Deduplicate, preserve order

    def _update_imports_in_test_file(self, test_output_file_path: Path, new_imports: list, package_stmt: str):
        """
        Update the import section of the test file by adding any new imports (deduplicated),
        keeping them after the package statement and before the class annotation/header.
        """
        with open(test_output_file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        # Find package line
        package_idx = 0
        for i, line in enumerate(lines):
            if line.strip().startswith('package '):
                package_idx = i
                break
        # Find where imports end (first non-import, non-empty after package)
        import_start = package_idx + 1
        import_end = import_start
        while import_end < len(lines) and (lines[import_end].strip().startswith('import ') or lines[import_end].strip() == ''):
            import_end += 1
        # Collect existing imports
        existing_imports = set()
        for i in range(import_start, import_end):
            if lines[i].strip().startswith('import '):
                existing_imports.add(lines[i].strip())
        # Add new imports, deduplicated
        all_imports = list(existing_imports)
        for imp in new_imports:
            if imp not in existing_imports:
                all_imports.append(imp)
        # Rebuild file
        new_lines = []
        new_lines.extend(lines[:import_start])
        if all_imports:
            new_lines.extend([imp + '\n' for imp in all_imports])
            new_lines.append('\n')
        new_lines.extend(lines[import_end:])
        with open(test_output_file_path, 'w', encoding='utf-8') as f:
            f.writelines(new_lines)

    def _extract_field_declarations_list(self, llm_output: str) -> list:
        """
        Extract field declarations (with @Mock, @InjectMocks, @MockBean, etc.) from LLM output as a list of lines.
        """
        lines = llm_output.splitlines()
        keep = []
        inside_field = False
        for line in lines:
            stripped = line.strip()
            if (stripped.startswith('@Mock') or
                stripped.startswith('@InjectMocks') or
                stripped.startswith('@MockBean') or
                stripped.startswith('@Autowired')):
                keep.append(line)
                inside_field = True
                continue
            if inside_field:
                if (stripped and not stripped.startswith('@')):
                    keep.append(line)
                    inside_field = False
        return keep

    def _update_fields_in_test_file(self, test_output_file_path: Path, new_fields: list):
        """
        Update the field section of the test file by adding any new field declarations (deduplicated),
        after the class header and before the first test method.
        """
        with open(test_output_file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        # Find the class header (first line ending with '{')
        class_header_idx = 0
        for i, line in enumerate(lines):
            if line.strip().endswith('{'):
                class_header_idx = i
                break
        # Find where the first @Test method starts (or end of field section)
        field_end = class_header_idx + 1
        while field_end < len(lines) and not lines[field_end].strip().startswith('@Test'):
            field_end += 1
        # Collect existing fields
        existing_fields = set()
        for i in range(class_header_idx + 1, field_end):
            l = lines[i].strip()
            if l:
                existing_fields.add(l)
        # Add new fields, deduplicated
        all_fields = list(existing_fields)
        for f in new_fields:
            if f.strip() and f.strip() not in existing_fields:
                all_fields.append(f.strip())
        # Rebuild file
        new_lines = []
        new_lines.extend(lines[:class_header_idx + 1])
        if all_fields:
            for f in all_fields:
                new_lines.append('    ' + f + '\n' if not f.startswith('    ') else f + '\n')
            new_lines.append('\n')
        new_lines.extend(lines[field_end:])
        with open(test_output_file_path, 'w', encoding='utf-8') as f:
            f.writelines(new_lines)

    def _extract_test_method_names(self, code: str) -> set:
        """
        Extract all @Test method names from the given Java code.
        """
        method_names = set()
        # Look for @Test ... void methodName(...)
        pattern = re.compile(r'@Test\s+(?:public\s+|private\s+|protected\s+)?void\s+(\w+)\s*\(', re.MULTILINE)
        for match in pattern.finditer(code):
            method_names.add(match.group(1))
        # Also handle @Test on previous line
        pattern2 = re.compile(r'@Test\s*\n\s*(?:public\s+|private\s+|protected\s+)?void\s+(\w+)\s*\(', re.MULTILINE)
        for match in pattern2.finditer(code):
            method_names.add(match.group(1))
        return method_names

    def _get_full_code_from_chromadb(self, filename: str) -> str:
        """
        Retrieve and concatenate all ChromaDB chunks for a given filename.
        """
        docs = self.vectorstore.similarity_search(f"Full code for {filename}", k=100, filter={"filename": filename})
        # Sort by chunk index if available
        try:
            docs = sorted(docs, key=lambda d: d.metadata.get('chunk_index', 0))
        except Exception:
            pass
        return '\n'.join([doc.page_content for doc in docs])

    def _build_full_context_from_chromadb(self, main_class_filename: str, dependency_filenames: list) -> str:
        """
        Build the full prompt context: main class code and all dependencies, from ChromaDB.
        """
        context = f"--- BEGIN MAIN CLASS UNDER TEST ---\n"
        main_code = self._get_full_code_from_chromadb(main_class_filename)
        context += main_code + "\n--- END MAIN CLASS UNDER TEST ---\n\n"
        for dep in dependency_filenames:
            if dep == main_class_filename:
                continue
            dep_code = self._get_full_code_from_chromadb(dep)
            context += f"--- BEGIN DEPENDENCY: {dep} ---\n{dep_code}\n--- END DEPENDENCY: {dep} ---\n\n"
        return context

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
def extract_class_signatures(java_code: str) -> str:
    """
    Given full Java class code, return a string with only the class declaration and public/protected method/field/constructor signatures.
    """
    try:
        tree = javalang.parse.parse(java_code)
    except Exception as e:
        print(f"[extract_class_signatures] javalang parse error: {e}")
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

def extract_minimal_class_for_methods(java_code: str, method_names: list) -> str:
    """
    Given full Java class code and a list of method names, return a minimal class code string containing:
    - The class declaration and all class-level annotations
    - All fields
    - The specified methods (with annotations, signatures, and bodies)
    - Any private/helper methods directly called by the batch methods (recursively)
    - Any inner classes referenced by included methods
    - PATCH: For each method, include all method-level annotations (lines starting with '@' immediately above the method signature)
    """
    import re
    try:
        tree = javalang.parse.parse(java_code)
    except Exception as e:
        print(f"[extract_minimal_class_for_methods] javalang parse error: {e}")
        # Instead of fallback, raise an error to avoid broken minimal class code
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
    if hasattr(main_class, 'annotations') and main_class.annotations:
        for anno in main_class.annotations:
            # Try to extract annotation line from code
            if hasattr(anno, 'position') and anno.position:
                start = anno.position[0] - 1
                end = anno.position[0]
                class_annos.append(java_code.splitlines()[start])
            else:
                class_annos.append(f"@{anno.name}")
    # Class declaration line
    class_decl_line = None
    for i, line in enumerate(java_code.splitlines()):
        if re.match(r'.*class\s+' + re.escape(main_class.name) + r'\b', line):
            class_decl_line = line
            break
    if not class_decl_line:
        class_decl_line = f"public class {main_class.name} {{"
    # Collect all fields (by line numbers)
    field_lines = []
    for field in main_class.fields:
        if hasattr(field, 'position') and field.position:
            start = field.position[0] - 1
            end = field.position[1] if hasattr(field, 'position') and field.position and len(field.position) > 1 else start + 1
            field_lines.extend(java_code.splitlines()[start:end])
        else:
            mods = ' '.join(field.modifiers)
            typ = field.type.name if hasattr(field.type, 'name') else str(field.type)
            decl = f"    {mods} {typ} " + ', '.join(d.name for d in field.declarators) + ';'
            field_lines.append(decl)
    # Map method name to method node
    method_map = {m.name: m for m in main_class.methods}
    # Helper: get method source by line numbers, including annotations
    def get_method_src_with_annotations(method):
        if hasattr(method, 'position') and method.position:
            start = method.position[0] - 1
            # Look upwards for annotations
            lines = java_code.splitlines()
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
            # Find referenced inner classes in this method
            # Simple heuristic: look for capitalized identifiers that match inner class names
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
            # Try to extract the code block for this inner class from the original code
            if hasattr(ic, 'position') and ic.position:
                start = ic.position[0] - 1
                # Find the end by matching braces
                lines = java_code.splitlines()[start:]
                brace_count = 0
                class_lines = []
                started = False
                for l in lines:
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
                # Fallback: just output class header
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
            # --- Generate the test case with feedback loop ---
            # Use batch mode for large classes
            public_methods_for_batch = None
            try:
                with open(java_file_path_abs, 'r', encoding='utf-8') as f:
                    class_code_for_batch = f.read()
                public_methods_for_batch = list(test_generator.extract_public_methods(class_code_for_batch))
            except Exception as e:
                print(f"[WARNING] Could not extract public methods for batch mode decision: {e}")
            # --- NEW: Always print method count and mode ---
            if public_methods_for_batch is not None:
                print(f"[DEBUG] Found {len(public_methods_for_batch)} public methods for {target_class_name}")
                print(f"[DEBUG] Extracted public methods: {public_methods_for_batch}")
                if len(public_methods_for_batch) > 8:
                    print(f"[DEBUG] Using batch mode for test generation.")
                else:
                    print(f"[DEBUG] Using single-shot mode for test generation.")
            else:
                print(f"[DEBUG] Could not determine public method count for {target_class_name}")
            if public_methods_for_batch and len(public_methods_for_batch) > 8:
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
            else:
                generated_test_code = test_generator.generate_test_case(
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
        print("Verify your ChromaDB setup and network connection for Google Generative AI API, and that file paths are correct.")


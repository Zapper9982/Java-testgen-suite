import sys
from pathlib import Path
import os
from typing import List, Dict, Any, Union

# Define the root of the 'testgen-automation' project (where this script lives)
# This is used for correctly importing internal modules like 'chroma_db'.
TESTGEN_AUTOMATION_ROOT = Path(__file__).parent.parent.parent

# Define the root of your Spring Boot project.
# This is the base path for where the generated test files will be saved.
SPRING_BOOT_PROJECT_ROOT = Path("/Users/tanmay/Desktop/AMRIT/BeneficiaryID-Generation-API")

# Add the 'src' directory of the testgen-automation project to sys.path.
# This ensures that imports like 'from chroma_db.chroma_client' work correctly.
TESTGEN_AUTOMATION_SRC_DIR = TESTGEN_AUTOMATION_ROOT / "src"
if str(TESTGEN_AUTOMATION_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(TESTGEN_AUTOMATION_SRC_DIR))
    print(f"Added {TESTGEN_AUTOMATION_SRC_DIR} to sys.path for internal module imports.")


# Import ChromaDB client functions (from your existing chroma_db module)
from chroma_db.chroma_client import get_chroma_client, get_or_create_collection

# LangChain Imports - Updated to langchain_community
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceBgeEmbeddings


# --- Import for Groq ---
from langchain_groq import ChatGroq
# --- END NEW ---

from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import torch # Still needed for embedding model device configuration

# --- Configuration ---
# NOW using Groq API configuration

# Groq API Configuration
GROQ_API_KEY = os.getenv("GROQ_API_KEY") # Recommended: Get from environment variable
if not GROQ_API_KEY:
    print("WARNING: GROQ_API_KEY environment variable not set. Please set it for Groq API calls.")
    print("Example: export GROQ_API_KEY='sk_your_groq_api_key_here'")
    
# --- CHANGE HERE: Using a Llama 3 model from Groq ---
LLM_MODEL_NAME_GROQ = "llama3-8b-8192" # Or "mixtral-8x7b-32768" for a larger model

# Embedding Model (still BGE for local embeddings in ChromaDB)
# This MUST match the model used when you ran src/embedder/embed_chunks.py
EMBEDDING_MODEL_NAME_BGE = "BAAI/bge-small-en-v1.5"
DEVICE_FOR_EMBEDDINGS = "cuda" if torch.cuda.is_available() else "cpu" 


# --- Helper Function to Derive Test File Paths ---
def get_test_paths(original_filepath_txt: str, project_root: Path):
    """
    Derives the original Java filename, test class name, and full output path
    for the generated JUnit test file.

    Args:
        original_filepath_txt (str): The 'filepath_txt' metadata from a chunk,
                                     e.g., "processed_output/com/pkg/MyService.txt".
        project_root (Path): The root directory of the Spring Boot project.

    Returns:
        Dict: A dictionary containing:
              - original_java_filename (str): e.g., "MyService.java"
              - test_output_dir (Path): The full directory path for the test file.
              - test_output_file_path (Path): The full path to the test file.
    """
    # 1. Get the path relative to processed_output/
    relative_path_from_processed_output = Path(original_filepath_txt).relative_to("processed_output")

    # 2. Get the package path (directory part)
    package_path = relative_path_from_processed_output.parent

    # 3. Get the original filename base (without .txt or .java)
    original_filename_base = relative_path_from_processed_output.stem 

    # 4. Construct the original .java filename for filtering
    original_java_filename = f"{original_filename_base}.java" 

    # 5. Construct the test class name (e.g., "MyServiceTest.java")
    test_class_name = f"{original_filename_base}Test.java"

    # 6. Construct the full output directory for the test file
    test_output_dir = project_root / "src" / "test" / "java" / package_path

    # 7. Construct the full output file path
    test_output_file_path = test_output_dir / test_class_name

    return {
        "original_java_filename": original_java_filename,
        "test_output_dir": test_output_dir,
        "test_output_file_path": test_output_file_path
    }


class TestCaseGenerator:
    """
    Generates JUnit 5 test cases using a LangChain RetrievalQA chain
    with a ChromaDB retriever and a Groq LLM.
    """
    def __init__(self, collection_name: str = "code_chunks_collection"):
        print("Initializing TestCaseGenerator with LangChain components (Groq LLM)...")
        # 1. Initialize Embedding Model (for LangChain's Chroma client)
        print(f"Loading embedding model: {EMBEDDING_MODEL_NAME_BGE} on {DEVICE_FOR_EMBEDDINGS}...")
        self.embeddings = HuggingFaceBgeEmbeddings(
            model_name=EMBEDDING_MODEL_NAME_BGE,
            model_kwargs={'device': DEVICE_FOR_EMBEDDINGS},
            encode_kwargs={'normalize_embeddings': True}
        )
        print("Embedding model loaded.")

        # 2. Instantiate ChromaDB as LangChain Vectorstore
        print(f"Connecting to ChromaDB collection: {collection_name}...")
        self.chroma_client = get_chroma_client()
        self.vectorstore = Chroma(
            client=self.chroma_client,
            collection_name=collection_name,
            embedding_function=self.embeddings # Pass the embedding function
        )
        # Initialize retriever without a specific filter initially
        self.retriever = self.vectorstore.as_retriever(
            search_type="mmr", 
            search_kwargs={"k": 6}, # No "filter" here yet
        )
        print("ChromaDB retriever instantiated (without initial filter).")

        # 3. Instantiate the LLM (Groq)
        self.llm = self._instantiate_llm()
        print("Groq LLM instantiated for LangChain.")

        # 4. Define the Prompt Template for RetrievalQA Chain
        template = """
As an expert Java developer and Spring Boot testing specialist, your task is to generate a comprehensive JUnit 5 test case.
Include necessary imports, annotations, and test methods (e.g., @BeforeEach, @Test).
Provide ONLY the Java code block. Do NOT include any conversational text, explanations, or extraneous characters outside the code block.

Here is the relevant code context from the project, retrieved from the vector database:

```java
{context}
// Begin generated test code
"""
        QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

        # 5. Instantiate the RetrievalQA Chain
        print("Instantiating LangChain RetrievalQA Chain...")
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.retriever,
            return_source_documents=True, # Ensure this is True to get source_documents
            chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
        )
        print("RetrievalQA Chain initialized.")

    def _instantiate_llm(self) -> ChatGroq:
        """Instantiates the Groq LLM for LangChain."""
        if not GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY environment variable is not set. Cannot initialize Groq LLM.")
        
        print(f"Using Groq LLM: {LLM_MODEL_NAME_GROQ}...")
        os.environ["GROQ_API_KEY"] = GROQ_API_KEY 
        return ChatGroq(model_name=LLM_MODEL_NAME_GROQ, temperature=0.7)

    def _update_retriever_filter(self, filename: str):
        """Updates the retriever's filter to target a specific filename."""
        self.retriever = self.vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": 6,
                "filter": {"filename": filename}
            },
        )
        # Update the QA chain with the new retriever
        self.qa_chain.retriever = self.retriever
        print(f"Retriever filter updated to target filename: '{filename}'")


    def generate_test_case(self, target_functionality_description: str) -> str:
        """
        Generates a JUnit 5 test case by querying the RetrievalQA chain.

        Args:
            target_functionality_description: A description of the functionality to test.

        Returns:
            The generated test case code as a string.
        """
        print(f"\nGenerating test case for: '{target_functionality_description}' with LangChain...")

        try:
            result = self.qa_chain({"query": target_functionality_description})
            response_text = result["result"]

            if "source_documents" in result:
                print("\n--- Retrieved Source Documents (Chunks) Fed to LLM ---")
                for i, doc in enumerate(result["source_documents"]):
                    print(f"\n--- Document {i+1} ---")
                    print(f"Source File: {doc.metadata.get('filename', 'N/A')}")
                    print(f"File Path: {doc.metadata.get('filepath_txt', 'N/A')}")
                    print(f"Class Name: {doc.metadata.get('class_name', 'N/A')}")
                    print(f"Method Name: {doc.metadata.get('method_name', 'N/A')}")
                    print(f"Chunk Type: {doc.metadata.get('type', 'N/A')}")
                    print(f"Start Char: {doc.metadata.get('start_char', 'N/A')}")
                    print(f"End Char: {doc.metadata.get('end_char', 'N/A')}")
                    print("Content:")
                    print(doc.page_content)
                    print("------------------------")
                print("--- End Retrieved Source Documents ---")

            generated_code = response_text.strip()

            if "```java" in generated_code:
                start_marker = "```java"
                end_marker = "```"
                start_index = generated_code.find(start_marker)
                
                if start_index != -1:
                    code_start = start_index + len(start_marker)
                    code_end = generated_code.find(end_marker, code_start)
                    if code_end != -1:
                        generated_code = generated_code[code_start:code_end].strip()
                        if generated_code.startswith("// Begin generated test code"):
                             generated_code = generated_code.replace("// Begin generated test code", "", 1).strip()
                        return generated_code
            
            if "// Begin generated test code" in generated_code:
                generated_code = generated_code.split("// Begin generated test code")[-1].strip()
            
            if not generated_code or generated_code.lower().startswith("here is the"):
                return f"LLM generated an incomplete or conversational response: \n---\n{response_text}\n---"

            return generated_code

        except Exception as e:
            return f"Error during LLM generation with LangChain (Groq): {e}. " \
                   "Ensure your GROQ_API_KEY is correct and you have access to Groq API."

# --- Main execution (for testing this module directly) ---
if __name__ == "__main__":
    try:
        test_generator = TestCaseGenerator(collection_name="code_chunks_collection") 

        # Define the target files and their corresponding queries.
        # The 'original_file_metadata_path' should be the exact value of the 'filepath_txt'
        # metadata found in your chunks.json for the file you want to test.
        # This allows dynamic path derivation.
        target_files_to_test = [
            {
                "original_file_metadata_path": "processed_output/com/iemr/common/bengen/service/BengenService.txt",
                "query_description": "Write comprehensive JUnit 5 test cases for the `BengenService.java` class, including all TODOs and assertions and make the necessary inputs, also to call functions from that .java file use the 'import com.iemr.common.bengen.service.BengenService;' this import please. Focus on unit tests for each public method. and write complete testcases"
            },
            # Add other files here as needed. Example:
            # {
            #     "original_file_metadata_path": "processed_output/com/iemr/common/bengen/config/quartz/ScheduleJobServiceForBenGen.txt",
            #     "query_description": "Generate JUnit 5 tests for the ScheduleJobServiceForBenGen class, covering job execution and configuration loading."
            # },
            # {
            #     "original_file_metadata_path": "processed_output/com/iemr/common/bengen/utils/CookieUtil.txt",
            #     "query_description": "Generate JUnit 5 test cases for the CookieUtil.java utility class, focusing on its cookie parsing and JWT token extraction methods."
            # }
        ]

        for target_info in target_files_to_test:
            original_filepath_txt = target_info['original_file_metadata_path']
            query = target_info['query_description']

            # Derive all necessary path components for the current target file
            # IMPORTANT: Pass SPRING_BOOT_PROJECT_ROOT here
            paths = get_test_paths(original_filepath_txt, SPRING_BOOT_PROJECT_ROOT)
            target_filename_for_filter = paths["original_java_filename"] # e.g., "BengenService.java"
            test_output_dir = paths["test_output_dir"]
            test_output_file_path = paths["test_output_file_path"]

            print("\n" + "="*80)
            print(f"QUERY: {query}")
            print(f"TARGET SOURCE FILE (for ChromaDB filtering): '{target_filename_for_filter}'")
            print(f"EXPECTED TEST OUTPUT PATH: '{test_output_file_path}'")
            print("="*80)

            # Update the retriever's filter for the current target file
            test_generator._update_retriever_filter(target_filename_for_filter)

            # Generate the test case
            generated_test_code = test_generator.generate_test_case(query)
            
            # Ensure the output directory exists
            os.makedirs(test_output_dir, exist_ok=True)

            # Write the generated test case to the file
            try:
                with open(test_output_file_path, 'w', encoding='utf-8') as f:
                    f.write(generated_test_code)
                print(f"\n[SUCCESS] Generated test case saved to: '{test_output_file_path}'")
            except Exception as e:
                print(f"\n[ERROR] Could not save test case to '{test_output_file_path}': {e}")

            print("\n--- GENERATED TEST CASE (Printed to Console for review) ---")
            print(generated_test_code)
            print("\n" + "="*80 + "\n")
            
    except ValueError as ve:
        print(f"Configuration Error: {ve}")
        print("Please ensure GROQ_API_KEY is set and other configurations are correct.")
    except Exception as e:
        print(f"An unexpected error occurred during main execution: {e}")
        print("Verify your ChromaDB setup and network connection for Groq API.")
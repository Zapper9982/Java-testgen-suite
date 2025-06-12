import sys
from pathlib import Path
import os
from typing import List, Dict, Any, Union

TESTGEN_AUTOMATION_ROOT = Path(__file__).parent.parent.parent

# defining the root of your Spring Boot project.
SPRING_BOOT_PROJECT_ROOT = Path("/Users/tanmay/Desktop/AMRIT/BeneficiaryID-Generation-API")
TESTGEN_AUTOMATION_SRC_DIR = TESTGEN_AUTOMATION_ROOT / "src"
if str(TESTGEN_AUTOMATION_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(TESTGEN_AUTOMATION_SRC_DIR))
    print(f"Added {TESTGEN_AUTOMATION_SRC_DIR} to sys.path for internal module imports.")

from chroma_db.chroma_client import get_chroma_client, get_or_create_collection
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import torch 


# API config
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") 
if not GOOGLE_API_KEY:
    print("WARNING: GOOGLE_API_KEY environment variable not set. Please set it for Gemini API calls.")
    print("Example: export GOOGLE_API_KEY='your_google_api_key_here'")

LLM_MODEL_NAME_GEMINI = "gemini-1.5-flash" 



EMBEDDING_MODEL_NAME_BGE = "BAAI/bge-small-en-v1.5"
DEVICE_FOR_EMBEDDINGS = "cuda" if torch.cuda.is_available() else "cpu" 


#dynamic output path generator 
def get_test_paths(original_filepath_txt: str, project_root: Path):
    relative_path_from_processed_output = Path(original_filepath_txt).relative_to("processed_output")
    package_path = relative_path_from_processed_output.parent
    original_filename_base = relative_path_from_processed_output.stem 
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
    def __init__(self, collection_name: str = "code_chunks_collection"):
        print("Initializing TestCaseGenerator with LangChain components (Google Gemini LLM)...")
        # initializing embedding model 
        print(f"Loading embedding model: {EMBEDDING_MODEL_NAME_BGE} on {DEVICE_FOR_EMBEDDINGS}...")
        self.embeddings = HuggingFaceBgeEmbeddings(
            model_name=EMBEDDING_MODEL_NAME_BGE,
            model_kwargs={'device': DEVICE_FOR_EMBEDDINGS},
            encode_kwargs={'normalize_embeddings': True}
        )
        print("Embedding model loaded.")
          
        #connection to chromaDB client
        print(f"Connecting to ChromaDB collection: {collection_name}...")
        self.chroma_client = get_chroma_client()
        self.vectorstore = Chroma(
            client=self.chroma_client,
            collection_name=collection_name,
            embedding_function=self.embeddings 
        )
        #initialising langchain retriever with a temporary kwarg of 6 --TODO make tht dynamic
        self.retriever = self.vectorstore.as_retriever(
            search_type="mmr", 
            search_kwargs={"k": 6},#TODO-- make it dynamic because number of methods in springboot class might vary.
        )
        print("ChromaDB retriever instantiated (without initial filter).")

        self.llm = self._instantiate_llm()
        print("Google Gemini LLM instantiated for LangChain.")

    #prompt for writing test , but currently hardcoded will be made dynamic --TODO
        template = """
As an expert Java developer and Spring Boot testing specialist, your task is to generate a comprehensive test cases.
Include necessary imports, annotations, and test methods (e.g., @BeforeEach, @Test).
Provide ONLY the Java code block. Do NOT include any conversational text, explanations, or extraneous characters outside the code block.
"I need comprehensive test cases for the `BengenService.java` class. Please ensure tests are **deterministic** and cover all public methods. **make sure to import BengenService.java using 'import com.iemr.common.bengen.service.BengenService;' ensure that you use Assertions to validate void functions also i want 100% coverage on JaCoCo Reports so focus on that .also dont use DataTypeConverters. And use Assertions
Here is the relevant code context from the project, retrieved from the vector database:

```java
{context}
// Begin generated test code
"""
        QA_CHAIN_PROMPT = PromptTemplate.from_template(template)


        print("Instantiating LangChain RetrievalQA Chain...")
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.retriever,
            return_source_documents=True, 
            chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
        )
        print("RetrievalQA Chain initialized.")

#  Using gemini ChatGoogleGenerativeAI
    def _instantiate_llm(self) -> ChatGoogleGenerativeAI:
        if not GOOGLE_API_KEY:
            raise ValueError("GOOGLE_API_KEY environment variable is not set. Cannot initialize Gemini LLM.")
        
        print(f"Using Google Gemini LLM: {LLM_MODEL_NAME_GEMINI}...")
        return ChatGoogleGenerativeAI(model=LLM_MODEL_NAME_GEMINI, temperature=0.7)


    def _update_retriever_filter(self, filename: str):
        """Updates the retriever's filter to target a specific filename."""
        self.retriever = self.vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": 6,
                "filter": {"filename": filename}
            },
        )
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
            return f"Error during LLM generation with LangChain (Google Gemini): {e}. " \
                   "Ensure your GOOGLE_API_KEY is correct and you have access to Google Generative AI API."


if __name__ == "__main__":
    try:
        test_generator = TestCaseGenerator(collection_name="code_chunks_collection") 

        target_files_to_test = [
            {
                "original_file_metadata_path": "processed_output/com/iemr/common/bengen/service/BengenService.txt",
                "query_description": "I need comprehensive test cases for the `BengenService.java` class. Please ensure tests are **deterministic** and cover all public methods. **make sure to import BengenService.java using 'import com.iemr.common.bengen.service.BengenService;' ensure that you use Assertions to validate void functions also i want 100% coverage on JaCoCo Reports so focus on that ."

            },
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

            # retriever filter is updated 
            test_generator._update_retriever_filter(target_filename_for_filter)

            # generation of the test case by calling the function with a query 
            generated_test_code = test_generator.generate_test_case(query)
            
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
        print("Please ensure GOOGLE_API_KEY is set and other configurations are correct.")
    except Exception as e:
        print(f"An unexpected error occurred during main execution: {e}")
        print("Verify your ChromaDB setup and network connection for Google Generative AI API.")
import sys
from pathlib import Path
from typing import List, Dict, Any, Union

# Ensure src/ is on the path for project-level imports
PROJECT_ROOT = Path(__file__).parent.parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
    print(f"Added {SRC_DIR} to sys.path for module imports.")

# Import ChromaDB client functions (from your existing chroma_db module)
from chroma_db.chroma_client import get_chroma_client, get_or_create_collection

# LangChain Imports
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceBgeEmbeddings

# --- NEW: Import for Groq ---
from langchain_groq import ChatGroq
# --- END NEW ---

from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import os # For environment variables like API keys
import torch # Still needed for embedding model device configuration

# --- Configuration ---
# NOW using Groq API configuration

# Groq API Configuration
GROQ_API_KEY = os.getenv("GROQ_API_KEY") # Recommended: Get from environment variable
if not GROQ_API_KEY:
    print("WARNING: GROQ_API_KEY environment variable not set. Please set it for Groq API calls.")
    print("Example: export GROQ_API_KEY='sk_your_groq_api_key_here'")
    # For quick testing, you can uncomment the line below and hardcode, but NOT recommended for production:
    # GROQ_API_KEY = "YOUR_HARDCODED_GROQ_API_KEY_HERE"
    
# --- CHANGE HERE: Using a Llama 3 model from Groq ---
LLM_MODEL_NAME_GROQ = "llama3-8b-8192" # Or "mixtral-8x7b-32768" for a larger model

# Embedding Model (still BGE for local embeddings in ChromaDB)
# This MUST match the model used when you ran src/embedder/embed_chunks.py
EMBEDDING_MODEL_NAME_BGE = "BAAI/bge-small-en-v1.5"
DEVICE_FOR_EMBEDDINGS = "cuda" if torch.cuda.is_available() else "cpu" 


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
        self.retriever = self.vectorstore.as_retriever(
            search_type="mmr", 
            search_kwargs={"k": 5} 
        )
        print("ChromaDB retriever instantiated.")

        # 3. Instantiate the LLM (Groq)
        self.llm = self._instantiate_llm()
        print("Groq LLM instantiated for LangChain.")

        # 4. Define the Prompt Template for RetrievalQA Chain
        template = """
As an expert Java developer and Spring Boot testing specialist, your task is to generate a comprehensive JUnit 5 test case.
The test should utilize MockMvc for controller testing and Mockito for mocking service/repository dependencies.
Focus on the functionality described. Include necessary imports, annotations, and test methods (e.g., @BeforeEach, @Test).
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
        # LangChain automatically picks up GROQ_API_KEY from os.environ
        os.environ["GROQ_API_KEY"] = GROQ_API_KEY 
        # Configure the ChatGroq instance
        return ChatGroq(model_name=LLM_MODEL_NAME_GROQ, temperature=0.7)


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

            # --- START NEW LOGGING CODE ---
            if "source_documents" in result:
                print("\n--- Retrieved Source Documents (Chunks) Fed to LLM ---")
                for i, doc in enumerate(result["source_documents"]):
                    print(f"\n--- Document {i+1} ---")
                    print(f"Source: {doc.metadata.get('source', 'N/A')}")
                    print(f"Start Line: {doc.metadata.get('start_line', 'N/A')}")
                    print(f"End Line: {doc.metadata.get('end_line', 'N/A')}")
                    print("Content:")
                    print(doc.page_content)
                    print("------------------------")
                print("--- End Retrieved Source Documents ---")
            # --- END NEW LOGGING CODE ---

            # --- ORIGINAL ROBUST CODE EXTRACTION LOGIC (Keep this) ---
            generated_code = response_text.strip()

            # 1. Try to extract from Markdown code block
            if "```java" in generated_code:
                # Find the start of the first Java code block
                start_marker = "```java"
                end_marker = "```"
                start_index = generated_code.find(start_marker)
                
                if start_index != -1:
                    code_start = start_index + len(start_marker)
                    code_end = generated_code.find(end_marker, code_start)
                    if code_end != -1:
                        generated_code = generated_code[code_start:code_end].strip()
                        # If the marker was inside the markdown, remove it from the extracted code
                        if generated_code.startswith("// Begin generated test code"):
                             generated_code = generated_code.replace("// Begin generated test code", "", 1).strip()
                        return generated_code
            
            # 2. Fallback: Try to extract using the custom marker if markdown block wasn't found or parsed
            if "// Begin generated test code" in generated_code:
                generated_code = generated_code.split("// Begin generated test code")[-1].strip()
            
            # If still nothing or just conversational text
            if not generated_code or generated_code.lower().startswith("here is the"):
                return f"LLM generated an incomplete or conversational response: \n---\n{response_text}\n---"

            return generated_code

        except Exception as e:
            return f"Error during LLM generation with LangChain (Groq): {e}. " \
                   "Ensure your GROQ_API_KEY is correct and you have access to Groq API."

# --- Main execution (for testing this module directly) ---
if __name__ == "__main__":
    # --- IMPORTANT SETUP FOR TESTING ---
    # 1. Set your GROQ_API_KEY environment variable.
    #    Example (in terminal before running): export GROQ_API_KEY="sk_..."
    # 2. Ensure ChromaDB is running (if persistent) and has been populated
    #    with your code chunks by running src/embedder/embed_chunks.py at least once.
    # 3. The 'code_chunks_collection' name must match what you used in embed_chunks.py.
    # -----------------------------------
    
    try:
        test_generator = TestCaseGenerator(collection_name="code_chunks_collection") 

        test_queries = [
            "Write a JUnit 5 test for the `BengGenservice`'s encryption() method", # <--- CHANGE MADE HERE
        ]

        for query in test_queries:
            print("\n" + "="*80)
            print(f"QUERY: {query}")
            print("="*80)
            generated_test = test_generator.generate_test_case(query)
            print("\n--- GENERATED TEST CASE ---")
            print(generated_test)
            print("\n" + "="*80 + "\n")
            
    except ValueError as ve:
        print(f"Configuration Error: {ve}")
        print("Please ensure GROQ_API_KEY is set and other configurations are correct.")
    except Exception as e:
        print(f"An unexpected error occurred during main execution: {e}")
        print("Verify your ChromaDB setup and network connection for Groq API.")
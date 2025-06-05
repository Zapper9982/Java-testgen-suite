import sys
from pathlib import Path
from typing import List, Dict, Any, Union

PROJECT_ROOT = Path(__file__).parent.parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from chroma_db.chroma_client import get_chroma_client , get_or_create_collection

# LangChain Imports
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import os # For environment variables like API keys

# --- Configuration ---
# ONLY OpenAI API configuration is active now

# OpenAI API Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") # Recommended: Get from environment variable
if not OPENAI_API_KEY:
    print("WARNING: OPENAI_API_KEY environment variable not set. Please set it for OpenAI API calls.")
    print("Example: export OPENAI_API_KEY='your_openai_api_key_here'")
    # For quick testing, you can uncomment the line below and hardcode, but NOT recommended for production:
    # OPENAI_API_KEY = "YOUR_HARDCODED_OPENAI_API_KEY_HERE"
    
LLM_MODEL_NAME_OPENAI = "gpt-3.5-turbo" # Or "gpt-4" for higher quality but higher cost

# Embedding Model (still BGE for local embeddings in ChromaDB)
# This MUST match the model used when you ran src/embedder/embed_chunks.py
EMBEDDING_MODEL_NAME_BGE = "BAAI/bge-small-en-v1.5"
# Device for the embedding model if it needs to load locally for LangChain's Chroma client
# (even if LLM is OpenAI, the embedding function for Chroma needs a device)
import torch
DEVICE_FOR_EMBEDDINGS = "cuda" if torch.cuda.is_available() else "cpu" 


class TestCaseGenerator:
    """
    Generates JUnit 5 test cases using a LangChain RetrievalQA chain
    with a ChromaDB retriever and an OpenAI LLM.
    """
    def __init__(self, collection_name: str = "code_chunks_collection"):
        print("Initializing TestCaseGenerator with LangChain components (OpenAI LLM)...")

        # 1. Initialize Embedding Model (for LangChain's Chroma client)
        # This must be the SAME model used during your `embed_chunks.py` process.
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
        # Configure the retriever to search for the top 5 most relevant chunks
        self.retriever = self.vectorstore.as_retriever(
            search_type="mmr", # "mmr" for Max Marginal Relevance (diversity + relevance)
                               # "similarity" or "similarity_score_threshold" are other options
            search_kwargs={"k": 5} # Number of chunks to retrieve
        )
        print("ChromaDB retriever instantiated.")

        # 3. Instantiate the LLM (OpenAI)
        self.llm = self._instantiate_llm()
        print("OpenAI LLM instantiated for LangChain.")

        # 4. Define the Prompt Template for RetrievalQA Chain
        # This prompt is designed to instruct the LLM on how to use the context.
        # It takes 'context' (retrieved chunks) and 'question' (your test query).
        template = """
As an expert Java developer and Spring Boot testing specialist, your task is to generate a comprehensive JUnit 5 test case.
The test should utilize MockMvc for controller testing and Mockito for mocking service/repository dependencies.
Focus on the functionality described. Include necessary imports, annotations, and test methods (e.g., @BeforeEach, @Test).
Provide only the Java code block.

Here is the relevant code context from the project, retrieved from the vector database:

```java
{context}
// Begin generated test code
"""
        QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

        # 5. Instantiate the RetrievalQA Chain
        print("Instantiating LangChain RetrievalQA Chain...")
        # The "stuff" chain type will take all retrieved documents and "stuff" them into a single prompt.
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.retriever,
            return_source_documents=True, # Set to True if you want to see which documents were retrieved
            chain_type_kwargs={"prompt": QA_CHAIN_PROMPT} # Pass our custom prompt template
        )
        print("RetrievalQA Chain initialized.")

    def _instantiate_llm(self) -> ChatOpenAI:
        """Instantiates the OpenAI LLM for LangChain."""
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY environment variable is not set. Cannot initialize OpenAI LLM.")
        
        print(f"Using OpenAI LLM: {LLM_MODEL_NAME_OPENAI}...")
        # LangChain automatically picks up OPENAI_API_KEY from os.environ
        return ChatOpenAI(model_name=LLM_MODEL_NAME_OPENAI, temperature=0.7)


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
            # The 'run' method executes the entire RAG process:
            # 1. Embeds the question.
            # 2. Retrieves relevant documents from ChromaDB using the retriever.
            # 3. Inserts retrieved documents into the prompt context.
            # 4. Sends the full prompt to the OpenAI LLM.
            # 5. Returns the LLM's generated response.
            result = self.qa_chain({"query": target_functionality_description})
            response_text = result["result"] # The generated text from the LLM

            # Post-process the response to extract just the Java code block
            # This assumes the LLM adheres to the "// Begin generated test code" marker
            generated_code = response_text.split("// Begin generated test code")[-1].strip()
            if "```" in generated_code: # Remove markdown code block fences if present
                generated_code = generated_code.split("```")[0].strip()

            return generated_code

        except Exception as e:
            # Improved error handling to give more context
            return f"Error during LLM generation with LangChain: {e}. " \
                   "Ensure your OPENAI_API_KEY is correct and you have credits."

# --- Main execution (for testing this module directly) ---
if __name__ == "__main__":

    try:
        test_generator = TestCaseGenerator(collection_name="code_chunks_collection") 

        test_queries = [
            "Write a JUnit 5 test for the ScheduleJobServiceForBenGen.java method.",
            "Generate a MockMvc test for the ScheduleJobServiceForBenGen.java  endpoint to verify successful user retrieval."
           
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
        print("Please ensure OPENAI_API_KEY is set and other configurations are correct.")
    except Exception as e:
        print(f"An unexpected error occurred during main execution: {e}")
        print("Verify your ChromaDB setup and network connection for OpenAI API.")
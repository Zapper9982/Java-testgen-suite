# Java Test Generation Suite

---

## 🚀 Project Overview

Welcome to the **Java Test Generation Suite**! This project automates the creation of comprehensive **JUnit 5** test cases for Java Spring Boot services and controllers, leveraging **Mockito** for mocking and **MockMvc** for integration testing. The goal is to reduce manual test writing, accelerate development, and improve code quality.

---

## 💡 The Vision: Automated Test Generation

Our suite follows a structured approach to dynamically generate relevant test cases. Here's a high-level look at the project's flow:

![Project Flow](https://github.com/user-attachments/assets/c51a2223-bcdc-45ea-a8fc-11449e504b86)

---

## 📦 Project Structure & Pipeline

**Main Pipeline Steps:**
1. **Preprocessing:** Extract and clean Java source/config files from your Spring Boot project.
2. **Chunking:** Split code into semantic chunks with metadata for LLM context.
3. **Embedding:** Embed chunks into a ChromaDB vector database for similarity search.
4. **Analysis:** Analyze the codebase to discover test targets (services/controllers).
5. **Test Example Indexing:** Index real test/source pairs for retrieval-augmented generation (RAG).
6. **Test Generation:** Use LLMs to generate new JUnit tests, leveraging context and examples.
7. **Test Execution:** Run generated tests and verify results.

**Key Directories:**
- `scripts/` — Pipeline scripts (preprocessing, chunking, embedding, indexing)
- `pre-processing/` — Preprocessing utilities
- `src/` — Core modules (LLM, analyzer, test runner, chroma client)
- `processed_output/`, `chunked_output/`, `analysis_results/` — Intermediate data
- `javabridge/` — Java subprocess bridge for robust Java parsing (uses JavaParser)
- `lib/` — External Java libraries (e.g., JavaParser JARs)

---

## 🗂️ Directory Tree & Explanations

```
Java-testgen-suite/
├── analysis_results/         # Stores discovered Spring Boot targets (JSON)
├── chroma_db/               # ChromaDB client and helpers
├── javabridge/              # JavaParserBridge for robust Java parsing
├── lib/                     # JavaParser JARs
├── pre-processing/          # Java code cleaning scripts
├── processed_output/        # Cleaned Java files for chunking
├── scripts/                 # Chunking, embedding, and indexing scripts
├── src/
│   ├── analyzer/            # Code analysis utilities
│   ├── llm/                 # Main LLM test generation logic
│   └── test_runner/         # Java test runner integration
├── .env                     # Environment variables
├── run.sh                   # Main pipeline entrypoint
├── README.md
└── requirements.txt
```

**Directory/Module Explanations:**
- **analysis_results/**: Output of code analysis (e.g., discovered services/controllers and their dependencies).
- **chroma_db/**: Python modules for interacting with ChromaDB, the vector database used for code and test chunk retrieval.
- **javabridge/**: Java code (and compiled classes) for robust Java parsing using JavaParser, called from Python via subprocess.
- **lib/**: Holds external Java libraries required by the Java bridge (e.g., JavaParser JAR).
- **pre-processing/**: Scripts for cleaning Java source files (removing comments, normalizing whitespace) before chunking.
- **processed_output/**: Output directory for cleaned Java files, used as input for chunking and embedding.
- **scripts/**: Utility scripts for chunking code, embedding into ChromaDB, and indexing real test/source examples.
- **src/analyzer/**: Python modules for analyzing the Java codebase to discover Spring Boot services/controllers and their dependencies.
- **src/llm/**: Main logic for LLM-powered test generation, including batching, feedback, and prompt templates.
- **src/test_runner/**: Runs generated tests using Maven/Gradle and parses the results for feedback.
- **.env**: Stores environment variables such as API keys and project paths.
- **run.sh**: Main shell script to run the full pipeline.
- **README.md**: Documentation for setup and usage.
- **requirements.txt**: Python dependencies for the project.

---

## 🛠️ Scripts & Modules Overview

### Pipeline Scripts (`scripts/`)
- **chunker.py** — Splits cleaned code into semantic chunks with metadata for LLMs.
- **embed_chunks.py** — Embeds chunks into ChromaDB for similarity search.
- **index_test_examples.py** — Indexes real test/source file pairs into ChromaDB for RAG.
- **rebuild_chroma.py** — (Empty placeholder; can be used for DB rebuild automation.)

### Preprocessing (`pre-processing/`)
- **processing.py** — Cleans and extracts relevant files from the Java codebase, removing comments and unnecessary lines. Outputs `.txt` files to `processed_output/`.

### Core Modules (`src/`)
- **analyzer/code_analyzer.py** — Discovers Spring Boot services/controllers and their dependencies.
- **analyzer/code_analysis_utils.py** — Utilities for analyzing Java files, extracting imports, and dependency resolution.
- **llm/test_case_generator.py** — Main pipeline for generating JUnit tests using LLMs and RAG. CLI entrypoint.
- **llm/test_prompt_templates.py** — Strict prompt templates and few-shot examples for LLM test generation.
- **test_runner/java_test_runner.py** — Runs generated tests using Maven/Gradle and parses results.
- **chroma_db/chroma_client.py** — Utilities for connecting to ChromaDB.
- **main.py** — (Empty placeholder; can be used for orchestration.)

---

## ⚙️ Requirements

Install dependencies with:
```bash
pip install -r requirements.txt
```

**requirements.txt:**
chromadb, pyyaml, transformers, torch, xmltodict, langchain, sentence-transformers, huggingface-hub, tiktoken, requests, python-dotenv, langchain-text-splitters, langchain-experimental, langchain-huggingface, openai, langchain-google-genai, langchain-anthropic, langchain-groq

**Other requirements:**
- Python 3.8+
- Java (for running Maven/Gradle tests)
- Maven or Gradle installed

---

## 🔧 Environment Variables & Configuration

- `SPRING_BOOT_PROJECT_PATH` — Absolute path to your Spring Boot project root (must be set before running any scripts)
- `GOOGLE_API_KEY` — (Optional, for Gemini LLM integration)
- `BUILD_TOOL` — (Optional, set to `maven` or `gradle`)

Set these in your shell or in `run.sh` before running the pipeline.

---

## 🚦 How to Use: End-to-End Pipeline

1. **Set up your environment:**
   ```bash
   export SPRING_BOOT_PROJECT_PATH="/absolute/path/to/your/spring-boot-project"
   export GOOGLE_API_KEY="your_google_api_key"  # Optional, for Gemini
   export BUILD_TOOL="maven"  # or "gradle"
   ```

2. **Preprocess the codebase:**
   ```bash
   python3 pre-processing/processing.py
   ```

3. **Chunk the processed code:**
   ```bash
   python3 scripts/chunker.py
   ```

4. **Embed chunks into ChromaDB:**
   ```bash
   python3 scripts/embed_chunks.py
   ```

5. **Analyze the codebase for test targets:**
   ```bash
   python3 src/analyzer/code_analyzer.py
   ```

6. **Index real test/source examples (for RAG):**
   ```bash
   python3 scripts/index_test_examples.py
   ```

7. **Generate JUnit tests using LLMs:**
   ```bash
   python3 src/llm/test_case_generator.py
   ```

8. **Review and run generated tests:**
   - Tests are saved in your Spring Boot project under `src/test/java/`.
   - Use Maven/Gradle to run and verify tests, or let the pipeline do it automatically.

---

## 🧩 Example: Adding a New Spring Boot Project
1. Set `SPRING_BOOT_PROJECT_PATH` to your new project.
2. Run the pipeline steps above.
3. Generated tests will appear in your project's `src/test/java/` directory.

---

## 📝 Troubleshooting & Tips
- **Missing environment variables:** Ensure `SPRING_BOOT_PROJECT_PATH` is set before running any scripts.
- **Model download issues:** The first run may take time to download HuggingFace models.
- **ChromaDB errors:** Ensure ChromaDB is installed and accessible.
- **Test failures:** Check the output logs for suggestions and error details.
- **Empty scripts:** Some scripts (e.g., `rebuild_chroma.py`, `main.py`) are placeholders for future automation.
- **JavaParser errors:** If you see parsing errors, check that your Java files are complete and valid. The pipeline prints problematic code snippets for debugging.
- **Apple Silicon Support:** Embedding runs on `mps` device if available for faster processing.

---

## 📈 Current Progress

- **Semantic Chunker Developed:** Robust chunker creates semantic code chunks with metadata for LLM context.
- **Embedder Functionality Implemented:** Embeds chunks into ChromaDB for similarity search.
- **Retrieval QA Chain Initiated:** LangChain QA retrieval chain fetches relevant documents for LLM queries.
- **Test Example Indexing:** Real test/source pairs indexed for RAG-based test generation.
- **LLM Integration:** Supports Gemini-1.5-flash and Groq Llama (llama3-8b-8192) for test generation.
- **Batch Mode:** For large classes, tests are generated in batches (e.g., 3 methods at a time) with minimal class context to avoid LLM context limits.
- **Minimal Class Extraction:** Uses a Java subprocess (`javabridge/JavaParserBridge.java`) for robust extraction of only the methods and fields needed for each batch.
- **Strict Prompting:** Prompts enforce no hallucinated methods, fields, or classes; only use code present in the provided context; proper use of JUnit 5 and Mockito; no unnecessary mocking or stubbing.
- **Feedback Loop:** If generated tests fail to compile or run, errors are parsed and used to re-prompt the LLM for corrections.
- **Full Project Test Verification:** After all tests are generated, the pipeline will run a full `mvn clean verify` or `gradle clean test` to ensure all generated tests compile and pass.

---

## 🛠️ Tech Stack

- **Python 3.8+** — Orchestrates the entire suite.
- **Java** — Used for robust Java parsing via JavaParserBridge.
- **LangChain** — Semantic chunking, QA retrieval, and LLM orchestration.
- **ChromaDB** — Vector database for code/test chunk storage and retrieval.
- **Hugging Face Transformers** — Embedding model for code chunks (e.g., BAAI/bge-small-en-v1.5).
- **Sentence Transformers** — For additional embedding support.
- **LLM Models:** Google Gemini (via LangChain), Groq Llama, OpenAI, Anthropic (pluggable).
- **javalang** — Lightweight Java AST parsing in Python.
- **JavaParser** — Via Java subprocess for signature/minimal class extraction.
- **Spring Boot** — Target Java project.
- **Maven/Gradle** — Test execution and verification.
- **dotenv** — For environment variable management.
- **torch** — For device selection (e.g., Apple Silicon MPS support for embeddings).

---

## ❓ FAQ

- **Q: Where are generated tests saved?**  
  A: In your Spring Boot project under `src/test/java/` in the correct package.

- **Q: What if a test file already exists?**  
  A: The pipeline skips generation for existing test files.

- **Q: How do I add more real test examples for RAG?**  
  A: Use `scripts/index_test_examples.py` to index more test/source pairs.

- **Q: What if I get JavaParser errors?**  
  A: Ensure your Java files are complete and valid. The pipeline prints problematic code snippets for debugging.
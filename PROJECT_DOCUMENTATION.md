
# Java Test Generation Suite: Technical Documentation

## 1. Project Overview

The **Java Test Generation Suite** is a powerful automation tool designed to accelerate the development lifecycle by automatically generating high-quality JUnit 5 test cases for Java Spring Boot applications. It leverages Large Language Models (LLMs), Retrieval-Augmented Generation (RAG), and static code analysis to produce context-aware, compiling, and effective tests for services and controllers.

The primary goal is to minimize the manual effort required for writing unit and integration tests, ensure consistent test coverage, and allow developers to focus on feature development.

---

## 2. How It Works: The Automated Pipeline

The suite operates through a sophisticated pipeline that transforms a raw Java codebase into a suite of ready-to-use tests.

![Project Flow](https://github.com/user-attachments/assets/c51a2223-bcdc-45ea-a8fc-11449e504b86)

### **Step 1: Pre-processing (`pre-processing/processing.py`)**
- **Input**: Path to the target Java Spring Boot project.
- **Process**: The script recursively scans the target project, identifies `.java` source files, and cleans them by removing comments and normalizing whitespace.
- **Output**: Cleaned versions of the Java files are stored as `.txt` files in the `processed_output/` directory, mirroring the original package structure.

### **Step 2: Code Chunking (`scripts/chunker.py`)**
- **Input**: Cleaned source files from `processed_output/`.
- **Process**: Each Java file is parsed and split into smaller, semantically meaningful chunks (e.g., method-level chunks). Metadata, such as file path and class name, is attached to each chunk.
- **Output**: A `chunks.json` file containing all the code chunks and their metadata is saved in `chunked_output/`.

### **Step 3: Vector Embedding (`scripts/embed_chunks.py`)**
- **Input**: `chunked_output/chunks.json`.
- **Process**: The text from each code chunk is converted into a numerical vector representation (embedding) using a sentence-transformer model. These embeddings are then stored in a **ChromaDB** vector database. This allows for efficient similarity searches later.
- **Output**: A populated ChromaDB instance in the `chromadata/` directory.

### **Step 4: Code Analysis (`src/analyzer/code_analyzer.py`)**
- **Input**: The path to the target Java project.
- **Process**: This module analyzes the Java codebase to identify key components. It finds all classes annotated with `@Service` and `@RestController`, determines their dependencies (e.g., other services, repositories), and resolves their import paths.
- **Output**: A `spring_boot_targets.json` file in `analysis_results/`, which lists all testable targets and their related files.

### **Step 5: Test Generation (`src/llm/test_case_generator.py`)**
- **Input**: A specific target class to test (e.g., `com.example.MyService`).
- **Process**: This is the core of the suite.
    1.  **Context Retrieval**: It fetches the source code of the target class and its dependencies. For dependencies, it often uses just the method signatures to keep the context concise.
    2.  **Batch Processing**: It identifies all public methods in the target class and groups them into smaller batches to avoid overwhelming the LLM.
    3.  **Prompt Engineering**: For each batch, it constructs a detailed prompt containing the minimal code required for the test, dependency signatures, and strict instructions for generating JUnit 5 + Mockito tests.
    4.  **LLM Invocation**: It sends the prompt to a Google Gemini model.
    5.  **Incremental Merging**:
        *   For the **first batch**, the generated test code is saved directly to the final test file (e.g., `MyServiceTest.java`).
        *   For **subsequent batches**, a second LLM call is made to intelligently merge the newly generated test methods into the existing `MyServiceTest.java` file, preserving imports and class structure.
    6.  **Error Handling & Feedback Loop**: After each batch, the generated test is immediately compiled and run using the `java_test_runner`. If there are compilation errors or test failures, the error output is fed back into the prompt for the next attempt, allowing the LLM to self-correct.
- **Output**: A complete, compiling JUnit test file in the target project's `src/test/java` directory.

### **Step 6: Test Execution (`src/test_runner/java_test_runner.py`)**
- **Input**: The path to a generated test file.
- **Process**: This module acts as a bridge to the Java build system. It dynamically constructs and executes a `mvn test` or `gradle test` command to run the specified test file.
- **Output**: It captures and parses the `stdout` and `stderr` from the build tool to determine success, failure, or compilation errors, which are then returned to the test generator.

---

## 3. Setup and Configuration Guide

Follow these steps to set up and run the Java Test Generation Suite.

### **Prerequisites**
*   **Python 3.8+**: Ensure you have a working Python installation.
*   **Java JDK 11+**: Required for the JavaParser bridge and for compiling/running tests in the target project.
*   **Apache Maven** or **Gradle**: The build tool used by your target Spring Boot project must be installed and available in your system's PATH.
*   **Git**: For cloning the repository.

### **Step 1: Clone the Repository**
```bash
git clone <your-repository-url>
cd Java-testgen-suite
```

### **Step 2: Set Up Python Environment**
Create a virtual environment and install the required Python packages.
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### **Step 3: Configure Environment Variables**
Create a `.env` file in the root of the project directory. This file is critical for configuring API keys and project paths.

```sh
# .env file

# --- Google API Key ---
# Your API key for Google Gemini
GOOGLE_API_KEY="YOUR_GOOGLE_API_KEY"

# --- Target Project Path ---
# The absolute path to the Java Spring Boot project you want to generate tests for
SPRING_BOOT_PROJECT_ROOT="/path/to/your/java/project"
```

### **Step 4: Build the Java Bridge**
The suite uses a small Java program to parse Java code robustly. You need to compile it once.
```bash
cd javabridge
javac -cp ../lib/javaparser-core-3.25.4.jar JavaParserBridge.java
cd ..
```

---

## 4. How to Run the Suite

The entire pipeline can be executed using the `run.sh` script. This script automates all the steps from pre-processing to test generation.

### **Running the Full Pipeline**
Execute the `run.sh` script from the root directory:
```bash
bash run.sh
```
The script will:
1.  Run the pre-processing, chunking, and embedding steps.
2.  Analyze the target project to find testable classes.
3.  Iterate through the identified targets and generate a test case for each one.

### **Generating a Test for a Single Class**
You can also run the test generator directly to target a specific class. This is useful for debugging or re-generating a single test file.

```bash
python3 src/llm/test_case_generator.py --target-class "com.example.your.ClassName"
```

The generated test files will be placed in the appropriate `src/test/java/...` directory within your target Spring Boot project.

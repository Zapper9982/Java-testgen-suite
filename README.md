# Java Test Generation Suite

---

## 🚀 Project Overview

Welcome to the **Java Test Generation Suite**! This project automates the creation of JUnit 5 test cases for Java Spring Boot applications. It leverages Large Language Models (LLMs) to generate tests and iteratively improves test coverage based on feedback from JaCoCo code coverage reports. The core idea is to intelligently identify and target under-tested areas of the codebase, dynamically adjusting prompts to the LLM to generate more effective tests over multiple iterations.

---

## 📈 Key Features

*   **Automated Test Generation**: Utilizes LLMs (specifically Gemini 1.5 Flash via Google API) to generate JUnit 5 test cases for Spring Boot services and controllers.
*   **Iterative Coverage Improvement**: Employs an iterative process where JaCoCo code coverage reports are analyzed after each round of test generation and execution.
*   **Targeted Prompt Engineering**: Dynamically adjusts prompts to the LLM to focus on methods and classes that have low coverage, aiming to improve specific areas.
*   **Build Tool Integration**: Supports both Maven and Gradle based Spring Boot projects for building the project and generating JaCoCo reports.
*   **Vector Database for Context**: Uses ChromaDB with Hugging Face embeddings (BAAI/bge-small-en-v1.5) to store and retrieve relevant code chunks, providing context to the LLM.
*   **Configurable Pipeline**: Key parameters like maximum iterations, target coverage percentage, and build tool can be configured via environment variables.
*   **GitHub Actions Workflow**: Includes a CI workflow (`.github/workflows/coverage_check.yml`) to automate the test generation and coverage checking process, suitable for integration into development pipelines.
*   **Modular Design**: The system is broken down into distinct Python scripts/modules for pre-processing, code analysis, chunking, embedding, LLM interaction, and test execution.

---

## 🌊 Project Flow

The system operates in an iterative loop:

1.  **Pre-processing**: Java source files are processed to remove comments, empty lines, etc.
2.  **Code Analysis**: The Spring Boot application is analyzed to identify target classes (services, controllers) for test generation.
3.  **Chunking & Embedding**: The relevant Java source code (target classes and their dependencies) is split into semantic chunks and stored in a ChromaDB vector database.
4.  **Test Case Generation**: For each target class (or focused methods within a class):
    *   Relevant code chunks are retrieved from ChromaDB.
    *   An LLM (Gemini 1.5 Flash) is prompted with the code context and specific instructions (including focusing on low-coverage methods in later iterations) to generate a JUnit 5 test class.
    *   The generated test is saved to the appropriate test directory in the target Spring Boot project.
5.  **Test Execution & Coverage Analysis**:
    *   The target Spring Boot project is built using the configured build tool (Maven/Gradle).
    *   Tests (including newly generated ones) are executed.
    *   A JaCoCo code coverage report (XML) is generated.
6.  **Coverage Evaluation & Iteration**:
    *   The JaCoCo report is parsed to determine overall line coverage and method-level coverage.
    *   If the overall coverage meets the `TARGET_COVERAGE` or `MAX_ITERATIONS` is reached, the process stops.
    *   Otherwise, methods with coverage below the target are identified.
    *   The list of these low-coverage methods is fed back into the Test Case Generation step for the next iteration, guiding the LLM to focus on these areas.

*(Note: A visual diagram illustrating this iterative flow would be beneficial here but is out of scope for this text-based update.)*

---

## 🛠️ How to Run

Follow these steps to set up and run the Java Test Generation Suite:

**1. Prerequisites:**

*   **Python**: Version 3.9 or higher.
*   **Java Development Kit (JDK)**: Version 11, 17, or as required by your Spring Boot project.
*   **Build Tool**: Apache Maven or Gradle installed and configured for your Spring Boot project.
*   **Git**: For cloning the repository.
*   **Python Dependencies**: Install using `pip install -r requirements.txt` (ensure this file is present and up-to-date in the repository).

**2. Environment Variables (Crucial):**

You **must** set the following environment variables:

*   `SPRING_BOOT_PROJECT_ROOT`: The absolute path to the root directory of your target Java Spring Boot project.
    *   Example: `export SPRING_BOOT_PROJECT_ROOT="/home/user/dev/my-spring-app"`
*   `GOOGLE_API_KEY`: Your Google API key for accessing the Gemini LLM.
    *   Example: `export GOOGLE_API_KEY="AIzaSy..."`

Optionally, you can also set these to override defaults:

*   `MAX_ITERATIONS`: Maximum number of test generation iterations. Defaults to `5` (as set in `src/main.py`).
    *   Example: `export MAX_ITERATIONS=3`
*   `TARGET_COVERAGE`: Desired overall line coverage percentage (0.0 to 1.0). Defaults to `0.9` (i.e., 90%) (as set in `src/main.py`).
    *   Example: `export TARGET_COVERAGE=0.85`
*   `BUILD_TOOL`: Specify "maven" or "gradle". Defaults to "maven" if not set (this default is handled by `src/test_runner/java_test_runner.py` and `src/main.py`).
    *   Example: `export BUILD_TOOL="gradle"`

**3. Execution:**

It is recommended to use the provided shell script to run the pipeline:

*   **Using `run.sh` (Recommended):**
    1.  Make the script executable: `chmod +x run.sh`
    2.  **Important**: Edit `run.sh` to set your `SPRING_BOOT_PROJECT_ROOT` and `GOOGLE_API_KEY` values. You can also uncomment and set optional variables.
    3.  Execute the script: `./run.sh`

    The `run.sh` script includes sanity checks for the required environment variables.

*   **Directly using `python src/main.py`:**
    1.  Ensure all required environment variables are exported in your current shell session.
    2.  Run the main script: `python3 src/main.py` (or `python src/main.py` depending on your Python installation).

**4. Output:**

*   The pipeline will log its progress to the console.
*   Generated test files will be saved directly into the `src/test/java/...` directory of your `SPRING_BOOT_PROJECT_ROOT`.
*   JaCoCo reports will be generated in the standard output directories of your build tool (e.g., `target/site/jacoco/jacoco.xml` for Maven).

---

## ⚙️ GitHub Workflow for CI

This project includes a GitHub Actions workflow defined in `.github/workflows/coverage_check.yml`. This workflow automates the execution of the test generation and coverage analysis pipeline on pushes and pull requests to the `main` or `master` branches.

Key aspects of the workflow:
*   Sets up Java and Python environments.
*   Installs Python dependencies.
*   Runs `src/main.py`.
*   Requires `GOOGLE_API_KEY` to be set as a repository secret in GitHub Actions settings.
*   The `SPRING_BOOT_PROJECT_ROOT` is assumed to be the root of the checkout repository by default but can be configured.
*   The workflow's success or failure is determined by the exit code of `src/main.py` (i.e., whether the target coverage was achieved).
*   It can optionally upload JaCoCo reports as build artifacts.

---

## 💻 Tech Stack

*   **Orchestration & Logic**: Python 3
*   **LLM Interaction**: LangChain, Google Generative AI (for Gemini 1.5 Flash)
*   **Vector Database**: ChromaDB
*   **Embeddings**: Hugging Face BAAI/bge-small-en-v1.5
*   **Java Build & Coverage**: Maven or Gradle, JaCoCo
*   **CI/CD**: GitHub Actions

---

## ☁️ Future Enhancements / TODO

*   Refine prompt engineering for even more precise test generation.
*   Optimize context retrieval from ChromaDB.
*   Allow selection of specific classes/packages to target from `code_analyzer.py` output.
*   More sophisticated error handling and recovery within the pipeline.
*   UI for easier configuration and monitoring (potentially).
*   Support for other types of tests (e.g., performance, security) if feasible.

---

*This README provides a guide to understanding, running, and contributing to the Java Test Generation Suite.*

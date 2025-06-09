# Java Test Generation Suite

---

## 🚀 Project Overview

Welcome to the **Java Test Generation Suite**! This project aims to automate the creation of comprehensive **JUnit 5** test cases for Java Spring Boot services and controllers, leveraging **Mockito** for mocking and **MockMvc** for integration testing. Our goal is to significantly reduce the manual effort involved in writing tests, thereby accelerating development cycles and improving code quality.

---

## 💡 The Vision: Automated Test Generation

Our suite follows a structured approach to dynamically generate relevant test cases. Here's a high-level look at the project's flow:

![Project Flow](https://github.com/user-attachments/assets/c51a2223-bcdc-45ea-a8fc-11449e504b86)

---

## 📈 Current Progress

I have laid a strong foundation for the core functionalities of the test generation suite:

1.  **Semantic Chunker Developed:** Successfully built a robust chunker that creates **semantic chunks** from source code. These chunks are enriched with ample metadata, crucial for the LLM's understanding and context retention.
2.  **Embedder Functionality Implemented:** Created a function to embed these semantic chunks into our vector database, enabling efficient similarity searches.
3.  **Retrieval QA Chain Initiated:** Set up the initial **LangChain QA retrieval chain**. This foundational step allows us to fetch relevant documents (currently hardcoded for proof of concept) based on queries.

---

## 🎯 Next Week's Goals

 immediate focus is on bringing dynamic intelligence and robustness to the system:

1.  **Complete Dynamic Retrieval QA Chain:** Fully implement the dynamic retrieval process to intelligently fetch context for test case generation.
2.  **Develop CodeAnalyser Function:** Build out the `CodeAnalyser` function. This crucial component will parse Java source files and leverage **JaCoCo reports** to dynamically adjust prompts, ensuring the generated test cases are highly relevant and target areas needing coverage.
3.  **Optimize LLM API Usage:** Devise strategies to run the entire chain efficiently, specifically addressing and mitigating **rate limiting** issues with the Gemini or OpenAI API keys.

---

## 🛠️ Tech Stack

This project is built using a powerful combination of modern technologies:

* **Python:** The primary programming language orchestrating the entire suite.
* **LangChain:** Utilized for advanced capabilities like semantic chunking and constructing the QA retrieval chain.
* **ChromaDB:** Our chosen Vector Database for storing and retrieving embedded code chunks efficiently.
* **Hugging Face (BAAI/bge-small-en-v1.5):** The embedding model used for converting code chunks into high-quality vector representations.
* **LLM Model - Groq Llama (llama3-8b-8192):** Currently leveraging this fast and efficient LLM for test case generation (exploring alternatives for scalability).

* GOOD NEWS -  INTEGRATION WITH Gemini-1.5-flash was sucessfull and now will be used to generate testcases . :))

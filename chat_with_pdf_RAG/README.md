
---

# Intelligent PDF Assistant

This app allows you to chat with the contents of a PDF document using Retrieval-Augmented Generation (RAG). You can upload a PDF, ask questions about its content, and receive answers based on the information within the document.

## Table of Contents

- [Dependencies](#dependencies)
- [Installation](#installation)
- [Usage](#usage)
  - [Running the Application](#running-the-application)
  - [Interacting with the App](#interacting-with-the-app)

## Dependencies

- **Streamlit:** A Python library for building web applications.
- **Langchain (various modules):** Libraries for handling documents, embeddings, vector stores, text splitting, etc.
- **Ollama Embeddings and LLMs:** Models used for generating embeddings and language-based responses.

## Installation

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/projects/chat_with_pdf_RAG.git
   cd chat_with_pdf_RAG
   ```

2. **Set Up a Virtual Environment (Recommended):**

   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install Ollama (if not already installed):**

   Install Ollama on your local machine from the [official website](https://ollama.com/). And then pull the LLM model :- llama3.2 and Embeddings model :- mxbai-embed-large:

   ```bash
   ollama pull llama3.2:latest && ollama pull mxbai-embed-large:latest
   ```

4. **Install Dependencies:**

   Ensure you have the required Python packages installed. You can do this using pip:

   ```bash
   pip install streamlit langchain_community langchain_core langchain_experimental langchain_ollama pdfplumber
   ```

   OR

   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Running the Application

To run the application, execute the following command:

```bash
streamlit run chat_with_pdf_RAG.py
```

This will start a local web server, typically running on `http://localhost:8501`, where you can interact with the app through your web browser.

### Interacting with the App

1. Launch the Streamlit app.
2. Upload your PDF file.
3. Choose a Large Language Model (LLM) provider and select a model.
4. Enter your API key if necessary.
5. Ask questions about the PDF contents in the chat interface.

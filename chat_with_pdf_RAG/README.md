# Intelligent PDF Assistant with RAG

This app allows you to chat with the contents of a PDF document using Retrieval-Augmented Generation (RAG). You can upload a PDF, ask questions about its content & receive answers based on the information within the document.

## Demo

![chat_with_pdf_using_RAG_demo](https://github.com/tdikonda/projects/blob/main/chat_with_pdf_RAG/demo/chat_with_PDF_using_RAG_demo.gif)

## Table of Contents

- [Dependencies](#dependencies)
- [Installation](#installation)
- [Usage](#usage)

## Dependencies

- **[Streamlit](https://streamlit.io/) :** A Python library for building web applications.
- **[LangChain](https://www.langchain.com/) :** Libraries for handling documents, embeddings, vector stores, text splitting etc.
- **[Ollama](https://ollama.com/) :** Ollama is used to access Large Language Models (LLMs) & Eembeddings models which are running locally.

## Installation

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/tdikonda/projects.git
   cd chat_with_pdf_using_RAG
   ```

2. **Set Up a Virtual Environment (Recommended):**

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate  # On Windows use `.venv/Scripts/activate.bat`
   ```

3. **Install Ollama (if not already installed):**

   Install Ollama on your local machine from the [official website](https://ollama.com/). Pull the LLM model for example ***llama3.2*** and embeddings model for example ***mxbai-embed-large***

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

1. **Running the Application**

   To run the application, execute the following command:

   ```bash
   streamlit run chat_with_pdf_using_RAG.py
   ```

   This will start a local web server, typically running on `http://localhost:8501`, where you can interact with the app through your web browser.

2. **Interacting with the App**

   - Select a Large Language Model (LLM) provider and select the model.
   - Enter your API key if necessary.
   - Upload your PDF file.
   - Ask questions about the PDF contents in the chat interface.

# Web Article Summarizer

This application uses [Streamlit](https://streamlit.io/), [Langchain](https://www.langchain.com/) and a locally running Large Language Model to fetch, extract, and summarize the content of web articles.

## Features

* Fetches content from a given URL.
* Extracts the main textual content from the web page using [Trafilatura](https://trafilatura.readthedocs.io/en/latest/).
* Summarizes the extracted text using a Large Language Model (LLM) served via Ollama.
* Allows selection from available Ollama models installed locally.
* Simple web interface built with Streamlit.

## Demo

![web_article_summarizer_demo](https://github.com/tdikonda/projects/blob/main/web_article_summarizer/demo/web_article_summarizer_demo.gif)

## Requirements

* Python 3.8+
* [Ollama](https://ollama.com/) installed and running locally.
* At least one LLM model pulled via Ollama (e.g. `ollama pull llama3.2` OR `ollama pull gemma3`).
* Python libraries:
  * `langchain`
  * `langchain_core`
  * `langchain_ollama`
  * `requests`
  * `streamlit`
  * `trafilatura`

## Setup

1. **Install Ollama:**
    Follow the instructions on the Ollama website to download and install it for your operating system.

2. **Run Ollama & Pull a Model:**
    Ensure the Ollama application or service is running. Then pull a model you want to use for summarization (e.g., Llama 3.2 OR Gemma3):

    ```bash
    ollama pull llama3.2 OR ollama pull gemma3
    ```

    You can pull other models as needed (e.g., `mistral`, `phi3`). The application will detect available models.

3. **Clone the Repository:**

    ```bash
    git clone https://github.com/tdikonda/projects.git
    cd web_article_summarizer
    ```

4. **Create a Virtual Environment (Recommended):**

    ```bash
    python3 -m venv .venv
    source .venv/bin/activate  # On Windows use `.venv/Scripts/activate.bat`
    ```

5. **Install Python Dependencies:**

    ```bash
    pip install streamlit langchain langchain-core langchain-community requests trafilatura
    ```

    OR

    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. **Run the Streamlit Application:**
    Make sure Ollama service is running in the background. Then run the Streamlit app from your terminal:

    ```bash
    streamlit run web_article_summarizer.py
    ```

2. **Use the Interface:**
    * The application will open in your web browser.
    * The sidebar will show available Ollama models detected on your system. Select the model you wish to use.
    * Enter the full URL (including `http://` or `https://`) of the web article you want to summarize in the input field.
    * Click the "✨ Summarize" button.
    * Wait for the process to complete (fetching, extracting and summarizing). A spinner will indicate progress.
    * The generated summary will appear below the input field.

## Notes & Limitations

* **Best Performance:** The summarizer works best with standard web articles, blog posts & informational content where the main text is clearly structured in HTML.
* **Potential Issues:**
  * **Paywalls:** Articles behind hard paywalls cannot be accessed.
  * **JavaScript-Heavy Sites:** Content heavily reliant on client-side JavaScript rendering might not be extracted correctly by Trafilatura.
  * **Dynamic Content/SPAs:** Single Page Applications (SPAs) or sites with highly dynamic structures may pose challenges for content extraction.

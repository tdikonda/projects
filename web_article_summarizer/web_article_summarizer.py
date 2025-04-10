'''
Web Article Summarizer application using Streamlit, Langchain and Ollama.
This script fetches content from a given URL, extracts the main text and then summarizes it using an LLM model served via Ollama.
'''

import logging

import requests
import streamlit as st
from langchain.chains.summarize import load_summarize_chain
from langchain_core.documents import Document
from langchain_ollama.llms import OllamaLLM
from trafilatura import baseline, fetch_url

# Setup basic logging for debugging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%m/%d/%Y %I:%M:%S %p',
    level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
OLLAMA_API_BASE = "http://localhost:11434/api"


def get_llm_models():
    """
    Retrieves a list of available LLM models from the local Ollama instance.
    """
    try:
        response = requests.get(f"{OLLAMA_API_BASE}/tags")
        if response.status_code == 200:
            models = response.json()
            return [model["name"] for model in models["models"]]
        else:
            st.error(
                f"Error getting LLM models: Status code {response.status_code}")
    except requests.exceptions.ConnectionError:
        st.error("Could not connect to Ollama. Make sure it's running.")
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")


def run_summarization(url: str, ollama_model: str):
    """Fetches, extracts, and summarizes content from a given URL.
    """
    try:
        # Fetch and Extract content using Trafilatura
        logger.info(f"Fetching URL: {url}")
        downloaded = fetch_url(url)
        if not downloaded:
            st.error(
                "Failed to download URL content. Check the URL or network connection."
            )

        logger.info("Extracting web article contents using trafilatura...")
        # Use baseline for cleaner text extraction
        _, text, _ = baseline(downloaded)
        if not text:
            st.error("Could not extract web article contents.")

        logger.info("Successfully extracted web article content.")

        # Prepare Langchain Documents
        docs = [Document(page_content=text, metadata={"source": url})]

        # Initialize LLM and Chain
        logger.info(f"Initializing Ollama LLM with model: {ollama_model}")
        llm = OllamaLLM(
            model=llm_model,
            temperature=0.2)  # Lower temperature for more factual summaries

        # Using refine chain type, suitable for longer documents
        chain = load_summarize_chain(
            llm, chain_type="refine",
            verbose=False)  # Set verbose=True in console for details

        # Invoke summarization Chain
        logger.info("Invoking summarization chain...")
        result = chain.invoke({"input_documents": docs})
        # result = chain.invoke(input=docs)
        summary = result.get('output_text', None)

        if not summary:
            st.error("Failed to generate summary.")

        logger.info("Summarization complete.")
        return summary

    except Exception as e:
        logger.error(
            f"An error occurred during the summarization process: {e}",
            exc_info=True)
        st.error(f"An unexpected error occurred: {e}")


# --- Streamlit App UI Configuration ---
st.set_page_config(
    page_title="Web Article Summarizer",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded")

with st.sidebar:
    # Model Selection in the sidebar
    with st.expander("🤖 Model Selection", expanded=True):
        # Get available LLM models
        llm_models = get_llm_models()
        if not llm_models:
            st.warning(
                "⚠️ No LLM models found. Make sure Ollama is running locally.")
        else:
            llm_model = st.selectbox(
                label="Select Ollama Model",
                options=llm_models,
                help="Select from your locally available LLM models.")

st.title("🌐 Web Article Summarizer")
st.markdown("Enter Web Article URL to get its summary")

st.markdown(
    """
    <div style="background-color: #cefaca; padding: 12px; border-radius: 5px; border-left: 5px solid #4caf50;">
        <p style="color: #333; margin: 0;">
            <b>Note:</b> The summarizer works best with news articles, blog posts & informational content.
        </p>
    </div>

    <div style="background-color: #ffe6e6; padding: 10px; border-radius: 5px; border-left: 5px solid #e30909;">
        <p style="color: #800000; margin: 0;">
            <b>Caution:</b> Articles behind paywalls, interactive content & heavy JavaScript driven websites may not work.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

input_url = st.text_input(
    label="",
    placeholder="https://example.com/article",
    label_visibility="hidden")

if st.button("✨ Summarize"):
    # Validate the input URL
    if not input_url or not input_url.lower().startswith(
        ('http://', 'https://')):
        st.error("Please enter a valid URL starting with http:// or https://")
    else:
        # Use spinner during processing
        with st.spinner("Summarizing article.... Please wait.", show_time=True):
            try:
                summary = run_summarization(input_url, llm_model)
                st.success("Summarization complete!")
                st.header("📝 Summary")
                st.markdown(
                    summary
                )  # Display the summary using markdown for better formatting
            except Exception as e:
                st.error(f"An error occurred: {e}")
                st.stop()

# App footer
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: #888; font-size: 0.8em;">
        Built with Streamlit and LangChain
    </div>
    """,
    unsafe_allow_html=True)

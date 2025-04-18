"""
This app allows you to chat with the contents of a PDF document using Retrieval-Augmented Generation (RAG).
Upload your PDF, ask questions and get answers based on the document's content.
"""

import base64
import gc
import os
import tempfile

import requests
import streamlit as st
from langchain_anthropic import ChatAnthropic
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_experimental.text_splitter import SemanticChunker
from langchain_ollama import OllamaEmbeddings
from langchain_ollama.llms import OllamaLLM
from langchain_openai.chat_models import ChatOpenAI

# MMTEB: Massive Multilingual Text Embedding Benchmark
# The MMTEB leaderboard compares text embedding models on 1000+ languages
# https://huggingface.co/spaces/mteb/leaderboard
# Initialize embeddings and vector database
EMBEDDINGS_MODEL = "mxbai-embed-large:latest"
embeddings = OllamaEmbeddings(model=EMBEDDINGS_MODEL)
vector_db = InMemoryVectorStore(embeddings)

# LLM Providers
DEFAULT_LLM_PROVIDERS = {
    "OpenAI": ["gpt-4o-mini", "gpt-4o", "o1", "o1-mini", "o3-mini"],
    "Anthropic": [
        "Claude 3.7 Sonet", "Claude 3.5 Opus", "Claude 3.5 Sonet",
        "Claude 3.5 Haiku"
    ],
    "Ollama": []  # Will be populated dynamically
}
OLLAMA_API_URL = "http://localhost:11434/api"

# Improved RAG prompt template with clear instructions and formatting
TEMPLATE = """
You are an expert research assistant tasked with reasoning and answering questions.

Context information:
{context}

Question:
{question}

Use the provided context to answer the question. If you don't know the answer, just say that "I don't know" but don't make up an answer on your own. Be clear, well-structured and factual in your responses. If appropriate, use bullet points or numbered lists for clarity.

Answer:
"""

st.set_page_config(
    page_title="Intelligent PDF Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded")


def reset_chat():
    """Reset the chat session by clearing messages and collecting garbage."""
    st.session_state.messages = []
    gc.collect()


def get_ollama_models():
    """
    Get list of available Ollama models from local instance.

    Returns:
        List[str]: Names of available Ollama models or empty list if Ollama is not running
    """
    try:
        response = requests.get(f"{OLLAMA_API_URL}/tags")
        if response.status_code == 200:
            models = response.json()
            return [model["name"] for model in models["models"]]
        else:
            st.error(
                f"Error getting Ollama models: Status code {response.status_code}"
            )
    except requests.exceptions.RequestException:
        return []  # Silent fail, UI will show appropriate message


def display_pdf(file):
    """Display the uploaded PDF file in an iframe."""
    base64_pdf = base64.b64encode(file.read()).decode("utf-8")

    # Embedding PDF in HTML
    pdf_display = f"""
        <iframe
            src="data:application/pdf;base64,{base64_pdf}"
            width="400px"
            height="100%"
            type="application/pdf"
            style="height:100vh; width:100%"
        >
        </iframe>
    """

    # Displaying File
    st.markdown(pdf_display, unsafe_allow_html=True)


def load_pdf(file_path):
    """Load a PDF file and return its content as a document."""
    try:
        document_loader = PDFPlumberLoader(file_path)
        return document_loader.load()
    except Exception as e:
        st.error(f"Error loading PDF: {e}")
        return None


def chunk_document(document):
    """
    Chunk the document into smaller parts for better processing.

    Returns:
        List[Document]: List of processed chunks
    """
    # 5 Levels Of Text Splitting
    # https://github.com/FullStackRetrieval-com/RetrievalTutorials/blob/a4570f3c4883eb9b835b0ee18990e62298f518ef/tutorials/LevelsOfTextSplitting/5_Levels_Of_Text_Splitting.ipynb
    try:
        text_splitter = SemanticChunker(embeddings=embeddings)
        return text_splitter.split_documents(document)
    except Exception as e:
        st.error(f"Error chunking document: {e}")
        return None


def index_documents(document_chunks):
    """Index the documents into a vector database."""
    try:
        vector_db.add_documents(document_chunks)
    except Exception as e:
        st.error(f"Error indexing documents: {e}")


def find_relevant_docs(query):
    """
    Find relevant documents based on the query.

    Returns:
        List[Document]: List of relevant documents
    """
    try:
        return vector_db.similarity_search(query)
    except Exception as e:
        st.error(f"Error finding relevant documents: {e}")


def select_llm(provider, llm_model):
    """Select the appropriate LLM based on the provider."""
    if provider == "OpenAI":
        return ChatOpenAI(model=llm_model, streaming=True)
    elif provider == "Anthropic":
        return ChatAnthropic(model=llm_model, streaming=True)
    else:
        return OllamaLLM(model=llm_model)


def generate_answer(user_question, context_documents, provider, llm_model):
    """
    Generate an answer based on the user's question and relevant documents.

    Returns:
        str: Generated answer
    """
    try:
        context = "\n\n".join([doc.page_content for doc in context_documents])
        prompt = ChatPromptTemplate.from_template(TEMPLATE)
        model = select_llm(provider, llm_model)
        chain = prompt | model
        return chain.invoke({"question": user_question, "context": context})
    except Exception as e:
        st.error(f"Error generating answer: {e}")


with st.sidebar:
    with st.expander("🤖 Model Selection", expanded=True):
        provider = st.radio(
            "Select LLM Provider", ["OpenAI", "Anthropic", "Ollama"],
            help="Choose which Large Language Model provider to use",
            horizontal=True)

        # Show models based on provider
        if provider == "Ollama":
            ollama_models = get_ollama_models()
            if not ollama_models:
                st.warning(
                    "⚠️ No Ollama models found. Make sure Ollama is running.")
                st.stop()
            models = ollama_models
        else:
            models = DEFAULT_LLM_PROVIDERS[provider]

        llm_model = st.selectbox("Select Model", models)

    # API key section
    with st.expander("🔑 API Keys", expanded=True):
        st.info(
            "API keys are stored temporarily in memory and cleared when you close the browser."
        )
        if provider == "OpenAI":
            openai_api_key = st.text_input(
                label="OpenAI API Key",
                type="password",
                placeholder="sk-...",
                help="Get yours at https://platform.openai.com/api-keys")
            if openai_api_key:
                os.environ["OPENAI_API_KEY"] = openai_api_key
        elif provider == "Anthropic":
            anthropic_api_key = st.text_input(
                label="Anthropic API Key",
                type="password",
                placeholder="sk-ant-...",
                help="Get yours at https://console.anthropic.com/settings/keys")
            if anthropic_api_key:
                os.environ["ANTHROPIC_API_KEY"] = anthropic_api_key

    # upload PDF file
    uploaded_file = st.file_uploader(
        label="Upload PDF file", type="pdf", accept_multiple_files=False)

    if uploaded_file:
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                file_path = os.path.join(temp_dir, uploaded_file.name)

                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                with st.spinner("Processing uploaded document..."):
                    try:
                        raw_document = load_pdf(file_path)
                        if raw_document is None:
                            st.stop()

                        processed_chunks = chunk_document(raw_document)
                        if processed_chunks is None:
                            st.stop()
                        index_documents(processed_chunks)
                    except Exception as e:
                        st.error(f'Error processing PDF: {str(e)}')
                        st.stop()

            # Inform the user that the file is processed and display the uploaded PDF
            st.success(
                f"PDF file '{uploaded_file.name}' uploaded and processed successfully ✅."
            )
            display_pdf(uploaded_file)

        except Exception as e:
            st.error(f"An error occurred: {e}")
            st.stop()

col1, col2 = st.columns([6, 1])

with col1:
    st.title("📚 Intelligent PDF Assistant")
    st.markdown("""
        This app allows you to chat with the contents of a PDF document using Retrieval-Augmented Generation (RAG).
        Upload your PDF, ask questions and get answers based on the document's content.
        """)

with col2:
    st.button("Clear ↺", on_click=reset_chat)

# Initialize chat history
if "messages" not in st.session_state:
    reset_chat()

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if user_query := st.chat_input("Ask questions about the PDF"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": user_query})
    # Display user message in chat message container
    with st.chat_message(name="user", avatar="🧑‍💻"):
        st.write(user_query)

    # with st.spinner("Analyzing document..."):
    try:
        related_documents = find_relevant_docs(user_query)

        # Create a placeholder for the streaming response
        with st.chat_message(name="assistant", avatar="🤖"):
            message_placeholder = st.empty()
            full_response = ""

            # Iterate through the streaming response chunks
            for chunk in generate_answer(user_query, related_documents,
                                         provider, llm_model):
                full_response += chunk
                message_placeholder.markdown(full_response +
                                             "▌")  # Add a cursor effect

            message_placeholder.markdown(
                full_response)  # Display the final response

            # Add assistant response to chat history
            st.session_state.messages.append({
                "role": "assistant",
                "content": full_response
            })
    except Exception as e:
        st.error(f"An error occurred while processing your query: {e}")

# App footer
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: #888; font-size: 0.8em;">
        Built with Streamlit and LangChain
    </div>
    """,
    unsafe_allow_html=True)

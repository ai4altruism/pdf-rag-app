import os
import streamlit as st
import tempfile
import logging
from dotenv import load_dotenv
from utils.document_processor import DocumentProcessor
from utils.chroma_store import ChromaStore
from utils.llm_client import LLMClient

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# Initialize session state
def init_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None
    if "document_processor" not in st.session_state:
        st.session_state.document_processor = None
    if "llm_client" not in st.session_state:
        st.session_state.llm_client = None


# Display information about the documents
def display_document_info():
    if st.session_state.vector_store:
        doc_count = st.session_state.vector_store.get_document_count()
        if doc_count > 0:
            st.info(f"Documents in index: {doc_count} chunks")
        else:
            st.info("No documents in index. Please upload a PDF document.")
    else:
        st.warning("Vector store not initialized. Please refresh the page.")


# Initialize components
def initialize_components():
    # Load config from environment
    chunk_size = int(os.getenv("CHUNK_SIZE", "1000"))
    chunk_overlap = int(os.getenv("CHUNK_OVERLAP", "200"))
    embedding_model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    embedding_dimensions = os.getenv("EMBEDDING_DIMENSIONS")
    if embedding_dimensions:
        embedding_dimensions = int(embedding_dimensions)

    llm_api_key = os.getenv("LLM_API_KEY")
    llm_api_base_url = os.getenv("LLM_API_BASE_URL")
    llm_model_name = os.getenv("LLM_MODEL_NAME")
    reranking_system_prompt = os.getenv("RERANKING_SYSTEM_PROMPT")
    answer_system_prompt = os.getenv("ANSWER_SYSTEM_PROMPT")

    # Check for missing environment variables
    missing_vars = []
    for var_name, var_value in [
        ("LLM_API_KEY", llm_api_key),
        ("LLM_API_BASE_URL", llm_api_base_url),
        ("LLM_MODEL_NAME", llm_model_name),
        ("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY")),
    ]:
        if not var_value:
            missing_vars.append(var_name)

    if missing_vars:
        st.error(f"Missing required environment variables: {', '.join(missing_vars)}")
        st.stop()

    # Initialize components if not already done
    if not st.session_state.document_processor:
        st.session_state.document_processor = DocumentProcessor(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            embedding_model_name=embedding_model,
            embedding_dimensions=embedding_dimensions,
        )

    if not st.session_state.vector_store:
        st.session_state.vector_store = ChromaStore(
            embedding_model_name=embedding_model,
            embedding_dimensions=embedding_dimensions,
            persist_directory="data/chroma_db",
            collection_name="pdf_documents",
        )

    if not st.session_state.llm_client:
        st.session_state.llm_client = LLMClient(
            api_key=llm_api_key,
            api_base_url=llm_api_base_url,
            model_name=llm_model_name,
            reranking_system_prompt=reranking_system_prompt,
            answer_system_prompt=answer_system_prompt,
        )


# Process uploaded PDF
def process_uploaded_pdf(uploaded_file):
    try:
        # Create a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(uploaded_file.getvalue())
            temp_file_path = temp_file.name

        with st.spinner("Processing PDF..."):
            # Process the PDF - this returns Document objects
            document_chunks = st.session_state.document_processor.process_pdf(
                temp_file_path
            )

            # Add documents to ChromaDB
            st.session_state.vector_store.add_documents(document_chunks)

            # Clean up the temporary file
            os.unlink(temp_file_path)

            st.success(
                f"Successfully processed '{uploaded_file.name}' and added {len(document_chunks)} chunks to the vector store"
            )
    except Exception as e:
        logger.error(f"Error processing PDF: {str(e)}")
        st.error(f"Error processing PDF: {str(e)}")


# Process user query
def process_query(query):
    try:
        with st.spinner("Searching documents..."):
            # Add user message to chat
            st.session_state.messages.append({"role": "user", "content": query})

            # Search vector store - ChromaDB returns Document objects
            search_results = st.session_state.vector_store.search(query, top_k=10)

            if not search_results:
                answer = "I couldn't find any relevant information in the documents. Please try a different query or upload more documents."
            else:
                # Rerank chunks using LLM
                reranked_chunks = st.session_state.llm_client.rerank_chunks(
                    query, search_results, top_k=3
                )

                # Generate answer
                answer = st.session_state.llm_client.generate_answer(
                    query, reranked_chunks
                )

            # Add assistant message to chat
            st.session_state.messages.append({"role": "assistant", "content": answer})
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        error_message = f"Error processing your query: {str(e)}"
        st.session_state.messages.append(
            {"role": "assistant", "content": error_message}
        )


# Display chat messages
def display_chat():
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])


# Main app layout
def main():
    st.set_page_config(
        page_title="RAG Document Assistant", page_icon="📚", layout="wide"
    )

    # Initialize app
    init_session_state()
    initialize_components()

    # Sidebar
    with st.sidebar:
        st.title("📚 Document Manager")

        # Upload documents
        st.header("Upload Documents")
        uploaded_file = st.file_uploader("Upload a PDF document", type="pdf")
        if uploaded_file:
            process_uploaded_pdf(uploaded_file)

        # Document info
        st.header("Document Information")
        display_document_info()

        # Clear index button
        if st.button("Clear Document Index"):
            if st.session_state.vector_store:
                st.session_state.vector_store.clear()
                st.success("Document index cleared successfully")
                st.rerun()

    # Main content
    st.title("🤖 Document Question Answering")
    st.write("Upload PDF documents and ask questions about their content.")

    # Chat interface
    display_chat()

    # Query input
    if query := st.chat_input("Ask a question about your documents..."):
        process_query(query)
        st.rerun()


if __name__ == "__main__":
    main()

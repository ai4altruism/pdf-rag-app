# 📚 PDF RAG Application

A Retrieval-Augmented Generation (RAG) application for extracting information from PDF documents using advanced chunking, OpenAI embeddings, ChromaDB for vector storage, LLM-based re-ranking, and question answering.

## Features

- **Advanced PDF Processing**: Upload, extract text, and chunk using RecursiveCharacterTextSplitter for more semantically meaningful chunks
- **OpenAI Embeddings**: Generate high-quality embeddings using OpenAI's latest models (text-embedding-3-small/large)
- **ChromaDB Vector Storage**: Store and retrieve document chunks efficiently using ChromaDB
- **LLM Re-ranking**: Use a language model to improve the relevance of retrieved chunks
- **Question Answering**: Generate concise answers based on document context
- **Web Interface**: Simple Streamlit UI for document upload and Q&A

## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/pdf-rag-app.git
cd pdf-rag-app
```

### 2. Create a Virtual Environment (Recommended)

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Create a `.env` file in the project root with your API keys and configuration:

```plaintext
# LLM API Configuration
LLM_API_KEY=your_api_key_here
LLM_API_BASE_URL=https://api.provider.com/v1
LLM_MODEL_NAME=model_name_here

# OpenAI API Configuration (for embeddings)
OPENAI_API_KEY=your_openai_api_key_here

# Vector Database Configuration
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
EMBEDDING_MODEL=text-embedding-3-small
EMBEDDING_DIMENSIONS=1536  # Optional: Can be reduced for better efficiency

# Re-ranking System Prompt
RERANKING_SYSTEM_PROMPT=You are an expert at analyzing document chunks for relevance to a query. For each document chunk, assign a relevance score from 0-10. Focus on semantic relevance rather than keyword matching. A document with a score of 10 perfectly answers the query, while a score of 0 means it's completely irrelevant. Explain your reasoning for each score.

# Answer Generation System Prompt
ANSWER_SYSTEM_PROMPT=You are a helpful assistant that answers questions based on the provided document context. Only answer what can be inferred from the context. If the context doesn't provide enough information to answer the question, state that clearly. Do not make up information.
```

### 5. Create Required Directories

The application will automatically create the necessary directories, but you can create them manually if needed:

```bash
mkdir -p data/pdfs
mkdir -p data/chroma_db
```

### 6. Run the Application

```bash
streamlit run app.py
```

This will start the Streamlit server and open the application in your default web browser.

## Using the Application

1. **Upload Documents**: Use the sidebar to upload PDF documents.
2. **Ask Questions**: Type your questions in the chat input at the bottom of the page.
3. **View Answers**: See the AI-generated answers based on your documents.
4. **Manage Documents**: Clear the document index using the button in the sidebar if needed.

## How It Works

1. **Document Processing Pipeline**:

   - PDF documents are loaded and text is extracted
   - Text is split into semantic chunks using RecursiveCharacterTextSplitter with natural language boundaries
   - Chunks are stored in ChromaDB with embeddings from OpenAI's embedding models

2. **Query Processing Pipeline**:
   - User query is embedded using the same OpenAI model
   - Similar document chunks are retrieved from ChromaDB using vector similarity search
   - Retrieved chunks are re-ranked by the LLM based on relevance to the query
   - The most relevant chunks are used as context for the final answer

## Advanced Features

### OpenAI Embeddings

The application uses OpenAI's powerful embedding models for better semantic understanding:

- **Configurable Models**: Choose between different OpenAI embedding models
- **Dimension Reduction**: Optionally reduce embedding dimensions to optimize storage and cost
- **Batch Processing**: Efficiently process documents in batches to respect API rate limits

### Optimized Document Chunking

- **Semantic Chunking**: Uses RecursiveCharacterTextSplitter with multiple separators
- **Natural Boundaries**: Respects paragraphs, sentences, and other linguistic structures
- **Configurable Parameters**: Adjust chunk size and overlap to suit your documents

### ChromaDB Integration

- **Persistent Storage**: Document embeddings are stored persistently for future use
- **Efficient Retrieval**: Fast and accurate similarity search for relevant chunks
- **Collection Management**: Create, update, and clear document collections

## Customization

- Adjust chunk size and overlap in the `.env` file to optimize for your documents
- Modify system prompts to change the behavior of the re-ranking and answer generation
- Experiment with different embedding models and dimensions for better performance or reduced cost
- Configure batch sizes for processing large document collections

## Requirements

- Python 3.8+
- Streamlit
- PyPDF
- OpenAI API access
- LangChain
- ChromaDB

## Directory Structure

```
pdf-rag-app/
├── .env                     # Environment configuration file
├── app.py                   # Main Streamlit application
├── requirements.txt         # Project dependencies
├── README.md                # Project documentation
├── .gitignore               # Git ignore configuration
├── utils/                   # Utility modules
│   ├── __init__.py          # Make utils a proper package
│   ├── document_processor.py # PDF loading, chunking, and embedding
│   ├── chroma_store.py      # ChromaDB vector store operations
│   └── llm_client.py        # LLM API client for reranking and QA
└── data/                    # Data storage directories
    ├── pdfs/                # Directory for uploaded PDFs
    └── chroma_db/           # ChromaDB persistence directory
```

## License

This project is licensed under the GNU General Public License v3.0 - see the [LICENSE](LICENSE) file for details.

When using or distributing this software, please attribute as follows:

```
PDF RAG Application
Copyright (c) 2025 AI for Altruism Inc
License: GNU GPL v3.0
```

## Contact

For questions, suggestions, or collaboration opportunities, please contact:

Email: team@ai4altruism.org

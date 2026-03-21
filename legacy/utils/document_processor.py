import os
import logging
from typing import List, Dict, Any
import pypdf
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from openai import OpenAI
import time
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class OpenAIEmbeddings:
    def __init__(
        self,
        api_key: str = None,
        model_name: str = "text-embedding-3-small",
        dimensions: int = None,
    ):
        """
        Initialize the OpenAI embeddings client.

        Args:
            api_key: OpenAI API key (defaults to env variable)
            model_name: OpenAI embedding model to use
            dimensions: Optional parameter to reduce embedding dimensions
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model_name = model_name
        self.dimensions = dimensions
        self.max_retries = 3
        self.retry_delay = 1

        if not self.api_key:
            raise ValueError(
                "OpenAI API key is required. Set it as OPENAI_API_KEY environment variable."
            )

        self.client = OpenAI(api_key=self.api_key)
        logger.info(f"Initialized OpenAI embeddings with model: {model_name}")

    def get_embedding(self, text: str) -> List[float]:
        """
        Get embedding for a single text string.

        Args:
            text: Text to embed

        Returns:
            Embedding vector
        """
        return self.get_embeddings([text])[0]

    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Get embeddings for a list of texts with retry logic.

        Args:
            texts: List of strings to embed

        Returns:
            List of embedding vectors
        """
        attempts = 0
        while attempts < self.max_retries:
            try:
                # Create the embeddings request
                params = {"model": self.model_name, "input": texts}

                # Add dimensions parameter if specified
                if self.dimensions:
                    params["dimensions"] = self.dimensions

                response = self.client.embeddings.create(**params)

                # Extract embeddings from response
                embeddings = [data.embedding for data in response.data]

                return embeddings

            except Exception as e:
                attempts += 1
                logger.warning(
                    f"Error creating OpenAI embeddings (attempt {attempts}/{self.max_retries}): {str(e)}"
                )
                if attempts < self.max_retries:
                    # Exponential backoff
                    sleep_time = self.retry_delay * (2 ** (attempts - 1))
                    logger.info(f"Retrying in {sleep_time} seconds...")
                    time.sleep(sleep_time)
                else:
                    logger.error(
                        f"Failed to create embeddings after {self.max_retries} attempts"
                    )
                    raise


class DocumentProcessor:
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        embedding_model_name: str = "text-embedding-3-small",
        embedding_dimensions: int = None,
    ):
        """
        Initialize the document processor.

        Args:
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
            embedding_model_name: Name of the OpenAI embedding model
            embedding_dimensions: Optional parameter to reduce embedding dimensions
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # Create OpenAI embedding model
        logger.info(f"Initializing OpenAI embedding model: {embedding_model_name}")
        self.embedding_model = OpenAIEmbeddings(
            model_name=embedding_model_name, dimensions=embedding_dimensions
        )
        logger.info("OpenAI embedding model initialized successfully")

    def load_pdf(self, file_path: str) -> str:
        """
        Extract text from a PDF file.

        Args:
            file_path: Path to the PDF file

        Returns:
            Extracted text content
        """
        logger.info(f"Loading PDF from {file_path}")

        try:
            with open(file_path, "rb") as file:
                pdf_reader = pypdf.PdfReader(file)
                text = ""

                # Extract text from each page
                for page_num in range(len(pdf_reader.pages)):
                    page = pdf_reader.pages[page_num]
                    text += page.extract_text() + "\n\n"

                logger.info(
                    f"Successfully extracted {len(pdf_reader.pages)} pages from {file_path}"
                )
                return text
        except Exception as e:
            logger.error(f"Error loading PDF {file_path}: {str(e)}")
            raise

    def chunk_text(self, text: str, metadata: Dict[str, Any] = None) -> List[Document]:
        """
        Split text into chunks with metadata using RecursiveCharacterTextSplitter.

        Args:
            text: Text to split into chunks
            metadata: Additional metadata to include with each chunk

        Returns:
            List of Document objects with text and metadata
        """
        if metadata is None:
            metadata = {}

        logger.info(f"Chunking text of length {len(text)} characters")

        # Create a LangChain Document
        doc = Document(page_content=text, metadata=metadata)

        # Configure the text splitter with better parameters for PDF content
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
            keep_separator=True,
        )

        # Split the document
        chunks = splitter.split_documents([doc])
        logger.info(
            f"Created {len(chunks)} chunks using RecursiveCharacterTextSplitter"
        )

        # Add chunk_id to each document's metadata
        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk_id"] = i
            chunk.metadata["chunk_size"] = len(chunk.page_content)

        return chunks

    def create_embeddings_batch(
        self, chunks: List[Document], batch_size: int = 10
    ) -> List[Document]:
        """
        Create embeddings for Document objects in batches.

        Args:
            chunks: List of Document objects
            batch_size: Number of chunks to process in each batch

        Returns:
            List of Document objects with embeddings added to metadata
        """
        logger.info(
            f"Creating embeddings for {len(chunks)} chunks in batches of {batch_size}"
        )

        # Process chunks in batches
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            batch_end = min(i + batch_size, len(chunks))

            # Extract text from chunks
            texts = [chunk.page_content for chunk in batch]

            logger.info(
                f"Processing batch {i//batch_size + 1}/{(len(chunks)-1)//batch_size + 1} (chunks {i+1}-{batch_end})"
            )

            try:
                # Generate embeddings for the batch
                embeddings = self.embedding_model.get_embeddings(texts)

                # Add embeddings to chunks' metadata
                for j, embedding in enumerate(embeddings):
                    batch[j].metadata["embedding"] = embedding

            except Exception as e:
                logger.error(f"Error creating embeddings for batch: {str(e)}")
                # Set a placeholder for failed embeddings
                for chunk in batch:
                    if "embedding" not in chunk.metadata:
                        chunk.metadata["embedding_error"] = str(e)

        # Count chunks that have embeddings
        successful_embeddings = sum(
            1 for chunk in chunks if "embedding" in chunk.metadata
        )
        logger.info(
            f"Successfully created embeddings for {successful_embeddings}/{len(chunks)} chunks"
        )

        return chunks

    def process_pdf(
        self, file_path: str, create_embeddings: bool = False, batch_size: int = 10
    ) -> List[Document]:
        """
        Process a PDF: load, chunk, and optionally create embeddings.

        Args:
            file_path: Path to the PDF file
            create_embeddings: Whether to create embeddings (False by default since ChromaDB can handle this)
            batch_size: Number of chunks to process in each batch when creating embeddings

        Returns:
            List of Document objects with text and optional embeddings
        """
        # Extract filename for metadata
        filename = os.path.basename(file_path)

        # Load PDF
        text = self.load_pdf(file_path)

        # Create metadata
        metadata = {"source": filename, "file_path": file_path, "document_type": "pdf"}

        # Chunk text
        chunks = self.chunk_text(text, metadata)

        # Create embeddings if requested
        if create_embeddings:
            chunks = self.create_embeddings_batch(chunks, batch_size=batch_size)

        return chunks

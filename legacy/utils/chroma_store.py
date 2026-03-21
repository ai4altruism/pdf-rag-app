import os
import logging
from typing import List, Dict, Any, Optional, Tuple
import chromadb
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class ChromaStore:
    def __init__(
        self,
        embedding_model_name: str = "text-embedding-3-small",
        embedding_dimensions: int = None,
        persist_directory: str = "data/chroma_db",
        collection_name: str = "pdf_documents",
        api_key: str = None,
    ):
        """
        Initialize the ChromaDB vector store with OpenAI embeddings.

        Args:
            embedding_model_name: Name of the OpenAI embedding model
            embedding_dimensions: Optional parameter to reduce embedding dimensions
            persist_directory: Directory to persist the ChromaDB
            collection_name: Name of the ChromaDB collection
            api_key: OpenAI API key (defaults to env variable)
        """
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.embedding_model_name = embedding_model_name
        self.embedding_dimensions = embedding_dimensions

        # Create directory if it doesn't exist
        os.makedirs(persist_directory, exist_ok=True)

        # Initialize OpenAI embedding function
        logger.info(f"Initializing OpenAI embedding model: {embedding_model_name}")
        self.openai_api_key = api_key or os.getenv("OPENAI_API_KEY")

        if not self.openai_api_key:
            raise ValueError(
                "OpenAI API key is required. Set it as OPENAI_API_KEY environment variable."
            )

        # Configure OpenAI embeddings with optional dimensions parameter
        embedding_kwargs = (
            {"dimensions": embedding_dimensions} if embedding_dimensions else {}
        )

        self.embeddings = OpenAIEmbeddings(
            model=embedding_model_name,
            openai_api_key=self.openai_api_key,
            **embedding_kwargs,
        )

        # Initialize ChromaDB client
        self.chroma_client = chromadb.PersistentClient(path=persist_directory)

        # Initialize or load collection
        self._initialize_chroma()

        logger.info(
            f"ChromaStore initialized with collection '{collection_name}' and model '{embedding_model_name}'"
        )

    def _initialize_chroma(self):
        """
        Initialize or load existing ChromaDB collection.
        """
        try:
            # Try to get the collection if it exists
            self.chroma_client.get_collection(self.collection_name)
            logger.info(f"Loaded existing collection '{self.collection_name}'")
        except Exception:
            # Create collection if it doesn't exist
            self.chroma_client.create_collection(self.collection_name)
            logger.info(f"Created new collection '{self.collection_name}'")

        # Initialize LangChain wrapper
        self.db = Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embeddings,
            persist_directory=self.persist_directory,
        )

    def add_documents(self, documents: List[Document], batch_size: int = 20):
        """
        Add documents to the vector store in batches.

        Args:
            documents: List of Document objects to add
            batch_size: Number of documents to add in each batch
        """
        if not documents:
            logger.warning("No documents to add")
            return

        logger.info(
            f"Adding {len(documents)} documents to ChromaDB in batches of {batch_size}"
        )

        # Process in batches to handle rate limits and large document sets
        total_batches = (len(documents) + batch_size - 1) // batch_size

        for i in range(0, len(documents), batch_size):
            batch = documents[i : i + batch_size]
            batch_end = min(i + batch_size, len(documents))

            logger.info(
                f"Processing batch {i//batch_size + 1}/{total_batches} (documents {i+1}-{batch_end})"
            )

            try:
                # Add documents to ChromaDB through LangChain interface
                self.db.add_documents(batch)

                # Sleep briefly to respect API rate limits
                if i + batch_size < len(documents):
                    time.sleep(0.5)

            except Exception as e:
                logger.error(f"Error adding batch to ChromaDB: {str(e)}")
                # Continue with next batch rather than failing completely
                continue

        logger.info(f"Successfully added documents to ChromaDB")

    def search(self, query: str, top_k: int = 5) -> List[Document]:
        """
        Search for documents similar to the query.

        Args:
            query: Query text
            top_k: Number of top documents to retrieve

        Returns:
            List of Document objects similar to the query
        """
        logger.info(f"Searching for query: '{query}' with top_k={top_k}")

        try:
            # Perform similarity search
            results = self.db.similarity_search(query, k=top_k)

            logger.info(f"Found {len(results)} results")
            return results
        except Exception as e:
            logger.error(f"Error searching ChromaDB: {str(e)}")
            return []

    def search_with_scores(
        self, query: str, top_k: int = 5
    ) -> List[Tuple[Document, float]]:
        """
        Search for documents with relevance scores.

        Args:
            query: Query text
            top_k: Number of top documents to retrieve

        Returns:
            List of tuples (Document, score)
        """
        logger.info(f"Searching with scores for query: '{query}' with top_k={top_k}")

        try:
            # Perform similarity search with scores
            results = self.db.similarity_search_with_relevance_scores(query, k=top_k)

            logger.info(f"Found {len(results)} results with scores")
            return results
        except Exception as e:
            logger.error(f"Error searching ChromaDB with scores: {str(e)}")
            return []

    def clear(self):
        """
        Clear all documents from the collection.
        """
        logger.info(f"Clearing collection '{self.collection_name}'")

        try:
            # Delete collection
            self.chroma_client.delete_collection(self.collection_name)

            # Recreate collection
            self.chroma_client.create_collection(self.collection_name)

            # Reinitialize LangChain wrapper
            self.db = Chroma(
                collection_name=self.collection_name,
                embedding_function=self.embeddings,
                persist_directory=self.persist_directory,
            )

            logger.info("Collection cleared successfully")
        except Exception as e:
            logger.error(f"Error clearing collection: {str(e)}")
            raise

    def get_document_count(self) -> int:
        """
        Get the number of documents in the collection.

        Returns:
            Number of documents
        """
        try:
            collection = self.chroma_client.get_collection(self.collection_name)
            count = collection.count()
            return count
        except Exception as e:
            logger.error(f"Error getting document count: {str(e)}")
            return 0

    def get_collection_info(self) -> Dict[str, Any]:
        """
        Get information about the collection.

        Returns:
            Dictionary with collection information
        """
        try:
            collection = self.chroma_client.get_collection(self.collection_name)
            count = collection.count()

            return {
                "name": self.collection_name,
                "document_count": count,
                "embedding_model": self.embedding_model_name,
                "embedding_dimensions": self.embedding_dimensions or "default",
                "persist_directory": self.persist_directory,
            }
        except Exception as e:
            logger.error(f"Error getting collection info: {str(e)}")
            return {"name": self.collection_name, "error": str(e)}

# utils/openai_embeddings.py
import os
import logging
from typing import List, Union
from openai import OpenAI

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

        if not self.api_key:
            raise ValueError("OpenAI API key is required")

        self.client = OpenAI(api_key=self.api_key)
        logger.info(f"Initialized OpenAI embeddings with model: {model_name}")

    def get_embeddings(self, texts: Union[str, List[str]]) -> List[List[float]]:
        """
        Get embeddings for a text or list of texts.

        Args:
            texts: A string or list of strings to embed

        Returns:
            A list of embedding vectors
        """
        # Ensure texts is a list
        if isinstance(texts, str):
            texts = [texts]

        try:
            # Create the embeddings request
            params = {"model": self.model_name, "input": texts}

            # Add dimensions parameter if specified
            if self.dimensions:
                params["dimensions"] = self.dimensions

            response = self.client.embeddings.create(**params)

            # Extract embeddings from response
            embeddings = [data.embedding for data in response.data]

            logger.info(f"Successfully created {len(embeddings)} embeddings")
            return embeddings

        except Exception as e:
            logger.error(f"Error creating OpenAI embeddings: {str(e)}")
            raise

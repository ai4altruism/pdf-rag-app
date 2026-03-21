import os
import json
import logging
from typing import List, Dict, Any, Optional, Tuple
from openai import OpenAI
import time
from langchain_core.documents import Document

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class LLMClient:
    def __init__(
        self,
        api_key: str,
        api_base_url: str,
        model_name: str,
        reranking_system_prompt: str,
        answer_system_prompt: str,
    ):
        """
        Initialize the LLM client.

        Args:
            api_key: API key for the LLM service
            api_base_url: Base URL for the LLM API
            model_name: Name of the model to use
            reranking_system_prompt: System prompt for re-ranking
            answer_system_prompt: System prompt for answer generation
        """
        self.model_name = model_name
        self.reranking_system_prompt = reranking_system_prompt
        self.answer_system_prompt = answer_system_prompt

        # Initialize client
        self.client = OpenAI(api_key=api_key, base_url=api_base_url)

        # Check if model supports temperature parameter
        self.supports_temperature = True

        logger.info(f"Initialized LLM client for model: {model_name}")

    def _call_llm(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.2,
        max_retries: int = 3,
    ) -> Optional[str]:
        """
        Call the LLM API with retries.

        Args:
            messages: List of message dictionaries
            temperature: Temperature for generation
            max_retries: Maximum number of retries

        Returns:
            Generated text or None if all retries fail
        """
        retries = 0
        while retries < max_retries:
            try:
                # Prepare parameters - conditionally include temperature
                params = {
                    "model": self.model_name,
                    "messages": messages,
                }

                # Only include temperature if the model supports it
                if self.supports_temperature:
                    params["temperature"] = temperature

                response = self.client.chat.completions.create(**params)
                return response.choices[0].message.content
            except Exception as e:
                retries += 1

                # Check if the error is related to temperature parameter
                error_message = str(e)
                if "temperature" in error_message and "not supported" in error_message:
                    logger.warning(
                        "Model does not support temperature parameter, disabling it for future requests"
                    )
                    self.supports_temperature = False
                    continue  # Retry immediately without temperature

                logger.warning(
                    f"LLM API call failed (attempt {retries}/{max_retries}): {str(e)}"
                )
                if retries < max_retries:
                    # Exponential backoff
                    time.sleep(2**retries)
                else:
                    logger.error(f"LLM API call failed after {max_retries} attempts")
                    return None

    def rerank_chunks(
        self, query: str, chunks: List[Document], top_k: int = 3
    ) -> List[Document]:
        """
        Rerank chunks using the LLM.

        Args:
            query: User query
            chunks: List of Document objects from the vector store
            top_k: Number of top chunks to return after reranking

        Returns:
            List of reranked Document objects
        """
        if not chunks:
            logger.warning("No chunks to rerank")
            return []

        logger.info(f"Reranking {len(chunks)} chunks for query: '{query}'")

        # Prepare chunks for reranking
        chunks_content = []
        for i, chunk in enumerate(chunks):
            chunks_content.append(f"DOCUMENT {i+1}:\n{chunk.page_content}")

        chunks_text = "\n\n".join(chunks_content)

        # Create reranking prompt
        reranking_prompt = f"""
Query: {query}

Documents to evaluate:
{chunks_text}

For each document, assign a relevance score from 0-10 based on how well it answers the query.
Format your response as a JSON object with document numbers as keys and objects with 'score' and 'reasoning' as values.
Example:
{{
  "1": {{ "score": 8, "reasoning": "This document directly addresses..." }},
  "2": {{ "score": 3, "reasoning": "This document only tangentially mentions..." }}
}}
"""

        # Call LLM for reranking
        messages = [
            {"role": "system", "content": self.reranking_system_prompt},
            {"role": "user", "content": reranking_prompt},
        ]

        reranking_result = self._call_llm(messages)

        if not reranking_result:
            logger.warning("Reranking failed, returning original order")
            return chunks[:top_k]

        try:
            # Extract JSON from response (handle potential text before/after JSON)
            reranking_result = reranking_result.strip()
            # Find the first '{' and the last '}'
            start_idx = reranking_result.find("{")
            end_idx = reranking_result.rfind("}") + 1
            if start_idx >= 0 and end_idx > start_idx:
                json_str = reranking_result[start_idx:end_idx]
                scores = json.loads(json_str)
            else:
                raise ValueError("No valid JSON found in response")

            # Add scores to chunks
            for i, chunk in enumerate(chunks):
                doc_id = str(i + 1)
                if doc_id in scores:
                    # Store scores in the document's metadata
                    chunk.metadata["llm_score"] = scores[doc_id]["score"]
                    chunk.metadata["llm_reasoning"] = scores[doc_id]["reasoning"]
                else:
                    chunk.metadata["llm_score"] = 0
                    chunk.metadata["llm_reasoning"] = "Not evaluated by LLM"

            # Sort chunks by LLM score
            reranked_chunks = sorted(
                chunks, key=lambda x: x.metadata.get("llm_score", 0), reverse=True
            )

            logger.info("Chunks reranked successfully")
            return reranked_chunks[:top_k]

        except Exception as e:
            logger.error(f"Error parsing reranking result: {str(e)}")
            logger.error(f"Raw result: {reranking_result}")
            # Fall back to original order
            return chunks[:top_k]

    def generate_answer(self, query: str, context_chunks: List[Document]) -> str:
        """
        Generate an answer based on the query and context chunks.

        Args:
            query: User query
            context_chunks: List of Document objects

        Returns:
            Generated answer
        """
        logger.info(
            f"Generating answer for query: '{query}' with {len(context_chunks)} context chunks"
        )

        # Prepare context
        context_content = []
        for i, chunk in enumerate(context_chunks):
            score_info = ""
            if "llm_score" in chunk.metadata:
                score_info = f" [Relevance Score: {chunk.metadata['llm_score']}/10]"

            # Include source information if available
            source_info = ""
            if "source" in chunk.metadata:
                source_info = f" [Source: {chunk.metadata['source']}]"

            context_content.append(
                f"DOCUMENT {i+1}{score_info}{source_info}:\n{chunk.page_content}"
            )

        context_text = "\n\n".join(context_content)

        # Create answer prompt
        answer_prompt = f"""
Question: {query}

Context:
{context_text}

Based on the provided context, please answer the question. If the context doesn't contain enough information to provide a complete answer, acknowledge the limitations in your response.
"""

        # Call LLM for answer
        messages = [
            {"role": "system", "content": self.answer_system_prompt},
            {"role": "user", "content": answer_prompt},
        ]

        answer = self._call_llm(messages)

        if not answer:
            return "I apologize, but I'm unable to generate an answer at this time. Please try again later."

        return answer

import requests
from dotenv import load_dotenv

load_dotenv()

import concurrent.futures
import os
from typing import Any, List

from llama_index.core.embeddings import BaseEmbedding


class EmebeddingsClient(BaseEmbedding):
    """
    A client class for interacting with an embeddings server to retrieve embeddings for text data.
    """

    model_name: str = os.getenv("EMBEDDING_MODEL")

    def _get_query_embedding(self, query: str) -> List[float]:
        """
        Get the embedding for a single query.

        Args:
            query (str): The query text.

        Returns:
            List[float]: The embedding vector for the query.

        """
        response = requests.post(
            os.getenv("EMBEDDINGS_SERVER_URL"),
            json={"prompt": query, "model": self.model_name},
        )
        # raise error if the response is not successful
        response.raise_for_status()
        # return the embeddings from the response
        return response.json().get("embedding")

    def _get_text_embedding(self, text: str) -> List[float]:
        """
        Get the embedding for a single text.

        Args:
            text (str): The text.

        Returns:
            List[float]: The embedding vector for the text.

        """
        response = requests.post(
            os.getenv("EMBEDDINGS_SERVER_URL"),
            json={"prompt": text, "model": self.model_name},
        )
        # raise error if the response is not successful
        response.raise_for_status()
        # return the embeddings from the response
        return response.json().get("embedding")

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Get the embeddings for a list of texts.

        Args:
            texts (List[str]): The list of texts.

        Returns:
            List[List[float]]: The list of embedding vectors for the texts.

        """
        embeddings = []
        # use ThreadPoolExecutor to make multiple requests concurrently
        with concurrent.futures.ThreadPoolExecutor() as executor:
            responses = [
                executor.submit(
                    requests.post,
                    os.getenv("EMBEDDINGS_SERVER_URL"),
                    json={"prompt": text, "model": self.model_name},
                )
                for text in texts
            ]

            # collect the embeddings from the responses
            for response in concurrent.futures.as_completed(responses):
                # covert future to response object
                response = response.result()
                # raise error if the response is not successful
                response.raise_for_status()

                # get the embeddings from the response
                embedding = response.json().get("embedding")
                embeddings.append(embedding)

        return embeddings

    async def _aget_query_embedding(self, query: str) -> List[float]:
        """
        Get the embedding for a single query asynchronously.

        Args:
            query (str): The query text.

        Returns:
            List[float]: The embedding vector for the query.

        """
        return self._get_query_embedding(query)

    async def _aget_text_embedding(self, text: str) -> List[float]:
        """
        Get the embedding for a single text asynchronously.

        Args:
            text (str): The text.

        Returns:
            List[float]: The embedding vector for the text.

        """
        return self._get_text_embedding(text)

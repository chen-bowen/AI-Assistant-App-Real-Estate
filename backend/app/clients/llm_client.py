import json
import os
from typing import Any, List, Optional

import requests
from dotenv import load_dotenv
from llama_index.core.llms import (CompletionResponse, CompletionResponseGen,
                                   CustomLLM, LLMMetadata)
from llama_index.core.llms.callbacks import llm_completion_callback

from app.settings import llm_config_from_env

load_dotenv()


class LLMClient(CustomLLM):
    """
    A client class for interacting with the LLM (Language Model) service (currently Ollama).

    Attributes:
        context_window (int): The size of the context window for LLM.
        num_output (int): The number of output tokens to generate.
        model_name (str): The name of the LLM model to use.

    Methods:
        metadata(): Get the metadata of the LLM model.
        complete(prompt: str, **kwargs: Any): Perform a completion request to the LLM service.
        stream_complete(prompt: str, **kwargs: Any): Perform a streaming completion request to the LLM service.
    """

    context_window: int = 8000
    num_output: int = 256
    model_name: str = os.getenv("LLM_MODEL")

    @property
    def metadata(self) -> LLMMetadata:
        """Get LLM metadata."""
        return LLMMetadata(
            context_window=self.context_window,
            num_output=self.num_output,
            model_name=self.model_name,
        )

    @llm_completion_callback()
    def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        """
        Perform a completion request to the LLM service without streaming

        Args:
            prompt (str): The prompt text for completion.

        Returns:
            CompletionResponse: The completion response from the LLM service.
        """
        # send request to LLM server and process the response to text
        response = requests.post(
            os.getenv("LLM_SERVER_URL"),
            json={
                "messages": [
                    {
                        "role": "system",
                        "content": os.getenv("SYSTEM_PROMPT"),
                    },
                    {"role": "user", "content": prompt},
                ],
                "temperature": os.getenv("LLM_TEMPERATURE"),
                "max_tokens": -1,
                "stream": False,
            },
        )
        # raise error if the response is not successful
        response.raise_for_status()
        # return the completion response object
        response = response.json().get("choices")[0].get("message").get("content")
        return CompletionResponse(text=response)

    @llm_completion_callback()
    def stream_complete(self, prompt: str, context: Optional[List[Any]] = [], **kwargs: Any) -> CompletionResponseGen:
        """
        Perform a streaming completion request to the LLM service.

        Args:
            prompt (str): The prompt text for completion.

        Yields:
            CompletionResponse: The completion response from the LLM service.
        """

        # Send a POST request to the LLM server with streaming enabled
        res = requests.post(
            os.getenv("LLM_SERVER_URL"),
            json={
                "messages": [
                    {
                        "role": "system",
                        "content": os.getenv("SYSTEM_PROMPT"),
                    },
                    {"role": "user", "content": prompt},
                ],
                "temperature": os.getenv("LLM_TEMPERATURE"),
                "max_tokens": -1,
                "stream": True,
            },
        )
        # Raise an exception if the response is not successful
        res.raise_for_status()

        # for each line in the response, yield a completion response
        response = ""
        for line in res.iter_lines():
            line = line.decode("utf-8").lstrip("data: ")
            if line != "[DONE]" and line.strip():  # check if line is not empty and not [DONE]

                # load the response body into a JSON
                body = json.loads(line)

                # get streamed token from the response
                delta = body.get("choices")[0].get("delta")

                # if a new token is present, add it to the response
                if delta:
                    # add streaming token to CompletionResponse
                    token = delta.get("content")
                    response += token
                    yield CompletionResponse(text=response, delta=token)


def get_llm() -> LLMClient:
    """
    Get an instance of the LLMClient class.

    Returns:
        LLMClient: An instance of the LLMClient class.
    """
    return LLMClient(**llm_config_from_env())

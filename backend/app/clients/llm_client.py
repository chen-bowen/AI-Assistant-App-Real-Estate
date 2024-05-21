import json
import os
from typing import Any, Dict, Sequence, Tuple

import httpx
from httpx import Timeout
from llama_index.core.base.llms.types import (
    ChatMessage,
    ChatResponse,
    ChatResponseGen,
    ChatResponseAsyncGen,
    CompletionResponse,
    CompletionResponseAsyncGen,
    CompletionResponseGen,
    LLMMetadata,
    MessageRole,
)
from llama_index.core.bridge.pydantic import Field
from llama_index.core.constants import DEFAULT_CONTEXT_WINDOW, DEFAULT_NUM_OUTPUTS
from llama_index.core.llms.callbacks import llm_chat_callback, llm_completion_callback
from llama_index.core.llms.custom import CustomLLM

DEFAULT_REQUEST_TIMEOUT = 30.0


def get_additional_kwargs(response: Dict[str, Any], exclude: Tuple[str, ...]) -> Dict[str, Any]:
    return {k: v for k, v in response.items() if k not in exclude}


class LLMClient(CustomLLM):
    """Ollama LLM.

    Visit https://ollama.com/ to download and install Ollama.

    Run `ollama serve` to start a server.

    Run `ollama pull <name>` to download a model to run.

    Examples:
        `pip install llama-index-llms-ollama`

        ```python
        from llama_index.llms.ollama import Ollama

        llm = Ollama(model="llama2", request_timeout=60.0)

        response = llm.complete("What is the capital of France?")
        print(response)
        ```
    """

    inference_url: str = Field(
        default=os.getenv("LLM_SERVER_URL", "http://localhost:11434/api/chat"),
        description="Base url the model is hosted under.",
    )
    model: str = Field(description="The Ollama model to use.")
    temperature: float = Field(
        default=0.75,
        description="The temperature to use for sampling.",
        gte=0.0,
        lte=1.0,
    )
    context_window: int = Field(
        default=DEFAULT_CONTEXT_WINDOW,
        description="The maximum number of context tokens for the model.",
        gt=0,
    )
    request_timeout: float = Field(
        default=DEFAULT_REQUEST_TIMEOUT,
        description="The timeout for making http request to Ollama API server",
    )
    prompt_key: str = Field(default="prompt", description="The key to use for the prompt in API calls.")
    json_mode: bool = Field(
        default=False,
        description="Whether to use JSON mode for the Ollama API.",
    )
    additional_kwargs: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional model parameters for the Ollama API.",
    )

    @classmethod
    def class_name(cls) -> str:
        return "Ollama_llm"

    @property
    def metadata(self) -> LLMMetadata:
        """LLM metadata."""
        return LLMMetadata(
            context_window=self.context_window,
            num_output=DEFAULT_NUM_OUTPUTS,
            model_name=self.model,
            is_chat_model=True,  # Ollama supports chat API for all models
        )

    @property
    def _model_kwargs(self) -> Dict[str, Any]:
        base_kwargs = {
            "temperature": self.temperature,
            "num_ctx": self.context_window,
        }
        return {
            **base_kwargs,
            **self.additional_kwargs,
        }

    @llm_chat_callback()
    def stream_chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponseGen:
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": message.role.value,
                    "content": message.content,
                    **message.additional_kwargs,
                }
                for message in messages
            ],
            "options": self._model_kwargs,
            "stream": True,
            **kwargs,
        }

        if self.json_mode:
            payload["format"] = "json"

        with httpx.Client(timeout=Timeout(self.request_timeout)) as client:
            with client.stream(
                method="POST",
                url=self.inference_url,
                json=payload,
            ) as response:
                response.raise_for_status()
                text = ""
                for line in response.iter_lines():
                    # check if line is not empty
                    if line:
                        # streaming data is in the format of "data: {json}\n", strip to parse to json
                        line = line.lstrip("data: ")
                        # check if the line is the end of the response
                        if line == "[DONE]":
                            break
                        # load the response body into a JSON
                        body = json.loads(line)

                        # get streamed chunk from the response
                        message = body.get("choices")[0].get("delta", "")
                        delta = message.get("content", "")
                        text += delta

                        if message:
                            yield ChatResponse(
                                message=ChatMessage(
                                    content=text,
                                    role=MessageRole(message.get("role")),
                                    additional_kwargs=get_additional_kwargs(message, ("choices", "role")),
                                ),
                                delta=delta,
                                raw=body,
                                additional_kwargs=get_additional_kwargs(body, ("id", "object", "system_fingerprint", "choices")),
                            )

    @llm_chat_callback()
    async def achat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponseAsyncGen:
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": message.role.value,
                    "content": message.content,
                    **message.additional_kwargs,
                }
                for message in messages
            ],
            "options": self._model_kwargs,
            "stream": False,
            **kwargs,
        }

        if self.json_mode:
            payload["format"] = "json"

        async with httpx.AsyncClient(timeout=Timeout(self.request_timeout)) as client:
            response = await client.post(
                url=self.inference_url,
                json=payload,
            )
            response.raise_for_status()
            raw = response.json()
            message = raw["choices"][0]["message"]
            return ChatResponse(
                message=ChatMessage(
                    content=message.get("content"),
                    role=MessageRole(message.get("role")),
                    additional_kwargs=get_additional_kwargs(message, ("content", "role")),
                ),
                raw=raw,
                additional_kwargs=get_additional_kwargs(raw, ("id", "object", "system_fingerprint", "choices")),
            )

    @llm_completion_callback()
    def complete(self, prompt: str, formatted: bool = False, **kwargs: Any) -> CompletionResponse:
        payload = {
            self.prompt_key: prompt,
            "model": self.model,
            "options": self._model_kwargs,
            "stream": False,
            **kwargs,
        }

        if self.json_mode:
            payload["format"] = "json"

        with httpx.Client(timeout=Timeout(self.request_timeout)) as client:
            response = client.post(
                url=self.inference_url,
                json=payload,
            )
            response.raise_for_status()
            raw = response.json()
            text = raw.get("response")
            return CompletionResponse(
                text=text,
                raw=raw,
                additional_kwargs=get_additional_kwargs(raw, ("response",)),
            )

    @llm_completion_callback()
    async def acomplete(self, prompt: str, formatted: bool = False, **kwargs: Any) -> CompletionResponse:
        payload = {
            self.prompt_key: prompt,
            "model": self.model,
            "options": self._model_kwargs,
            "stream": False,
            **kwargs,
        }

        if self.json_mode:
            payload["format"] = "json"

        async with httpx.AsyncClient(timeout=Timeout(self.request_timeout)) as client:
            response = await client.post(
                url=self.inference_url,
                json=payload,
            )
            response.raise_for_status()
            raw = response.json()
            text = raw.get("response")
            return CompletionResponse(
                text=text,
                raw=raw,
                additional_kwargs=get_additional_kwargs(raw, ("response",)),
            )

    @llm_completion_callback()
    def stream_complete(self, prompt: str, formatted: bool = False, **kwargs: Any) -> CompletionResponseGen:
        payload = {
            self.prompt_key: prompt,
            "model": self.model,
            "options": self._model_kwargs,
            "stream": True,
            **kwargs,
        }

        if self.json_mode:
            payload["format"] = "json"

        with httpx.Client(timeout=Timeout(self.request_timeout)) as client:
            with client.stream(
                method="POST",
                url=self.inference_url,
                json=payload,
            ) as response:
                response.raise_for_status()
                text = ""
                for line in response.iter_lines():
                    if line:
                        chunk = json.loads(line)
                        delta = chunk.get("response")
                        text += delta
                        yield CompletionResponse(
                            delta=delta,
                            text=text,
                            raw=chunk,
                            additional_kwargs=get_additional_kwargs(chunk, ("response",)),
                        )

    @llm_completion_callback()
    async def astream_complete(self, prompt: str, formatted: bool = False, **kwargs: Any) -> CompletionResponseAsyncGen:
        payload = {
            self.prompt_key: prompt,
            "model": self.model,
            "options": self._model_kwargs,
            "stream": True,
            **kwargs,
        }

        if self.json_mode:
            payload["format"] = "json"

        async def gen() -> CompletionResponseAsyncGen:
            async with httpx.AsyncClient(timeout=Timeout(self.request_timeout)) as client:
                async with client.stream(
                    method="POST",
                    url=self.inference_url,
                    json=payload,
                ) as response:
                    async for line in response.aiter_text():
                        if line:
                            chunk = json.loads(line)
                            delta = chunk.get("response")
                            yield CompletionResponse(
                                delta=delta,
                                text=delta,
                                raw=chunk,
                                additional_kwargs=get_additional_kwargs(chunk, ("response",)),
                            )

        return gen()

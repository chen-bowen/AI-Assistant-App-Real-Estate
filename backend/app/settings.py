import os
from typing import Dict

from app.clients.llm_client import LLMClient
from llama_index.core.settings import Settings

from app.clients.embeddings_client import EmebeddingsClient


def llm_config_from_env() -> Dict:
    from llama_index.core.constants import DEFAULT_TEMPERATURE

    model = os.getenv("LLM_MODEL")
    temperature = os.getenv("LLM_TEMPERATURE", DEFAULT_TEMPERATURE)
    max_tokens = os.getenv("LLM_MAX_TOKENS")

    config = {
        "model": model,
        "temperature": float(temperature),
        "max_tokens": int(max_tokens) if max_tokens is not None else None,
    }
    return config


def embedding_config_from_env() -> Dict:
    model = os.getenv("EMBEDDING_MODEL")
    dimension = os.getenv("EMBEDDING_DIM")

    config = {
        "model": model,
        "dimension": int(dimension) if dimension is not None else None,
    }
    return config


def init_settings():
    embedding_configs = embedding_config_from_env()
    Settings.embed_model = EmebeddingsClient(**embedding_configs)
    Settings.llm = LLMClient(model=os.getenv("LLM_MODEL"))

    Settings.chunk_size = int(os.getenv("CHROMA_CHUNK_SIZE"))
    Settings.chunk_overlap = int(os.getenv("CHUNK_OVERLAP"))

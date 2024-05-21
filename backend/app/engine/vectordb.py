import os
from urllib.parse import urlparse
import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore


def get_vector_store():
    """
    Initializes and returns a ChromaVectorStore object based on the environment variables.

    Returns:
        ChromaVectorStore: The initialized ChromaVectorStore object.
    """
    # initialize chroma client according to https://docs.llamaindex.ai/en/stable/api_reference/storage/vector_store/chroma/
    client = chromadb.HttpClient(host=os.environ.get("CHROMA_HOST"), port=os.environ.get("CHROMA_PORT"))
    collection = client.create_collection(name=os.environ.get("CHROMA_COLLECTION_NAME"), get_or_create=True)

    return ChromaVectorStore(chroma_collection=collection)

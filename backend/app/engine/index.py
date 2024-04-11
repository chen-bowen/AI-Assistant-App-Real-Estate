import logging

from llama_index.core.indices.vector_store import VectorStoreIndex

from app.engine.utils import init_chroma_vector_store_from_env

logger = logging.getLogger("uvicorn")


def get_index():
    """
    Retrieves the index from ChromaDB.

    Returns:
        VectorStoreIndex: The index object.
    """
    logger.info("Connecting to index from ChromaDB...")
    # Initialize Chroma vector store
    store = init_chroma_vector_store_from_env()
    # Create index from vector store
    index = VectorStoreIndex.from_vector_store(store)
    logger.info("Finished connecting to index from ChromaDB.")  # Connection successful
    return index

import logging

from llama_index.core.indices.vector_store import VectorStoreIndex

from app.engine.vectordb import get_vector_store


logger = logging.getLogger("uvicorn")


def get_index():
    """
    Retrieves the index from ChromaDB.

    Returns:
        VectorStoreIndex: The index object.
    """
    logger.info("Connecting to index from ChromaDB...")
    # Initialize Chroma vector store
    store = get_vector_store()
    # Create index from vector store
    index = VectorStoreIndex.from_vector_store(store)
    logger.info("Finished connecting to index from ChromaDB.")  # Connection successful
    return index

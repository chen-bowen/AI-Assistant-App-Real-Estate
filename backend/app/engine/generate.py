import chunk

from dotenv import load_dotenv

load_dotenv()

import logging
import os

from llama_index.core.indices import VectorStoreIndex
from llama_index.core.storage import StorageContext

from app.engine.loader import get_documents
from app.engine.utils import init_chroma_vector_store_from_env
from app.settings import init_settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()


def generate_datasource():
    logger.info("Creating new index")
    # load the documents and create the index
    documents = get_documents()
    store = init_chroma_vector_store_from_env()
    storage_context = StorageContext.from_defaults(vector_store=store)
    VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        chunk_size=os.getenv("CHROMA_CHUNK_SIZE"),
        show_progress=True,  # this will show you a progress bar as the embeddings are created
    )
    logger.info(f"Successfully created embeddings in the Chroma vector store")


if __name__ == "__main__":
    init_settings()
    generate_datasource()

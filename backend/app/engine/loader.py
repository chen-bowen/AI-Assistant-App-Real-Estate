import os

from llama_index.core import SimpleDirectoryReader
from llama_parse import LlamaParse

DATA_DIR = "data"  # directory containing the documents


def get_documents():
    """
    Retrieves and loads documents from a directory using LlamaParse.

    Returns:
        A list of loaded documents.

    Raises:
        ValueError: If the LLAMA_CLOUD_API_KEY environment variable is not set.
    """
    # LLAMA_CLOUD_API_KEY environment variable must be set in .env file or shell environment
    if os.getenv("LLAMA_CLOUD_API_KEY") is None:
        raise ValueError(
            "LLAMA_CLOUD_API_KEY environment variable is not set. "
            "Please set it in .env file or in your shell environment then run again!"
        )
    # create parser
    parser = LlamaParse(result_type="markdown", verbose=True, language="en")

    # use reader to directly load the data into memory
    reader = SimpleDirectoryReader(DATA_DIR, file_extractor={".pdf": parser})
    documents = reader.load_data()
    return documents

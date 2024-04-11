import asyncio
import os

import requests
from fire import Fire


async def download_pdf_file(url: str) -> bool:
    """Download PDF from given URL to local directory.

    :param url: The url of the PDF file to be downloaded
    :return: True if PDF file was successfully downloaded, otherwise False.
    """

    # Request URL and get response object
    response = requests.get(url, stream=True)

    # isolate PDF filename from URL
    pdf_file_name = os.path.basename(url)
    if response.status_code == 200:
        # Save in current working directory
        filepath = os.path.join("./backend/data", pdf_file_name)
        with open(filepath, "wb") as pdf_object:
            pdf_object.write(response.content)
            print(f"{pdf_file_name} was successfully saved!")
            return True
    else:
        print(f"Uh oh! Could not download {pdf_file_name},")
        print(f"HTTP response status code: {response.status_code}")
        return False


def main_download_single_document(doc_url: str):
    """
    Script to upsert a single document by URL. metada_map parameter will be empty dict ({})
    This script is useful when trying to use your own PDF files.
    """
    asyncio.run(download_pdf_file(doc_url))


if __name__ == "__main__":
    Fire(main_download_single_document)

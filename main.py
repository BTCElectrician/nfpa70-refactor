import os
import logging
import asyncio
from dotenv import load_dotenv
from pathlib import Path

from data_processing.pdf_extractor import PDFExtractor
from data_processing.text_chunker import ElectricalCodeChunker
from azure_storage.blob_manager import BlobStorageManager
from azure_search.index_creator import create_search_index

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def async_chunk_content(pages_text, openai_api_key):
    """
    Runs the asynchronous chunking process with the ElectricalCodeChunker
    so that GPT-based analysis can occur properly.
    """
    async with ElectricalCodeChunker(openai_api_key=openai_api_key) as chunker:
        chunks = await chunker.chunk_nfpa70_content(pages_text)
    return chunks

def main():
    """
    Extracts PDF text, chunks it (including GPT analysis if openai_api_key is set),
    and saves the chunks to blob storage. Optionally creates/updates the search
    index schema. Does NOT actually index documents (see index_from_blob.py).
    """
    try:
        load_dotenv()

        # Required environment variables
        pdf_path = os.getenv('PDF_PATH')
        search_endpoint = os.getenv('AZURE_SEARCH_SERVICE_ENDPOINT')
        search_admin_key = os.getenv('AZURE_SEARCH_ADMIN_KEY')
        index_name = os.getenv('AZURE_SEARCH_INDEX_NAME', 'nfpa70-refactor')
        openai_api_key = os.getenv('OPENAI_API_KEY')

        # Validate required variables
        required = [pdf_path, search_endpoint, search_admin_key, openai_api_key]
        if any(v is None or v.strip() == "" for v in required):
            raise ValueError("One or more required environment variables are missing.")

        # Step 1: Extract PDF text
        extractor = PDFExtractor()
        pages_text = extractor.extract_text_from_pdf(
            pdf_path=Path(pdf_path),
            start_page=26,    # skip everything before page 26
            end_page=868,     # skip everything after page 868
            max_pages=5       # test with 5 pages
        )
        logger.info(f"Extracted text from {len(pages_text)} pages.")

        # Step 2: Run asynchronous chunking
        logger.info("Starting GPT-based chunking of the text...")
        chunks = asyncio.run(async_chunk_content(pages_text, openai_api_key))
        logger.info(f"Created {len(chunks)} text chunks.")

        # Step 3: Convert chunk objects into dictionaries with top-level fields
        logger.info("Converting chunk objects into dictionaries...")
        chunk_dicts = []
        for c in chunks:
            chunk_dicts.append({
                "content": c.content,
                "page_number": c.page_number,
                "article_number": c.article_number,
                "section_number": c.section_number,
                "article_title": c.article_title or "",
                "section_title": c.section_title or "",
                "context_tags": c.context_tags,
                "related_sections": c.related_sections
            })

        # Step 4: Save processed chunks to blob storage
        try:
            blob_manager = BlobStorageManager(container_name="nfpa70-pdf-chunks", blob_name="nfpa70_chunks.json")
            blob_manager.save_processed_data({"chunks": chunk_dicts})
            logger.info("Saved chunked data to blob storage for later indexing.")
        except Exception as e:
            logger.warning(f"Failed to save chunks to blob storage: {e}")

        # Optional: Create or update the search index (schema only, no docs indexed here).
        create_search_index(search_endpoint, search_admin_key, index_name)
        logger.info(f"Search index '{index_name}' created/updated successfully.")

        logger.info("Main process completed successfully. To index the data, run 'index_from_blob.py'.")

    except Exception as e:
        logger.error(f"Error in main process: {str(e)}")
        raise

if __name__ == "__main__":
    main()

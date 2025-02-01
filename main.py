import os
import logging
from dotenv import load_dotenv

from data_processing.pdf_extractor import PDFExtractor
from data_processing.text_chunker import ElectricalCodeChunker
from azure_storage.blob_manager import BlobStorageManager
from azure_search.index_creator import create_search_index

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Extracts PDF text, chunks it, and saves the chunks to blob storage."""
    try:
        load_dotenv()

        # Required environment variables
        pdf_path = os.getenv('PDF_PATH')
        search_endpoint = os.getenv('AZURE_SEARCH_SERVICE_ENDPOINT')
        search_admin_key = os.getenv('AZURE_SEARCH_ADMIN_KEY')
        index_name = os.getenv('AZURE_SEARCH_INDEX_NAME', 'nfpa70-index')
        openai_api_key = os.getenv('OPENAI_API_KEY')

        # Validate required variables
        required = [pdf_path, search_endpoint, search_admin_key, openai_api_key]
        if any(v is None or v.strip() == "" for v in required):
            raise ValueError("One or more required environment variables are missing.")

        # Step 1: Extract PDF text
        extractor = PDFExtractor()
        pages_text = extractor.extract_text_from_pdf(pdf_path)
        logger.info(f"Extracted text from {len(pages_text)} pages.")

        # Step 2: Chunk text
        chunker = ElectricalCodeChunker()
        chunks = chunker.chunk_nfpa70_content(pages_text)
        logger.info(f"Created {len(chunks)} text chunks.")

        # Optional: Save processed chunks to blob storage
        try:
            blob_manager = BlobStorageManager(container_name="processed-data", blob_name="nfpa70_chunks.json")
            # Convert chunk dataclass list to a serializable list of dicts
            chunk_dicts = [
                {
                    "content": c.content,
                    "metadata": {
                        "article": c.article_number,
                        "section": c.section_number,
                        "page": c.page_number
                    },
                    "context_tags": c.context_tags,
                    "related_sections": c.related_sections
                }
                for c in chunks
            ]
            blob_manager.save_processed_data({"chunks": chunk_dicts})
            logger.info("Saved chunked data to blob storage for later indexing.")
        except Exception as e:
            logger.warning(f"Failed to save chunks to blob storage: {e}")

        # Optionally, create or update the search index.
        # This does not perform document indexing—it only sets up the index schema.
        create_search_index(search_endpoint, search_admin_key, index_name)
        logger.info(f"Search index '{index_name}' created/updated successfully.")

        # Note: We are not calling the DataIndexer/index_documents here.
        logger.info("Main process completed without indexing. To index the data, run index_from_blob.py separately.")

    except Exception as e:
        logger.error(f"Error in main process: {str(e)}")
        raise

if __name__ == "__main__":
    main()

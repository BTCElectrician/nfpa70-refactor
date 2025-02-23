import os
import logging
import asyncio
from dotenv import load_dotenv
from pathlib import Path
import json

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

def get_page_sections() -> list[tuple[int, int]]:
    """
    Generate 50-page sections from start to last text-content page (775).
    Returns list of (start_page, end_page) tuples.
    Stops at page 775 to avoid processing table-only sections.
    """
    sections = []
    start_page = 26  # First content page
    while start_page <= 775:  # Last text content page (before tables)
        end_page = min(start_page + 49, 775)  # 50 pages or until end of text
        sections.append((start_page, end_page))
        start_page = end_page + 1
    return sections

async def process_section(
    extractor: PDFExtractor,
    pdf_path: str | Path,
    start_page: int,
    end_page: int,
    openai_api_key: str,
    blob_manager: BlobStorageManager
) -> None:
    """
    Process a single section of the document and save to its own blob file.
    
    Args:
        extractor: PDFExtractor instance
        pdf_path: Path to PDF file
        start_page: Starting page number
        end_page: Ending page number
        openai_api_key: OpenAI API key
        blob_manager: BlobStorageManager instance
    """
    logger.info(f"Processing pages {start_page} to {end_page}")
    
    # Extract text from PDF
    pages_text = extractor.extract_text_from_pdf(
        pdf_path=Path(pdf_path),
        start_page=start_page,
        end_page=end_page,
        max_pages=50  # Maintain your existing setting
    )
    logger.info(f"Extracted text from {len(pages_text)} pages ({start_page}-{end_page})")
    
    # Process chunks
    logger.info("Starting GPT-based chunking of the text...")
    chunks = await async_chunk_content(pages_text, openai_api_key)
    logger.info(f"Created {len(chunks)} text chunks.")
    
    # Convert chunk objects into dictionaries
    logger.info("Converting chunk objects into dictionaries...")
    chunk_dicts = []
    for c in chunks:
        # Debug log the chunk attributes
        logger.debug(f"Processing chunk with attributes:")
        logger.debug(f"  content type: {type(c.content)}")
        logger.debug(f"  page_number type: {type(c.page_number)}")
        logger.debug(f"  article_number type: {type(c.article_number)}")
        logger.debug(f"  context_tags type: {type(c.context_tags)}")
        
        chunk_dict = {
            "content": c.content,
            "page_number": c.page_number,
            "article_number": c.article_number,
            "section_number": c.section_number,
            "article_title": c.article_title or "",
            "section_title": c.section_title or "",
            "context_tags": list(c.context_tags) if c.context_tags else [],
            "related_sections": list(c.related_sections) if c.related_sections else []
        }
        chunk_dicts.append(chunk_dict)

    # Debug: Log the first chunk dict
    if chunk_dicts:
        logger.debug("First chunk dictionary structure:")
        logger.debug(json.dumps(chunk_dicts[0], indent=2, default=str))
    
    # Save this section's chunks to its own blob file
    try:
        data_to_save = {"chunks": chunk_dicts}
        # Debug: Try to serialize before sending to blob storage
        try:
            json.dumps(data_to_save, default=str)
            logger.debug("Data successfully serialized to JSON")
        except Exception as json_error:
            logger.error(f"JSON serialization test failed: {json_error}")
            raise

        # Create a new blob manager for this section
        blob_name = f"nfpa70_chunks_{start_page:03d}_{end_page:03d}.json"
        section_blob_manager = BlobStorageManager(
            container_name="nfpa70-pdf-chunks",
            blob_name=blob_name
        )
        section_blob_manager.save_processed_data(data_to_save)
        logger.info(f"Saved {len(chunk_dicts)} chunks from pages {start_page}-{end_page} to {blob_name}")

    except Exception as e:
        logger.error(f"Failed to save chunks for pages {start_page}-{end_page} to blob storage: {e}")
        raise

async def main():
    """
    Process entire NFPA 70 document, saving each section to its own blob file.
    """
    try:
        load_dotenv()

        # Required environment variables
        pdf_path = os.getenv('PDF_PATH')
        search_endpoint = os.getenv('AZURE_SEARCH_SERVICE_ENDPOINT')
        search_admin_key = os.getenv('AZURE_SEARCH_ADMIN_KEY')
        index_name = os.getenv('AZURE_SEARCH_INDEX_NAME', 'nfpa70-refactor')
        openai_api_key = os.getenv('OPENAI_API_KEY')
        storage_connection_string = os.getenv('AZURE_STORAGE_CONNECTION_STRING')

        # Validate required variables
        required = [
            pdf_path,
            search_endpoint,
            search_admin_key,
            openai_api_key,
            storage_connection_string
        ]
        if any(v is None or v.strip() == "" for v in required):
            raise ValueError("One or more required environment variables are missing.")

        # Initialize services
        extractor = PDFExtractor()
        blob_manager = BlobStorageManager(container_name="nfpa70-pdf-chunks")
        
        # Get all page sections to process
        sections = get_page_sections()
        logger.info(f"Starting processing of {len(sections)} sections")
        
        # Process each section
        for start_page, end_page in sections:
            try:
                await process_section(
                    extractor=extractor,
                    pdf_path=pdf_path,
                    start_page=start_page,
                    end_page=end_page,
                    openai_api_key=openai_api_key,
                    blob_manager=blob_manager
                )
            except Exception as e:
                logger.error(f"Error processing section {start_page}-{end_page}: {str(e)}")
                raise

        # Create or update the search index schema
        create_search_index(search_endpoint, search_admin_key, index_name)
        logger.info(f"Search index '{index_name}' created/updated successfully.")

        logger.info("Full document processing completed successfully")
        logger.info("To index the data, run 'index_from_blob.py'.")

    except Exception as e:
        logger.error(f"Error in main process: {str(e)}")
        raise

if __name__ == "__main__":
    asyncio.run(main())
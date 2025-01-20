from pathlib import Path
from typing import Final
from collections.abc import Sequence
from os import getenv
from dotenv import load_dotenv
import logging

from data_processing.pdf_extractor import PDFExtractor
from data_processing.text_chunker import ElectricalCodeChunker
from azure_search.index_creator import create_search_index
from azure_search.data_indexer import DataIndexer

# Set up logging with modern configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger: Final = logging.getLogger(__name__)

# Define required environment variables as constants
REQUIRED_VARS: Final[Sequence[str]] = (
    'PDF_PATH',
    'AZURE_SEARCH_SERVICE_ENDPOINT',
    'AZURE_SEARCH_ADMIN_KEY',
    'OPENAI_API_KEY'
)

def main() -> None:
    """Main process to extract, process, and index electrical code content."""
    try:
        # Load environment variables
        load_dotenv()
        
        # Get environment variables with type hints
        pdf_path: str | None = getenv('PDF_PATH')
        search_service_endpoint: str | None = getenv('AZURE_SEARCH_SERVICE_ENDPOINT')
        search_admin_key: str | None = getenv('AZURE_SEARCH_ADMIN_KEY')
        index_name: str = getenv('AZURE_SEARCH_INDEX_NAME', 'nfpa70-refactor')
        openai_api_key: str | None = getenv('OPENAI_API_KEY')

        # Validate environment variables
        missing_vars = [var for var in REQUIRED_VARS if not getenv(var)]
        if missing_vars:
            raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")

        # Convert string path to Path object and validate
        pdf_file = Path(pdf_path)
        if not pdf_file.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_file}")

        # Extract text from PDF with advanced cleaning
        logger.info("Extracting text from PDF...")
        pdf_extractor = PDFExtractor()
        pages_text = pdf_extractor.extract_text_from_pdf(pdf_file)
        logger.info(f"Successfully extracted text from {len(pages_text)} pages")

        # Process text into context-aware chunks
        logger.info("Processing text into chunks with electrical context...")
        chunker = ElectricalCodeChunker()
        chunks = chunker.chunk_nfpa70_content(pages_text)
        logger.info(f"Created {len(chunks)} context-aware chunks")

        # Create or update search index with enhanced schema
        logger.info("Creating search index...")
        create_search_index(search_service_endpoint, search_admin_key, index_name)

        # Index the documents with embeddings and context tags
        logger.info("Indexing documents...")
        indexer = DataIndexer(
            service_endpoint=search_service_endpoint,
            admin_key=search_admin_key,
            index_name=index_name,
            openai_api_key=openai_api_key
        )
        indexer.index_documents(chunks)

        logger.info("Process completed successfully!")

    except Exception as e:
        logger.error(f"Error in main process: {str(e)}")
        raise

if __name__ == "__main__":
    main() 
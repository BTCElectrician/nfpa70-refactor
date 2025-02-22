import os
import sys
import asyncio
from pathlib import Path
from dotenv import load_dotenv
import httpx
from openai import AsyncOpenAI
from loguru import logger
from tqdm import tqdm

# Add project root to Python path
project_root = str(Path(__file__).resolve().parents[1])
sys.path.append(project_root)

from data_processing.pdf_extractor import PDFExtractor
from data_processing.text_chunker import ElectricalCodeChunker
from azure_storage.blob_manager import BlobStorageManager
from data_processing.models import ChunkBatch

# Configure loguru
logger.remove()
logger.add(sys.stderr, level="DEBUG", format="{time} - {name} - {level} - {message}")

# Initialize OpenAI client with enhanced HTTPX configuration
def get_openai_client(api_key: str) -> AsyncOpenAI:
    # Enhanced timeout configuration
    timeout = httpx.Timeout(
        connect=10.0,    # Connection timeout
        read=60.0,       # Read timeout
        write=60.0,      # Write timeout
        pool=60.0        # Pool timeout
    )
    
    # Enhanced connection limits
    limits = httpx.Limits(
        max_connections=100,              # Increased for better concurrency
        max_keepalive_connections=20,     # Optimal keepalive connections
        keepalive_expiry=30.0            # Connection expiry time
    )
    
    # Create HTTP client with enhanced configuration
    http_client = httpx.AsyncClient(
        timeout=timeout,
        limits=limits,
        http2=True  # Enable HTTP/2 for better performance
    )
    
    return AsyncOpenAI(
        api_key=api_key,
        http_client=http_client,
        max_retries=5
    )

async def reprocess_section() -> None:
    """Reprocess PDF pages 66-75 with single-batch processing."""
    try:
        load_dotenv()
        pdf_path = os.getenv('PDF_PATH')
        openai_api_key = os.getenv('OPENAI_API_KEY')
        
        if not all([pdf_path, openai_api_key]):
            raise ValueError("Missing required environment variables (PDF_PATH, OPENAI_API_KEY)")

        extractor = PDFExtractor()
        client = get_openai_client(openai_api_key)

        # Keep existing page range for testing
        start_page = 66  # PDF page
        end_page = 75    # PDF page
        
        logger.info(f"Extracting PDF pages {start_page} to {end_page}...")
        pages_text = extractor.extract_text_from_pdf(
            pdf_path=Path(pdf_path),
            start_page=start_page,
            end_page=end_page,
            max_pages=10
        )
        
        logger.info(f"Extracted {len(pages_text)} pages with PDF numbers: {list(pages_text.keys())}")
        
        # Process all pages in one batch
        async with ElectricalCodeChunker(openai_api_key=openai_api_key) as chunker:
            # Estimate chunks based on content density
            estimated_chunks = len(pages_text) * 5  # Average 5 sections per page
            chunks = []
            
            with tqdm(total=estimated_chunks, desc="Processing chunks", file=sys.stderr) as pbar:
                # Process entire batch at once
                batch_chunks = await chunker.chunk_nfpa70_content(pages_text)
                chunks.extend(batch_chunks)
                pbar.update(len(batch_chunks))
                logger.debug(f"Processed {len(batch_chunks)} chunks in batch")

            # Log chunks with NFPA page numbers for verification
            for i, chunk in enumerate(chunks):
                logger.debug(f"Chunk {i}: Page {chunk.page_number}, "
                           f"Article {chunk.article_number}, Section {chunk.section_number}")
                logger.debug(f"  Content: {chunk.content[:100]}...")

            # Create batch for storage
            batch = ChunkBatch(chunks=chunks)
            
            # Use NFPA range for blob name (PDF page - 3)
            start_nfpa = start_page - 3
            end_nfpa = end_page - 3
            blob_name = f"nfpa70_chunks_{start_nfpa:03d}_{end_nfpa:03d}.json"
            
            blob_manager = BlobStorageManager(
                container_name="nfpa70-pdf-chunks",
                blob_name=blob_name
            )
            
            data_to_save = batch.to_storage_format()
            blob_manager.save_processed_data(data_to_save)
            logger.info(f"Successfully saved {len(chunks)} chunks to {blob_name}")

    except Exception as e:
        logger.error(f"Error reprocessing section: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    logger.debug("Starting reprocess_sections.py")
    asyncio.run(reprocess_section())
    logger.debug("Completed reprocess_sections.py")
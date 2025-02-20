import os
import sys
import logging
import asyncio
from pathlib import Path
from dotenv import load_dotenv

# Add project root to Python path
project_root = str(Path(__file__).resolve().parents[1])
sys.path.append(project_root)

from data_processing.pdf_extractor import PDFExtractor
from data_processing.text_chunker import ElectricalCodeChunker
from azure_storage.blob_manager import BlobStorageManager

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def reprocess_section():
    """Reprocess pages 25-75 and save to blob storage."""
    try:
        load_dotenv()

        # Validate environment variables
        pdf_path = os.getenv('PDF_PATH')
        openai_api_key = os.getenv('OPENAI_API_KEY')
        
        if not all([pdf_path, openai_api_key]):
            raise ValueError("Missing required environment variables (PDF_PATH, OPENAI_API_KEY)")

        # Initialize components
        extractor = PDFExtractor()
        
        # Extract specified pages
        logger.info("Extracting pages 25-75...")
        pages_text = extractor.extract_text_from_pdf(
            pdf_path=Path(pdf_path),
            start_page=25,
            end_page=75
        )
        logger.info(f"Extracted {len(pages_text)} pages")

        # Process chunks
        logger.info("Processing chunks with GPT analysis...")
        async with ElectricalCodeChunker(openai_api_key=openai_api_key) as chunker:
            chunks = await chunker.chunk_nfpa70_content(pages_text)
            logger.info(f"Created {len(chunks)} chunks")

            # Convert chunks to dictionary format
            chunk_dicts = [{
                "content": c.content,
                "page_number": c.page_number,
                "article_number": c.article_number,
                "section_number": c.section_number,
                "article_title": c.article_title or "",
                "section_title": c.section_title or "",
                "context_tags": list(c.context_tags),
                "related_sections": list(c.related_sections)
            } for c in chunks]

            # Save to blob storage
            blob_name = "nfpa70_chunks_025_075.json"
            blob_manager = BlobStorageManager(
                container_name="nfpa70-pdf-chunks",
                blob_name=blob_name
            )
            
            data_to_save = {"chunks": chunk_dicts}
            blob_manager.save_processed_data(data_to_save)
            logger.info(f"Successfully saved {len(chunks)} chunks to {blob_name}")

    except Exception as e:
        logger.error(f"Error reprocessing section: {str(e)}")
        raise

if __name__ == "__main__":
    asyncio.run(reprocess_section())
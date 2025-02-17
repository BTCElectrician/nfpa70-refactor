import asyncio
import os
from pathlib import Path
from dotenv import load_dotenv
from loguru import logger

from data_processing.text_chunker import ElectricalCodeChunker
from data_processing.pdf_extractor import PDFExtractor
from azure_storage.blob_manager import BlobStorageManager

async def process_definitions():
    """Process Article 100 definitions using existing chunker."""
    try:
        load_dotenv()
        
        # Initialize components
        pdf_extractor = PDFExtractor()
        blob_manager = BlobStorageManager(
            container_name="nfpa70-pdf-chunks",
            blob_name="nfpa70_chunks_026_075.json"  # Will overwrite existing file
        )
        
        # Extract definitions pages
        pdf_path = os.getenv('PDF_PATH')
        pages_text = pdf_extractor.extract_text_from_pdf(
            Path(pdf_path),
            start_page=26,
            end_page=75
        )
        
        # Use existing chunker with OpenAI
        openai_key = os.getenv('OPENAI_API_KEY')
        async with ElectricalCodeChunker(openai_api_key=openai_key) as chunker:
            chunks = await chunker.chunk_nfpa70_content(pages_text)
            
            # Save to blob storage
            data_to_save = {
                "chunks": [{
                    "content": c.content,
                    "page_number": c.page_number,
                    "article_number": c.article_number,
                    "section_number": c.section_number,
                    "article_title": c.article_title,
                    "section_title": c.section_title,
                    "context_tags": list(c.context_tags),
                    "related_sections": list(c.related_sections)
                } for c in chunks]
            }
            blob_manager.save_processed_data(data_to_save)
            
        logger.info(f"Successfully processed {len(chunks)} chunks")
        
    except Exception as e:
        logger.error(f"Error processing definitions: {str(e)}")
        raise

if __name__ == "__main__":
    asyncio.run(process_definitions())
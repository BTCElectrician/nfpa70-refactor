import os
import sys
import asyncio
import json
from pathlib import Path
from dotenv import load_dotenv
import httpx
from openai import AsyncOpenAI
from loguru import logger
from tenacity import AsyncRetrying, stop_after_attempt, retry_if_exception_type, wait_exponential

# Add project root to Python path
project_root = str(Path(__file__).resolve().parents[1])
sys.path.append(project_root)

from data_processing.pdf_extractor import PDFExtractor
from data_processing.text_chunker import ElectricalCodeChunker

# Configure loguru
logger.remove()
logger.add(sys.stderr, level="DEBUG", format="{time} - {name} - {level} - {message}")

def get_openai_client(api_key: str) -> AsyncOpenAI:
    return AsyncOpenAI(
        api_key=api_key,
        http_client=httpx.AsyncClient(
            timeout=httpx.Timeout(60.0),
            limits=httpx.Limits(max_connections=10, max_keepalive_connections=5)
        ),
        max_retries=5
    )

async def test_single_page() -> None:
    """Test GPT chunking on PDF page 67 (NFPA page 70-64)."""
    try:
        load_dotenv()
        pdf_path = os.getenv('PDF_PATH')
        openai_api_key = os.getenv('OPENAI_API_KEY')
        
        if not all([pdf_path, openai_api_key]):
            raise ValueError("Missing required environment variables (PDF_PATH, OPENAI_API_KEY)")

        # Extract page 67
        extractor = PDFExtractor()
        logger.info(f"Extracting PDF page 67...")
        pages_text = extractor.extract_text_from_pdf(
            pdf_path=Path(pdf_path),
            start_page=67,
            end_page=67,
            max_pages=1
        )
        
        if not pages_text:
            logger.error("No text extracted from PDF")
            return
            
        logger.info(f"Extracted {len(pages_text)} pages")

        # Process with GPT using existing chunker
        logger.info("Processing with hybrid chunking...")
        async with ElectricalCodeChunker(openai_api_key=openai_api_key) as chunker:
            try:
                chunks = await chunker.chunk_nfpa70_content({67: pages_text[67]})
                
                # Convert chunks to JSON-serializable format
                chunk_data = []
                for chunk in chunks:
                    chunk_dict = {
                        "article_number": chunk.article_number,
                        "section_number": chunk.section_number,
                        "page_number": chunk.page_number,
                        "article_title": chunk.article_title,
                        "section_title": chunk.section_title,
                        "content": chunk.content,
                        "context_tags": list(chunk.context_tags) if chunk.context_tags else [],
                        "related_sections": list(chunk.related_sections) if chunk.related_sections else []
                    }
                    chunk_data.append(chunk_dict)

                # Save to JSON file
                output_dir = Path("chunked_output")
                output_dir.mkdir(exist_ok=True)
                
                output_file = output_dir / "page_67_chunks.json"
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump({"chunks": chunk_data}, f, indent=2, ensure_ascii=False)
                
                logger.info(f"Successfully saved {len(chunks)} chunks to {output_file}")
                    
            except Exception as e:
                logger.error(f"Error processing chunks: {str(e)}")
                raise
                
    except Exception as e:
        logger.error(f"Error in test: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    logger.debug("Starting test_chunking.py")
    asyncio.run(test_single_page())
    logger.debug("Completed test_chunking.py")
import os
import json
from pathlib import Path
from dotenv import load_dotenv
from loguru import logger
import asyncio  # CHANGED: Replaced 'import time' with asyncio

from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery
from azure.core.credentials import AzureKeyCredential

from data_processing.pdf_extractor import PDFExtractor
from data_processing.text_chunker import ElectricalCodeChunker
from azure_search.index_creator import create_search_index
from azure_search.data_indexer import DataIndexer

def setup_logging():
    """Configure detailed logging for testing."""
    logger.remove()
    logger.add(
        "debug.log",
        format="{time} | {level} | {module}:{function}:{line} - {message}",
        level="DEBUG",
        rotation="1 day"
    )
    logger.add(
        lambda msg: print(msg),
        format="{time:HH:mm:ss} | {level} | {message}",
        level="INFO"
    )

def validate_environment():
    """Validate all required environment variables are present."""
    required_vars = {
        'PDF_PATH': 'Path to the PDF file',
        'AZURE_SEARCH_SERVICE_ENDPOINT': 'Azure Search service endpoint',
        'AZURE_SEARCH_ADMIN_KEY': 'Azure Search admin key',
        'OPENAI_API_KEY': 'OpenAI API key for embeddings'
    }
    
    missing = []
    for var, description in required_vars.items():
        if not os.getenv(var):
            missing.append(f"{var} ({description})")
    
    if missing:
        raise ValueError(f"Missing required environment variables: {', '.join(missing)}")

async def test_single_chunk_processing():  # CHANGED: Added async
    """Test the processing pipeline with enhanced logging."""
    try:
        setup_logging()
        logger.info("Starting test processing with vector validation")
        
        load_dotenv()
        validate_environment()
        
        # Load environment variables
        pdf_path = os.getenv('PDF_PATH')
        search_endpoint = os.getenv('AZURE_SEARCH_SERVICE_ENDPOINT')
        search_key = os.getenv('AZURE_SEARCH_ADMIN_KEY')
        openai_key = os.getenv('OPENAI_API_KEY')
        test_index_name = os.getenv('AZURE_SEARCH_INDEX_NAME', 'nfpa70-test')

        # Extract PDF text
        logger.info("=== PDF EXTRACTION START ===")
        extractor = PDFExtractor()
        pages_text = {}
        
        full_text = extractor.extract_text_from_pdf(Path(pdf_path))
        # Process pages from beginning, middle, and end of the document
        for page_num in range(22, 32):              # Beginning (10 pages)
            if page_num in full_text:
                pages_text[page_num] = full_text[page_num]
        for page_num in range(150, 160):            # Middle (10 pages)
            if page_num in full_text:
                pages_text[page_num] = full_text[page_num]
        for page_num in range(250, 260):            # End (10 pages)
            if page_num in full_text:
                pages_text[page_num] = full_text[page_num]
        
        if not pages_text:
            logger.error("Failed to get pages")
            return

        logger.info(f"Successfully extracted pages: {list(pages_text.keys())}")
        
        # Log raw page content
        logger.info("=== RAW PAGE CONTENT START ===")
        for page_num in sorted(pages_text.keys()):
            first_line = pages_text[page_num].split('\n')[0]
            logger.debug(f"Page {page_num} | First line: {first_line}")
        logger.info("=== RAW PAGE CONTENT END ===")

        # Process chunks
        logger.info("=== CHUNKS PROCESSING START ===")
        async with ElectricalCodeChunker(openai_api_key=openai_key) as chunker:  # CHANGED: Added async context manager
            chunks = await chunker.chunk_nfpa70_content(pages_text)
            logger.info(f"Created {len(chunks)} chunks")

            # Log chunk details
            for i, chunk in enumerate(chunks):
                first_line = chunk.content.split('\n')[0]
                logger.debug(
                    f"Chunk {i} | "
                    f"Page: {chunk.page_number} | "
                    f"Article: {chunk.article_number} | "
                    f"Section: {chunk.section_number} | "
                    f"First line: {first_line}"
                )
                logger.debug(f"Chunk {i} | Page: {chunk.page_number} | Article: {chunk.article_number} | " 
                             f"Section: {chunk.section_number} | First line: {first_line}")
        logger.info("=== CHUNKS PROCESSING END ===")

        # Set up search index
        logger.info(f"=== SEARCH INDEX SETUP START ===")
        create_search_index(search_endpoint, search_key, test_index_name)

        # Initialize indexer
        indexer = DataIndexer(
            service_endpoint=search_endpoint,
            admin_key=search_key,
            index_name=test_index_name,
            openai_api_key=openai_key
        )

        if not chunks:
            logger.warning("No chunks were created")
            return

        # Process and index chunks
        chunk_dicts = [{
            "content": chunk.content,
            "page_number": chunk.page_number,
            "article_number": chunk.article_number or "",
            "section_number": chunk.section_number or "",
            "article_title": chunk.article_title or "",
            "section_title": chunk.section_title or "",
            "context_tags": list(chunk.context_tags),
            "related_sections": list(chunk.related_sections)
        } for chunk in chunks]
        
        logger.info(f"Preparing to index {len(chunk_dicts)} chunks...")
        
        # Index chunks
        indexer.index_documents(chunk_dicts)
        logger.info(f"Successfully indexed {len(chunk_dicts)} chunks")
        logger.info("=== SEARCH INDEX SETUP END ===")

        # Verify indexing
        logger.info("=== SEARCH VERIFICATION START ===")
        search_client = SearchClient(
            endpoint=search_endpoint,
            index_name=test_index_name,
            credential=AzureKeyCredential(search_key)
        )
        
        # Generate embedding for search
        vector = indexer.generate_embeddings(chunk_dicts[0]["content"])
        logger.debug(f"Generated vector of length: {len(vector)}")
        
        # Create vector query
        vector_query = VectorizedQuery(
            vector=vector,
            k_nearest_neighbors=5,
            fields="content_vector"
        )

        # Wait for index update
        logger.info("Waiting for index update...")
        await asyncio.sleep(3)  # CHANGED: Using asyncio.sleep instead of time.sleep

        # Search for indexed content
        results = search_client.search(
            search_text=None,
            vector_queries=[vector_query],
            select=["id", "content", "page_number", "article_number", "section_number"]
        )

        results_list = list(results)
        logger.info(f"Found {len(results_list)} documents in test index")

        # Log search results
        if results_list:
            logger.info("=== SEARCH RESULTS START ===")
            for i, result in enumerate(results_list):
                first_line = result.get('content', '').split('\n')[0]
                logger.debug(
                    f"Result {i} | "
                    f"Page: {result.get('page_number')} | "
                    f"Article: {result.get('article_number')} | "
                    f"Section: {result.get('section_number')} | "
                    f"First line: {first_line}"
                )
                logger.debug(f"Result {i} | Page: {result.get('page_number')} | Article: {result.get('article_number')} | "
                             f"Section: {result.get('section_number')} | First line: {first_line}")
            logger.info("=== SEARCH RESULTS END ===")
            logger.info("Test completed successfully!")
        else:
            logger.warning("No documents found in index after upload")
        logger.info("=== SEARCH VERIFICATION END ===")

    except Exception as e:
        logger.error(f"Test failed: {str(e)}")
        raise

async def main():  # CHANGED: Added main function
    """Main entry point with proper async cleanup."""
    try:
        await test_single_chunk_processing()
    finally:
        pending = asyncio.all_tasks() - {asyncio.current_task()}
        for task in pending:
            task.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)

if __name__ == "__main__":
    asyncio.run(main())  # CHANGED: Using asyncio.run instead of direct function call
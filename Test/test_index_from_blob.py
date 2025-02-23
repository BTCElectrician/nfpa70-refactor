import os
import json
from pathlib import Path
from dotenv import load_dotenv
from loguru import logger
import asyncio

from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery
from azure.core.credentials import AzureKeyCredential

from azure_storage.blob_manager import BlobStorageManager
from azure_search.data_indexer import DataIndexer
from azure_search.index_creator import create_search_index

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
        'AZURE_SEARCH_SERVICE_ENDPOINT': 'Azure Search service endpoint',
        'AZURE_SEARCH_ADMIN_KEY': 'Azure Search admin key',
        'AZURE_SEARCH_INDEX_NAME': 'Azure Search index name',
        'OPENAI_API_KEY': 'OpenAI API key',
        'AZURE_STORAGE_CONNECTION_STRING': 'Azure Storage connection string'
    }
    
    missing = []
    for var, description in required_vars.items():
        if not os.getenv(var):
            missing.append(f"{var} ({description})")
    
    if missing:
        raise ValueError(f"Missing required environment variables: {', '.join(missing)}")

async def test_index_from_blob():
    """Test the indexing pipeline with real data from blob storage."""
    try:
        setup_logging()
        logger.info("Starting blob to index pipeline test")
        
        load_dotenv()
        validate_environment()
        
        # Load environment variables
        search_endpoint = os.getenv('AZURE_SEARCH_SERVICE_ENDPOINT')
        search_key = os.getenv('AZURE_SEARCH_ADMIN_KEY')
        openai_key = os.getenv('OPENAI_API_KEY')
        test_index_name = os.getenv('AZURE_SEARCH_INDEX_NAME', 'nfpa70-refactor-simp')

        # Load actual data from blob storage - first two chunks
        logger.info("=== LOADING FROM BLOB STORAGE ===")
        
        all_chunks = []
        file_names = [
            "nfpa70_chunks_026_075.json",  # First chunk (pages 26-75)
            "nfpa70_chunks_076_125.json"   # Second chunk (pages 76-125)
        ]
        
        for blob_name in file_names:
            blob_manager = BlobStorageManager(
                container_name="nfpa70-refactor-simp",
                blob_name=blob_name
            )
            
            blob_data = blob_manager.load_processed_data()
            chunks = blob_data.get("chunks", [])
            if chunks:
                all_chunks.extend(chunks)
                logger.info(f"Loaded {len(chunks)} chunks from {blob_name}")
        
        if not all_chunks:
            raise ValueError("No chunks found in blob storage")
            
        logger.info(f"Successfully loaded {len(all_chunks)} total chunks from blob storage")
        logger.debug(f"First chunk section number: {all_chunks[0].get('section_number', 'N/A')}")
        
        # Set up search index
        logger.info("=== SEARCH INDEX SETUP ===")
        create_search_index(search_endpoint, search_key, test_index_name)

        # Initialize indexer
        indexer = DataIndexer(
            service_endpoint=search_endpoint,
            admin_key=search_key,
            index_name=test_index_name,
            openai_api_key=openai_key
        )

        # Index the documents
        logger.info(f"Indexing {len(all_chunks)} chunks...")
        indexer.index_documents(all_chunks)
        
        # Verify indexing
        logger.info("=== VERIFYING SEARCH INDEX ===")
        search_client = SearchClient(
            endpoint=search_endpoint,
            index_name=test_index_name,
            credential=AzureKeyCredential(search_key)
        )
        
        # Generate vector for first chunk to test search
        vector = indexer.generate_embeddings(all_chunks[0]["content"])
        logger.debug(f"Generated test vector of length: {len(vector)}")
        
        vector_query = VectorizedQuery(
            vector=vector,
            k_nearest_neighbors=5,
            fields="content_vector"
        )

        # Wait for index update
        logger.info("Waiting for index update...")
        await asyncio.sleep(3)

        # Search for indexed content
        results = search_client.search(
            search_text=None,
            vector_queries=[vector_query],
            select=["id", "content", "page_number", "article_number", "section_number"]
        )

        results_list = list(results)
        logger.info(f"Found {len(results_list)} documents in test index")

        if results_list:
            logger.info("=== SEARCH RESULTS ===")
            for i, result in enumerate(results_list[:2]):  # Show first 2 results
                logger.info(
                    f"Result {i} | "
                    f"Page: {result.get('page_number')} | "
                    f"Article: {result.get('article_number')} | "
                    f"Section: {result.get('section_number')}"
                )
            logger.info("Test completed successfully!")
        else:
            logger.error("No documents found in index after upload")
            raise ValueError("Index verification failed - no documents found")

    except Exception as e:
        logger.error(f"Test failed: {str(e)}")
        raise

async def main():
    """Main entry point with proper async cleanup."""
    try:
        await test_index_from_blob()
    finally:
        pending = asyncio.all_tasks() - {asyncio.current_task()}
        for task in pending:
            task.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)

if __name__ == "__main__":
    asyncio.run(main())
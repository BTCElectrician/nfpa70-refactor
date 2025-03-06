import os
import logging
import asyncio
from dotenv import load_dotenv

from azure_storage.blob_manager import BlobStorageManager
from azure_search.data_indexer import DataIndexer
from azure_search.index_creator import create_search_index

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def main():
    """
    Loads pre-chunked data from all JSON files in blob storage and indexes them into Azure Cognitive Search.
    """
    load_dotenv()
    
    # Load environment variables for indexing
    search_endpoint = os.getenv('AZURE_SEARCH_SERVICE_ENDPOINT')
    search_admin_key = os.getenv('AZURE_SEARCH_ADMIN_KEY')
    index_name = os.getenv('AZURE_SEARCH_INDEX_NAME', 'nfpa70-refactor-simp')
    openai_api_key = os.getenv('OPENAI_API_KEY')

    if not all([search_endpoint, search_admin_key, openai_api_key]):
        logger.error("Missing required environment variables. Exiting.")
        return

    try:
        # Ensure search index exists with correct schema
        logger.info("Verifying search index structure...")
        create_search_index(search_endpoint, search_admin_key, index_name)
        
        # Initialize BlobStorageManager
        logger.info("Initializing blob storage manager...")
        blob_manager = BlobStorageManager(container_name="nfpa70-refactor-simp")
        
        # List all blobs starting with "nfpa70_chunks_"
        logger.info("Listing all chunk files in blob storage...")
        blob_list = blob_manager.container_client.list_blobs(name_starts_with="nfpa70_chunks_")
        
        # Initialize DataIndexer with context manager
        async with DataIndexer(
            service_endpoint=search_endpoint,
            admin_key=search_admin_key,
            index_name=index_name,
            openai_api_key=openai_api_key
        ) as indexer:
            try:
                total_chunks_indexed = 0
                
                # Process each blob
                for blob_item in blob_list:
                    blob_name = blob_item.name
                    logger.info(f"Processing blob: {blob_name}")
                    
                    # Update blob_manager to point to this specific blob
                    blob_manager.blob_name = blob_name
                    
                    # Load chunked data from this blob
                    blob_data = blob_manager.load_processed_data()
                    chunks = blob_data.get("chunks", [])
                    
                    if not chunks:
                        logger.warning(f"No chunk data found in {blob_name}. Skipping.")
                        continue
                    
                    logger.info(f"Loaded {len(chunks)} chunks from {blob_name}")
                    
                    # Index the chunks asynchronously
                    logger.info(f"Indexing {len(chunks)} chunks from {blob_name}...")
                    await indexer.index_documents(chunks)
                    total_chunks_indexed += len(chunks)
                    logger.info(f"Successfully indexed chunks from {blob_name}")
                
                logger.info(f"Indexing completed. Total chunks indexed: {total_chunks_indexed}")
            except Exception as e:
                logger.error(f"Error during blob processing: {str(e)}")
                # Uncomment below if you want exceptions to propagate
                # raise

    except Exception as e:
        logger.error(f"Error during indexing process: {str(e)}")
        raise

if __name__ == "__main__":
    asyncio.run(main())
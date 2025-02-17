import os
import logging
from dotenv import load_dotenv

from azure_storage.blob_manager import BlobStorageManager
from azure_search.data_indexer import DataIndexer
from azure_search.index_creator import create_search_index

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """
    Loads the pre-chunked data from blob storage and indexes it into Azure Cognitive Search.
    """
    load_dotenv()
    
    # Load environment variables for indexing
    search_endpoint = os.getenv('AZURE_SEARCH_SERVICE_ENDPOINT')
    search_admin_key = os.getenv('AZURE_SEARCH_ADMIN_KEY')
    index_name = os.getenv('AZURE_SEARCH_INDEX_NAME', 'nfpa70-index')
    openai_api_key = os.getenv('OPENAI_API_KEY')

    try:
        # Ensure search index exists with correct schema
        logger.info("Verifying search index structure...")
        create_search_index(search_endpoint, search_admin_key, index_name)
        
        # Step 1: Load chunked data from Blob Storage
        logger.info("Loading data from blob storage...")
        blob_manager = BlobStorageManager(container_name="nfpa70-pdf-chunks", blob_name="nfpa70_chunks.json")
        blob_data = blob_manager.load_processed_data()
        chunks = blob_data.get("chunks", [])

        if not chunks:
            logger.error("No chunk data found in blob storage. Exiting.")
            return
        
        logger.info(f"Successfully loaded {len(chunks)} chunks from blob storage")
        
        # Step 2: Index documents using DataIndexer
        logger.info("Initializing indexer...")
        indexer = DataIndexer(
            service_endpoint=search_endpoint,
            admin_key=search_admin_key,
            index_name=index_name,
            openai_api_key=openai_api_key
        )
        
        logger.info("Starting document indexing...")
        indexer.index_documents(chunks)
        logger.info("Indexing from blob storage completed successfully.")

    except Exception as e:
        logger.error(f"Error during indexing process: {str(e)}")
        raise

if __name__ == "__main__":
    main()
import os
import logging
from dotenv import load_dotenv

from azure_storage.blob_manager import BlobStorageManager
from azure_search.data_indexer import DataIndexer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    load_dotenv()
    
    # Load environment variables for indexing
    search_endpoint = os.getenv('AZURE_SEARCH_SERVICE_ENDPOINT')
    search_admin_key = os.getenv('AZURE_SEARCH_ADMIN_KEY')
    index_name = os.getenv('AZURE_SEARCH_INDEX_NAME', 'nfpa70-index')
    openai_api_key = os.getenv('OPENAI_API_KEY')
    
    # Load chunked data from Blob Storage
    blob_manager = BlobStorageManager(container_name="processed-data", blob_name="nfpa70_chunks.json")
    blob_data = blob_manager.load_processed_data()
    chunks = blob_data.get("chunks", [])
    
    if not chunks:
        logger.error("No chunk data found in blob storage. Exiting.")
        return
    
    # Index documents using DataIndexer
    indexer = DataIndexer(
        service_endpoint=search_endpoint,
        admin_key=search_admin_key,
        index_name=index_name,
        openai_api_key=openai_api_key
    )
    indexer.index_documents(chunks)
    logger.info("Indexing from blob storage completed successfully.")

if __name__ == "__main__":
    main()

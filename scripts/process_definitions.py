import asyncio
import os
from pathlib import Path
from dotenv import load_dotenv
from loguru import logger

from data_processing.definitions_chunker import DefinitionsChunker
from data_processing.pdf_extractor import PDFExtractor
from azure_storage.blob_manager import BlobStorageManager
from azure_search.definitions_index import create_definitions_index
from azure_search.definitions_indexer import DefinitionsIndexer  # Changed import

async def process_definitions():
    """Process Article 100 definitions using specialized definitions chunker."""
    try:
        load_dotenv()
        
        # Initialize components
        pdf_extractor = PDFExtractor()
        blob_manager = BlobStorageManager(
            container_name="nfpa70-pdf-chunks",
            blob_name="nfpa70_article100_definitions.json"  # Separate blob for definitions
        )
        
        # Extract Article 100 pages (pages 26-75 based on your comments)
        logger.info("Extracting Article 100 definitions...")
        pdf_path = os.getenv('PDF_PATH')
        pages_text = pdf_extractor.extract_text_from_pdf(
            Path(pdf_path),
            start_page=26,
            end_page=75
        )
        logger.info(f"Extracted {len(pages_text)} pages of definitions")
        
        # Set up specialized definitions index
        search_endpoint = os.getenv('AZURE_SEARCH_SERVICE_ENDPOINT')
        search_admin_key = os.getenv('AZURE_SEARCH_ADMIN_KEY')
        definitions_index_name = os.getenv('AZURE_SEARCH_DEFINITIONS_INDEX', 'nfpa70-definitions')
        
        logger.info("Creating specialized definitions index...")
        create_definitions_index(search_endpoint, search_admin_key, definitions_index_name)
        
        # Process definitions using specialized chunker
        openai_key = os.getenv('OPENAI_API_KEY')
        async with DefinitionsChunker(openai_api_key=openai_key) as chunker:
            logger.info("Processing definitions with specialized chunker...")
            definitions = await chunker.process_article_100(pages_text)
            
            # Convert definitions to dictionary format for storage and indexing
            definitions_data = {
                "definitions": [{
                    "term": d.term,
                    "definition": d.definition,
                    "page_number": d.page_number,
                    "context": d.context or "",
                    "cross_references": d.cross_references,
                    "info_notes": d.info_notes,
                    "committee_refs": d.committee_refs,
                    "section_refs": d.section_refs
                } for d in definitions]
            }
            
            # Save to blob storage
            logger.info(f"Saving {len(definitions)} definitions to blob storage...")
            blob_manager.save_processed_data(definitions_data)
            
            # Index the definitions using specialized indexer
            logger.info("Indexing definitions...")
            indexer = DefinitionsIndexer(  # Changed to DefinitionsIndexer
                service_endpoint=search_endpoint,
                admin_key=search_admin_key,
                index_name=definitions_index_name,
                openai_api_key=openai_key
            )
            
            # Index the definitions directly using the specialized indexer
            indexer.index_definitions(definitions_data["definitions"])  # Changed to use index_definitions method
            
        logger.info(f"Successfully processed {len(definitions)} definitions")
        
    except Exception as e:
        logger.error(f"Error processing definitions: {str(e)}")
        raise

if __name__ == "__main__":
    asyncio.run(process_definitions())
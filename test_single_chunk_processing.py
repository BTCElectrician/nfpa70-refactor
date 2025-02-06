import os
import json
from pathlib import Path
from dotenv import load_dotenv
from loguru import logger

# Azure
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery
from azure.core.credentials import AzureKeyCredential

# Local application imports
from data_processing.pdf_extractor import PDFExtractor
from data_processing.text_chunker import ElectricalCodeChunker
from azure_search.index_creator import create_search_index
from azure_search.data_indexer import DataIndexer

def setup_logging():
    """Configure detailed logging for testing."""
    logger.remove()  # Remove default handler
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

def test_single_chunk_processing():
    """
    Tests the pipeline on a single chunk of text with enhanced vector validation.
    Updated for current Azure Search vector query syntax.
    """
    try:
        # Set up logging
        setup_logging()
        logger.info("Starting test processing with vector validation")
        
        # 1. Validate environment
        load_dotenv()
        validate_environment()
        
        # Load environment variables
        pdf_path = os.getenv('PDF_PATH')
        search_endpoint = os.getenv('AZURE_SEARCH_SERVICE_ENDPOINT')
        search_key = os.getenv('AZURE_SEARCH_ADMIN_KEY')
        openai_key = os.getenv('OPENAI_API_KEY')
        test_index_name = os.getenv('AZURE_SEARCH_INDEX_NAME', 'nfpa70-test')

        # 2. Extract from PDF (first three pages for testing)
        logger.info("Extracting text from PDF (3 pages max)...")
        extractor = PDFExtractor()
        pages_text = extractor.extract_text_from_pdf(Path(pdf_path), max_pages=3)

        if pages_text:
            first_page_num = list(pages_text.keys())[0]
            sample_text = pages_text[first_page_num]
            logger.info(f"First page text sample: {sample_text[:200]}...")
        else:
            logger.warning("No text extracted from PDF")
            return

        # 3. Chunk the text
        logger.info("Processing text into chunks...")
        chunker = ElectricalCodeChunker(openai_api_key=openai_key)
        chunks = chunker.chunk_nfpa70_content(pages_text)
        logger.info(f"Created {len(chunks)} chunks")

        # 4. Create/update the index
        logger.info(f"Setting up test index: {test_index_name}")
        create_search_index(search_endpoint, search_key, test_index_name)

        # 5. Initialize the DataIndexer
        indexer = DataIndexer(
            service_endpoint=search_endpoint,
            admin_key=search_key,
            index_name=test_index_name,
            openai_api_key=openai_key
        )

        # 6. Process and index first chunk only
        if chunks:
            first_chunk = chunks[0]
            logger.debug(f"First chunk structure: {json.dumps(first_chunk.__dict__, indent=2)}")
            
            # Convert to dictionary with top-level fields (no nested metadata)
            chunk_dict = {
                "content": first_chunk.content,
                "page_number": first_chunk.page_number,
                "article_number": first_chunk.article_number,
                "section_number": first_chunk.section_number,
                "article_title": first_chunk.article_title or "",
                "section_title": first_chunk.section_title or "",
                "context_tags": list(first_chunk.context_tags or []),
                "related_sections": list(first_chunk.related_sections or []),
                "gpt_analysis": first_chunk.gpt_analysis or {}
            }
            
            logger.info("Preparing to index first chunk...")
            logger.debug(f"Chunk dictionary: {json.dumps(chunk_dict, indent=2)}")
            
            # Index this single chunk
            indexer.index_documents([chunk_dict])
            logger.info("Successfully indexed first chunk")

            # 7. Verify indexed content using current vector search syntax
            search_client = SearchClient(
                endpoint=search_endpoint,
                index_name=test_index_name,
                credential=AzureKeyCredential(search_key)
            )
            
            # Get the embedding for verification search
            vector = indexer.generate_embeddings(chunk_dict["content"])
            
            # Perform vector search with updated syntax
            results = search_client.search(
                search_text="",  # Empty for pure vector search
                vector_queries=[
                    VectorizedQuery(
                        vector=vector,
                        fields="content_vector",
                        k_nearest_neighbors=5,  # Updated parameter name
                        exhaustive=True
                    )
                ],
                select=["id", "content", "page_number"]
            )
            
            hits = list(results)
            logger.info(f"Found {len(hits)} documents in test index")
            
            if hits:
                logger.info("Test completed successfully!")
            else:
                logger.warning("No documents found in index after upload")
        else:
            logger.warning("No chunks created from PDF")

    except Exception as e:
        logger.error(f"Test failed: {str(e)}")
        raise

if __name__ == "__main__":
    test_single_chunk_processing()
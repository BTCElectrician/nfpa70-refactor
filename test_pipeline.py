import json
import os
from pathlib import Path

# Third-party imports - make sure to install these first
from dotenv import load_dotenv  # Updated from dotenv
from loguru import logger
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchIndex,
    SearchField,
    SearchFieldDataType,
    VectorSearch,
    HnswAlgorithmConfiguration,
    HnswParameters,
    SemanticConfiguration,
    SemanticField,
    SemanticSettings,
    VectorizedQuery,
    PrioritizedFields
)

# Local application imports
from data_processing.pdf_extractor import PDFExtractor
from data_processing.text_chunker import ElectricalCodeChunker
from azure_search.index_creator import create_search_index
from azure_search.data_indexer import DataIndexer

def test_single_chunk_processing():
    """Test the pipeline with a single chunk of text"""
    try:
        # Load environment variables
        load_dotenv()

        # Get configuration
        pdf_path = os.getenv('PDF_PATH')
        search_endpoint = os.getenv('AZURE_SEARCH_SERVICE_ENDPOINT')
        search_key = os.getenv('AZURE_SEARCH_ADMIN_KEY')
        openai_key = os.getenv('OPENAI_API_KEY')
        test_index_name = "nfpa70-test-index"

        # Initialize the Search Index Client
        index_client = SearchIndexClient(
            endpoint=search_endpoint,
            credential=search_key
        )

        # 1. Test PDF Extraction (first page only)
        logger.info("Testing PDF extraction...")
        extractor = PDFExtractor()
        pages_text = extractor.extract_text_from_pdf(Path(pdf_path), max_pages=1)

        # Log extracted text
        first_page_text = list(pages_text.values())[0]
        logger.info(f"\nExtracted text preview:\n{first_page_text[:500]}...")

        # 2. Test Chunking
        logger.info("\nTesting chunking process...")
        chunker = ElectricalCodeChunker(openai_api_key=openai_key)
        chunks = chunker.chunk_nfpa70_content(pages_text)

        # Log first chunk details
        first_chunk = chunks[0]
        logger.info("\nFirst chunk details:")
        logger.info(f"Page Number: {first_chunk.page_number}")
        logger.info(f"Article Number: {first_chunk.article_number}")
        logger.info(f"Section Number: {first_chunk.section_number}")
        logger.info(f"Context Tags: {first_chunk.context_tags}")
        logger.info(f"Related Sections: {first_chunk.related_sections}")
        logger.info(f"GPT Analysis: {json.dumps(first_chunk.gpt_analysis, indent=2)}")

        # 3. Test Index Creation
        logger.info("\nTesting index creation...")
        # Updated vector search configuration
        vector_search = VectorSearch(
            algorithms=[
                HnswAlgorithmConfiguration(
                    name="my-vector-config",
                    kind="hnsw",
                    parameters=HnswParameters(
                        m=4,
                        ef_construction=400,
                        ef_search=500,
                        metric="cosine"
                    )
                )
            ]
        )
        
        # Configure semantic search
        semantic_config = SemanticConfiguration(
            name="my-semantic-config",
            prioritized_fields=PrioritizedFields(
                title_field=SearchField(name="title", type=SearchFieldDataType.String),
                keywords_fields=[],
                content_fields=[SearchField(name="content", type=SearchFieldDataType.String)]
            )
        )
        
        semantic_settings = SemanticSettings(
            configurations=[semantic_config]
        )

        # Create index with updated configurations
        create_search_index(
            index_client, 
            test_index_name, 
            vector_search,
            semantic_settings
        )

        # 4. Test Single Document Indexing
        logger.info("\nTesting document indexing...")
        search_client = SearchClient(
            endpoint=search_endpoint,
            index_name=test_index_name,
            credential=search_key
        )
        
        indexer = DataIndexer(
            search_client=search_client,
            openai_api_key=openai_key
        )

        # Index only the first chunk
        test_chunks = [chunks[0]]
        indexer.index_documents(test_chunks)

        logger.info("\nTest completed successfully!")

    except Exception as e:
        logger.error(f"Test failed: {str(e)}")
        raise

if __name__ == "__main__":
    test_single_chunk_processing()
import os
import json
from pathlib import Path
from dotenv import load_dotenv
from loguru import logger

# Azure
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential

# Local application imports
from data_processing.pdf_extractor import PDFExtractor
from data_processing.text_chunker import ElectricalCodeChunker
from azure_search.index_creator import create_search_index
from azure_search.data_indexer import DataIndexer

def test_single_chunk_processing():
    """
    Tests the pipeline on a single chunk of text (first page).
    1) Extract from PDF
    2) Chunk
    3) Create/Update the index schema using the production create_search_index
    4) Index one chunk
    5) (Optionally) do a simple search to confirm success
    """
    try:
        # 1. Load environment variables
        load_dotenv()
        pdf_path = os.getenv('PDF_PATH')
        search_endpoint = os.getenv('AZURE_SEARCH_SERVICE_ENDPOINT')
        search_key = os.getenv('AZURE_SEARCH_ADMIN_KEY')
        openai_key = os.getenv('OPENAI_API_KEY')
        
        # You can either override with a test index name or use your normal production index
        test_index_name = os.getenv('AZURE_SEARCH_INDEX_NAME', 'nfpa70-index-test')

        if not all([pdf_path, search_endpoint, search_key, openai_key]):
            raise ValueError("Missing one or more required environment variables.")

        # 2. Extract from PDF (first three pages for testing)
        logger.info("Extracting text from PDF (3 pages max)...")
        extractor = PDFExtractor()
        pages_text = extractor.extract_text_from_pdf(Path(pdf_path), max_pages=3)

        # Just log a small snippet
        if pages_text:
            first_page_num = list(pages_text.keys())[0]
            logger.info(f"First page text preview:\n{pages_text[first_page_num][:500]}...")
        else:
            logger.warning("No text extracted. Is the PDF empty?")

        # 3. Chunk the text
        logger.info("Chunking text into code chunks...")
        chunker = ElectricalCodeChunker(openai_api_key=openai_key)
        chunks = chunker.chunk_nfpa70_content(pages_text)
        logger.info(f"Number of chunks created: {len(chunks)}")

        # 4. Create or update the index with the production fields
        logger.info(f"Creating/Updating the test index: {test_index_name} ...")
        create_search_index(search_endpoint, search_key, test_index_name)
        logger.info("Index creation/update complete.")

        # 5. Initialize the DataIndexer (production style) for indexing
        indexer = DataIndexer(
            service_endpoint=search_endpoint,
            admin_key=search_key,
            index_name=test_index_name,
            openai_api_key=openai_key
        )

        # 6. Index only the first chunk
        if chunks:
            first_chunk = chunks[0]
            # Convert it to the dictionary format the indexer expects:
            chunk_dict = {
                "content": first_chunk.content,
                "metadata": {
                    "article": first_chunk.article_number,
                    "section": first_chunk.section_number,
                    "page": first_chunk.page_number
                },
                "context_tags": first_chunk.context_tags,
                "related_sections": first_chunk.related_sections,
                "article_title": first_chunk.article_title or "",
                "section_title": first_chunk.section_title or "",
                "gpt_analysis": first_chunk.gpt_analysis or {}
            }
            # Index it
            logger.info("Indexing the first chunk only...")
            indexer.index_documents([chunk_dict])
        else:
            logger.warning("No chunks to index.")

        # 7. (Optional) Quick search test to confirm the doc is in the index
        search_client = SearchClient(
            endpoint=search_endpoint,
            index_name=test_index_name,
            credential=AzureKeyCredential(search_key)
        )
        results = search_client.search(search_text="electric", top=5)
        hits = list(results)
        logger.info(f"Found {len(hits)} results when searching for 'electric' in test index.")

        logger.info("Test completed successfully!")

    except Exception as e:
        logger.error(f"Test failed: {str(e)}")
        raise

if __name__ == "__main__":
    test_single_chunk_processing()

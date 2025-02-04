<file_contents>
File: /Users/collin/nfpa70-refactor/azure_function/function_app.py
```py
import azure.functions as func
import json
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from openai import OpenAI
import os
from typing import List, Dict, Optional
from loguru import logger

app = func.FunctionApp()

def parse_field_query(query: str) -> Dict[str, Optional[str]]:
    """
    Parse field-specific context from the query.
    
    Args:
        query: Natural language query from user
        
    Returns:
        Dictionary of extracted context (location, equipment, requirement)
    """
    # Common electrical terms to look for
    contexts = {
        'location': ['in', 'at', 'inside', 'outside', 'within', 'through'],
        'equipment': [
            'transformer', 'conduit', 'busway', 'panel', 'switchgear',
            'motor', 'generator', 'disconnect', 'receptacle', 'outlet',
            'luminaire', 'fixture', 'raceway', 'cable', 'wire'
        ],
        'requirement': [
            'clearance', 'spacing', 'distance', 'rating', 'size', 'grounding',
            'support', 'mounting', 'installation', 'protection', 'classification'
        ]
    }
    
    parsed = {
        'location': None,
        'equipment': None,
        'requirement': None
    }
    
    words = query.lower().split()
    for i, word in enumerate(words):
        # Look for location context
        if word in contexts['location'] and i + 1 < len(words):
            parsed['location'] = words[i + 1]
            
        # Look for equipment
        if word in contexts['equipment']:
            parsed['equipment'] = word
            
        # Look for requirement type
        if word in contexts['requirement']:
            parsed['requirement'] = word
            
    return parsed

def format_response(results: List[Dict]) -> Dict:
    """
    Format search results for field use.
    
    Args:
        results: Raw search results
        
    Returns:
        Formatted results with code references and context
    """
    formatted_results = []
    
    for result in results:
        formatted_result = {
            'code_reference': f"Section {result.get('section_number', 'N/A')}",
            'page_number': result.get('page_number', 'N/A'),
            'requirement': result.get('content', '').strip(),
            'score': result.get('@search.score', 0),
            'related_sections': result.get('related_sections', []),
            'context_tags': result.get('context_tags', [])
        }
        formatted_results.append(formatted_result)
    
    return {
        'results': formatted_results,
        'total_results': len(formatted_results)
    }

@app.route(route="search")
def search_nfpa70(req: func.HttpRequest) -> func.HttpResponse:
    """
    Enhanced search endpoint for field use.
    
    Args:
        req: HTTP request containing search query
        
    Returns:
        HTTP response with search results
    """
    try:
        # Get search parameters
        query = req.params.get('q')
        if not query:
            return func.HttpResponse(
                "Please provide a search query",
                status_code=400
            )

        # Parse field context from query
        field_context = parse_field_query(query)
        
        # Generate embedding for semantic search
        try:
            client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
            response = client.embeddings.create(
                input=[query],
                model="text-embedding-3-small"
            )
            search_vector = response.data[0].embedding
        except Exception as e:
            logger.error(f"Error generating embeddings: {str(e)}")
            return func.HttpResponse(
                json.dumps({"error": "Error generating query embedding"}),
                status_code=500,
                mimetype="application/json"
            )

        # Set up search client
        try:
            credential = AzureKeyCredential(os.environ["AZURE_SEARCH_API_KEY"])
            search_client = SearchClient(
                os.environ["AZURE_SEARCH_SERVICE_ENDPOINT"],
                os.environ["AZURE_SEARCH_INDEX_NAME"],
                credential
            )
        except Exception as e:
            logger.error(f"Error setting up search client: {str(e)}")
            return func.HttpResponse(
                json.dumps({"error": "Error connecting to search service"}),
                status_code=500,
                mimetype="application/json"
            )

        # Build filter based on context
        filter_conditions = []
        for context_type, value in field_context.items():
            if value:
                filter_conditions.append(f"context_tags/any(t: t eq '{value}')")
        
        filter_expression = " and ".join(filter_conditions) if filter_conditions else None

        # Perform hybrid search
        try:
            results = search_client.search(
                search_text=query,  # For keyword matching
                vector=search_vector,  # For semantic matching
                vector_fields="content_vector",
                filter=filter_expression,
                select=[
                    "section_number",
                    "content",
                    "page_number",
                    "related_sections",
                    "context_tags"
                ],
                top=5,
                semantic_configuration_name="my-semantic-config"
            )

            # Format results for field use
            formatted_results = format_response(list(results))
            
            return func.HttpResponse(
                json.dumps(formatted_results),
                mimetype="application/json"
            )

        except Exception as e:
            logger.error(f"Error performing search: {str(e)}")
            return func.HttpResponse(
                json.dumps({"error": "Error performing search"}),
                status_code=500,
                mimetype="application/json"
            )

    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return func.HttpResponse(
            json.dumps({"error": "Internal server error"}),
            status_code=500,
            mimetype="application/json"
        ) 
```

File: /Users/collin/nfpa70-refactor/azure_search/data_indexer.py
```py
import numpy as np
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from openai import OpenAI
from typing import List, Dict, Any, Optional
from loguru import logger
from tqdm import tqdm
import json

class DataIndexer:
    """Handles indexing of processed electrical code content into Azure Search."""
    
    def __init__(self, service_endpoint: str, admin_key: str, index_name: str, openai_api_key: str):
        """Initialize the indexer with necessary credentials and configuration."""
        self.search_client = SearchClient(
            endpoint=service_endpoint,
            index_name=index_name,
            credential=AzureKeyCredential(admin_key)
        )
        self.openai_client = OpenAI(api_key=openai_api_key)
        self.logger = logger

    def generate_embeddings(self, text: str, model: str = "text-embedding-3-small") -> List[float]:
        """Generate embeddings for text using OpenAI's API."""
        try:
            self.logger.debug(f"[generate_embeddings] Generating embedding for text (length: {len(text)})")
            response = self.openai_client.embeddings.create(
                input=[text],
                model=model
            )
            # Convert embedding to list of floats
            embedding = [float(x) for x in response.data[0].embedding]
            
            # Verify embedding dimension
            if len(embedding) != 1536:
                raise ValueError(f"Unexpected embedding dimension: {len(embedding)}")
            
            self.logger.debug(f"[generate_embeddings] Successfully generated embedding with dimension {len(embedding)}")
            return embedding
            
        except Exception as e:
            self.logger.error(f"Error generating embeddings: {str(e)}")
            raise

    def prepare_document(self, chunk: Dict[str, Any], chunk_id: int) -> Dict[str, Any]:
        """Prepare a document for indexing with proper vector handling."""
        try:
            self.logger.debug(f"[prepare_document] Processing chunk {chunk_id}")
            
            # Generate embedding for the content
            content_vector = self.generate_embeddings(chunk["content"])
            self.logger.debug(f"[prepare_document] Generated vector with shape: {len(content_vector)}")
            
            # Extract metadata
            metadata = chunk.get("metadata", {})
            
            # Handle GPT analysis
            gpt_analysis = chunk.get("gpt_analysis", "")
            if isinstance(gpt_analysis, (dict, list)):
                gpt_analysis = json.dumps(gpt_analysis)
            
            # Create document with all fields
            document = {
                "id": f"doc_{chunk_id}",
                "content": chunk["content"],
                "page_number": metadata.get("page", 0),
                "article_number": str(metadata.get("article") or ""),
                "section_number": str(metadata.get("section") or ""),
                "article_title": chunk.get("article_title") or "",
                "section_title": chunk.get("section_title") or "",
                "content_vector": content_vector,  # Send as direct array of floats
                "context_tags": list(chunk.get("context_tags") or []),
                "related_sections": list(chunk.get("related_sections") or []),
                "gpt_analysis": gpt_analysis
            }
            
            # Validate document
            self._validate_document(document)
            self.logger.debug(f"[prepare_document] Document {chunk_id} prepared successfully")
            
            return document
            
        except Exception as e:
            self.logger.error(f"Error preparing document {chunk_id}: {str(e)}")
            raise

    def _validate_document(self, document: Dict[str, Any]) -> None:
        """Validate document structure before indexing."""
        required_fields = [
            "id", "content", "page_number", "content_vector"
        ]
        for field in required_fields:
            if field not in document:
                raise ValueError(f"Missing required field: {field}")
        
        # Validate vector field
        if not isinstance(document["content_vector"], list):
            raise ValueError("content_vector must be a list")
        if len(document["content_vector"]) != 1536:
            raise ValueError(f"content_vector must have 1536 dimensions, got {len(document['content_vector'])}")
        if not all(isinstance(x, float) for x in document["content_vector"]):
            raise ValueError("All vector values must be float type")

    def index_documents(self, chunks: List[Dict[str, Any]], batch_size: int = 50) -> None:
        """Index all documents with progress tracking and error handling."""
        try:
            total_chunks = len(chunks)
            self.logger.info(f"Starting indexing of {total_chunks} chunks")
            documents = []
            
            # Process chunks with progress bar
            for i in tqdm(range(total_chunks), desc="Processing chunks"):
                try:
                    self.logger.debug(f"Processing chunk {i}")
                    doc = self.prepare_document(chunks[i], i)
                    documents.append(doc)
                    
                    # Upload in batches
                    if len(documents) >= batch_size or i == total_chunks - 1:
                        self.logger.debug(f"Uploading batch of {len(documents)} documents")
                        try:
                            results = self.search_client.upload_documents(documents=documents)
                            self.logger.info(f"Successfully indexed batch of {len(results)} documents")
                            documents = []
                        except Exception as e:
                            self.logger.error(f"Error uploading batch: {str(e)}")
                            raise
                
                except Exception as e:
                    self.logger.error(f"Error processing chunk {i}: {str(e)}")
                    raise
                    
        except Exception as e:
            self.logger.error(f"Error in indexing process: {str(e)}")
            raise
```

File: /Users/collin/nfpa70-refactor/azure_search/index_creator.py
```py
import os
from azure.core.credentials import AzureKeyCredential
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchIndex,
    SimpleField,
    SearchableField,
    SearchField,
    SearchFieldDataType,
    VectorSearch,
    HnswParameters,
    HnswAlgorithmConfiguration,
    VectorSearchProfile,
    SemanticConfiguration,
    SemanticField,
    SemanticSearch,
    SemanticPrioritizedFields
)
from loguru import logger

def create_search_index(service_endpoint: str, admin_key: str, index_name: str) -> None:
    """
    Create or update an Azure Cognitive Search index with vector search enabled.
    Includes fields like article_title, section_title, and gpt_analysis.
    Updated for azure-search-documents==11.5.2 with correct vector profile configuration.
    """
    try:
        credential = AzureKeyCredential(admin_key)
        index_client = SearchIndexClient(endpoint=service_endpoint, credential=credential)

        # Delete existing index if it exists
        try:
            index_client.delete_index(index_name)
            logger.info(f"Deleted existing index '{index_name}'")
        except Exception:
            logger.info(f"Index '{index_name}' does not exist yet")

        # Vector search configuration
        vector_search = VectorSearch(
            algorithms=[
                HnswAlgorithmConfiguration(
                    name="hnsw-config",  # Algorithm name
                    parameters=HnswParameters(
                        m=4,
                        ef_construction=400,
                        ef_search=500,
                        metric="cosine"
                    )
                )
            ],
            profiles=[
                VectorSearchProfile(
                    name="hnsw-profile",  # Profile name referenced by fields
                    algorithm_configuration_name="hnsw-config"  # Must match algorithm name
                )
            ]
        )

        fields = [
            SimpleField(name="id", type=SearchFieldDataType.String, key=True),
            SearchableField(
                name="content",
                type=SearchFieldDataType.String,
                analyzer_name="standard.lucene"
            ),
            SimpleField(
                name="page_number",
                type=SearchFieldDataType.Int32,
                filterable=True
            ),
            SearchableField(
                name="article_number",
                type=SearchFieldDataType.String,
                filterable=True,
                facetable=True
            ),
            SearchableField(
                name="section_number",
                type=SearchFieldDataType.String,
                filterable=True,
                facetable=True
            ),
            SearchableField(
                name="article_title",
                type=SearchFieldDataType.String
            ),
            SearchableField(
                name="section_title",
                type=SearchFieldDataType.String
            ),
            SearchableField(
                name="related_sections",
                type=SearchFieldDataType.Collection(SearchFieldDataType.String),
                filterable=True,
                facetable=True
            ),
            SearchableField(
                name="context_tags",
                type=SearchFieldDataType.Collection(SearchFieldDataType.String),
                filterable=True,
                facetable=True
            ),
            SearchableField(
                name="gpt_analysis",
                type=SearchFieldDataType.String
            ),
            # Vector field with correct v11.5.2 properties
            SearchField(
                name="content_vector",
                type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                searchable=True,
                vector_search_dimensions=1536,  # Correct property name for v11.5.2
                vector_search_profile_name="hnsw-profile"  # Must match profile name
            )
        ]

        # Semantic configuration
        semantic_config = SemanticConfiguration(
            name="my-semantic-config",
            prioritized_fields=SemanticPrioritizedFields(
                content_fields=[SemanticField(field_name="content")],
                title_field=SemanticField(field_name="section_title")
            )
        )
        semantic_search = SemanticSearch(configurations=[semantic_config])

        # Create index
        index = SearchIndex(
            name=index_name,
            fields=fields,
            vector_search=vector_search,
            semantic_search=semantic_search
        )

        logger.info(f"Creating search index '{index_name}' ...")
        index_client.create_or_update_index(index)
        logger.info(f"Index '{index_name}' created successfully.")

    except Exception as e:
        logger.error(f"Error creating/updating index: {str(e)}")
        raise
```

File: /Users/collin/nfpa70-refactor/data_processing/pdf_extractor.py
```py
import pymupdf 
import re
from typing import Dict, Optional
import logging
from pathlib import Path

class PDFExtractor:
    """Enhanced PDF text extraction for electrical code documents."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Common OCR/PDF extraction artifacts to clean
        self.corrections = {
            r'o;': 's',
            r'\"': '"',
            r'\\"': '"',
            r'\\\'': "'",
            r'\s+': ' ',
            r'(?<=\d)\s*\.\s*(?=\d)': '.',  # Fix broken decimal points
            r'(?<=\d)\s*-\s*(?=\d)': '-',    # Fix broken ranges
        }
        
        # Important electrical terms to preserve
        self.electrical_terms = {
            r'(\d+)\s*[Vv]\b': r'\1 volts',
            r'(\d+)\s*[Aa]\b': r'\1 amperes',
            r'(\d+)\s*[Ww]\b': r'\1 watts',
            r'(\d+)\s*VAC': r'\1 VAC',
            r'(\d+)\s*VDC': r'\1 VDC',
            r'(\d+)\s*AWG': r'\1 AWG',
            r'(\d+)\s*hp\b': r'\1 horsepower',
            r'(\d+)\s*kVA\b': r'\1 kVA',
            r'(\d+)\s*Hz\b': r'\1 Hz'
        }
        
        # Common electrical terms that might be broken
        self.term_fixes = {
            'ground ing': 'grounding',
            'bond ing': 'bonding',
            'circuit breaker': 'circuit breaker',
            'race way': 'raceway',
            'load center': 'load center',
            'sub panel': 'subpanel',
            'sub circuit': 'subcircuit'
        }

    def clean_text(self, text: str) -> str:
        """
        Clean extracted text while preserving important electrical terms.
        
        Args:
            text: Raw text from PDF
            
        Returns:
            Cleaned text with preserved electrical terminology
        """
        # Basic cleaning
        text = text.strip()
        
        # Apply OCR corrections
        for pattern, replacement in self.corrections.items():
            text = re.sub(pattern, replacement, text)
        
        # Fix broken electrical terms
        for broken, fixed in self.term_fixes.items():
            text = text.replace(broken, fixed)
        
        # Preserve electrical measurements and units
        for pattern, replacement in self.electrical_terms.items():
            text = re.sub(pattern, replacement, text)
        
        # Remove multiple spaces and normalize newlines
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        return text.strip()

    def extract_text_from_pdf(self, pdf_path: Path, max_pages: int = None) -> Dict[int, str]:
        """
        Extract text from PDF with optional page limit for testing.
        
        Args:
            pdf_path: Path to PDF file
            max_pages: Optional maximum number of pages to process
        
        Returns:
            Dict mapping page numbers to cleaned text
        """
        try:
            self.logger.info(f"Processing PDF: {pdf_path}")
            doc = pymupdf.open(pdf_path)
            pages_text = {}
            
            total_pages = len(doc)
            pages_to_process = min(max_pages, total_pages) if max_pages else total_pages
            
            for page_num in range(pages_to_process):
                page = doc[page_num]
                
                # Extract text with better layout preservation
                text = page.get_text("text")
                
                # Clean the extracted text
                cleaned_text = self.clean_text(text)
                
                # Only store non-empty pages
                if cleaned_text.strip():
                    # Store with page number (1-based)
                    pages_text[page_num + 1] = cleaned_text
                
            self.logger.info(f"Successfully processed {len(pages_text)} pages")
            return pages_text
            
        except Exception as e:
            self.logger.error(f"Error processing PDF: {str(e)}")
            raise ValueError(f"Failed to process PDF: {str(e)}") from e 
```

File: /Users/collin/nfpa70-refactor/azure_storage/blob_manager.py
```py
import os
import json
import logging

from azure.core.exceptions import ResourceExistsError, ResourceNotFoundError
from azure.storage.blob import BlobServiceClient, ContentSettings

class BlobStorageManager:
    """
    Basic blob manager for uploading/downloading JSON data.
    Relies on the AzureWebJobsStorage connection string in environment variables.
    """
    def __init__(self, container_name: str = "processed-data", blob_name: str = "processed_data.json"):
        self.connect_str = os.getenv('AzureWebJobsStorage')
        if not self.connect_str:
            raise ValueError("Missing AzureWebJobsStorage connection string in environment.")

        self.blob_service_client = BlobServiceClient.from_connection_string(self.connect_str)
        self.container_name = container_name
        self.blob_name = blob_name

        # Ensure container exists (create if needed)
        try:
            self.blob_service_client.create_container(self.container_name)
        except ResourceExistsError:
            pass  # container already exists

    def save_processed_data(self, data: dict) -> None:
        """Save a Python dictionary to Blob Storage as JSON."""
        try:
            container_client = self.blob_service_client.get_container_client(self.container_name)
            blob_client = container_client.get_blob_client(self.blob_name)

            blob_client.upload_blob(
                json.dumps(data),
                overwrite=True,
                content_settings=ContentSettings(
                    content_type='application/json',
                    cache_control='no-cache'
                )
            )
            logging.info(f"Data saved to blob: {self.blob_name}")

        except Exception as e:
            logging.error(f"Error saving data to blob: {str(e)}")
            raise

    def load_processed_data(self) -> dict:
        """Load data from Blob Storage and return as a Python dictionary."""
        try:
            container_client = self.blob_service_client.get_container_client(self.container_name)
            blob_client = container_client.get_blob_client(self.blob_name)

            downloaded = blob_client.download_blob().readall()
            return json.loads(downloaded)

        except ResourceNotFoundError:
            logging.warning(f"Blob {self.blob_name} not found in container {self.container_name}")
            return {}
        except Exception as e:
            logging.error(f"Error loading data from blob: {str(e)}")
            return {}

```

File: /Users/collin/nfpa70-refactor/.cursorrules
```cursorrules
Commit Message Prefixes:
* "fix:" for bug fixes
* "feat:" for new features
* "perf:" for performance improvements
* "docs:" for documentation changes
* "style:" for formatting changes
* "refactor:" for code refactoring
* "test:" for adding missing tests
* "chore:" for maintenance tasks
Rules:
* Use lowercase for commit messages
* Keep the summary line concise
* Include description for non-obvious changes
* Reference issue numbers when applicable
Documentation

* Maintain clear README with setup instructions
* Document API interactions and data flows
* Keep manifest.json well-documented
* Don't include comments unless it's for complex logic
* Document permission requirements
Development Workflow

* Use proper version control
* Implement proper code review process
* Test in multiple environments
* Follow semantic versioning for releases
* Maintain changelog

```

File: /Users/collin/nfpa70-refactor/main-backup.py
```py
from pathlib import Path
from typing import Final
from collections.abc import Sequence
from os import getenv
from dotenv import load_dotenv
import logging

from data_processing.pdf_extractor import PDFExtractor
from data_processing.text_chunker import ElectricalCodeChunker
from azure_search.index_creator import create_search_index
from azure_search.data_indexer import DataIndexer

# Set up logging with modern configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger: Final = logging.getLogger(__name__)

# Define required environment variables as constants
REQUIRED_VARS: Final[Sequence[str]] = (
    'PDF_PATH',
    'AZURE_SEARCH_SERVICE_ENDPOINT',
    'AZURE_SEARCH_ADMIN_KEY',
    'OPENAI_API_KEY'
)

def main() -> None:
    """Main process to extract, process, and index electrical code content."""
    try:
        # Load environment variables
        load_dotenv()
        
        # Get environment variables with type hints
        pdf_path: str | None = getenv('PDF_PATH')
        search_service_endpoint: str | None = getenv('AZURE_SEARCH_SERVICE_ENDPOINT')
        search_admin_key: str | None = getenv('AZURE_SEARCH_ADMIN_KEY')
        index_name: str = getenv('AZURE_SEARCH_INDEX_NAME', 'nfpa70-refactor')
        openai_api_key: str | None = getenv('OPENAI_API_KEY')

        # Validate environment variables
        missing_vars = [var for var in REQUIRED_VARS if not getenv(var)]
        if missing_vars:
            raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")

        # Convert string path to Path object and validate
        pdf_file = Path(pdf_path)
        if not pdf_file.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_file}")

        # Extract text from PDF with advanced cleaning
        logger.info("Extracting text from PDF...")
        pdf_extractor = PDFExtractor()
        pages_text = pdf_extractor.extract_text_from_pdf(pdf_file)
        logger.info(f"Successfully extracted text from {len(pages_text)} pages")

        # Process text into context-aware chunks
        logger.info("Processing text into chunks with electrical context...")
        chunker = ElectricalCodeChunker()
        chunks = chunker.chunk_nfpa70_content(pages_text)
        logger.info(f"Created {len(chunks)} context-aware chunks")

        # Create or update search index with enhanced schema
        logger.info("Creating search index...")
        create_search_index(search_service_endpoint, search_admin_key, index_name)

        # Index the documents with embeddings and context tags
        logger.info("Indexing documents...")
        indexer = DataIndexer(
            service_endpoint=search_service_endpoint,
            admin_key=search_admin_key,
            index_name=index_name,
            openai_api_key=openai_api_key
        )
        indexer.index_documents(chunks)

        logger.info("Process completed successfully!")

    except Exception as e:
        logger.error(f"Error in main process: {str(e)}")
        raise

if __name__ == "__main__":
    main() 
```

File: /Users/collin/nfpa70-refactor/index_from_blob.py
```py
import os
import logging
from dotenv import load_dotenv

from azure_storage.blob_manager import BlobStorageManager
from azure_search.data_indexer import DataIndexer

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
    
    # Step 1: Load chunked data from Blob Storage
    blob_manager = BlobStorageManager(container_name="processed-data", blob_name="nfpa70_chunks.json")
    blob_data = blob_manager.load_processed_data()
    chunks = blob_data.get("chunks", [])

    if not chunks:
        logger.error("No chunk data found in blob storage. Exiting.")
        return
    
    # Step 2: Index documents using DataIndexer
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

```

File: /Users/collin/nfpa70-refactor/data_processing/text_chunker.py
```py
import re
from typing import List, Dict, Optional
from dataclasses import dataclass, field
import logging
import json
from openai import OpenAI

@dataclass
class CodeChunk:
    """Represents a chunk of electrical code with context."""
    content: str
    page_number: int
    article_number: Optional[str] = None
    article_title: Optional[str] = None
    section_number: Optional[str] = None
    section_title: Optional[str] = None
    context_tags: List[str] = field(default_factory=list)
    related_sections: List[str] = field(default_factory=list)
    gpt_analysis: Dict = field(default_factory=dict)

class ElectricalCodeChunker:
    """Enhanced chunking for electrical code text with comprehensive NEC terminology."""
    
    def __init__(self, openai_api_key: Optional[str] = None):
        self.logger = logging.getLogger(__name__)
        self.client = OpenAI(api_key=openai_api_key) if openai_api_key else None
        
        # Comprehensive NEC terminology mapping
        self.context_mapping = {
            # Power Distribution
            'service_equipment': [
                'service equipment', 'service entrance', 'service drop', 'service lateral',
                'meter', 'metering', 'service disconnect', 'main disconnect', 'service panel',
                'switchboard', 'switchgear', 'panelboard'
            ],
            
            # Conductors and Raceways
            'conductors': [
                'conductor', 'wire', 'cable', 'AWG', 'kcmil', 'THHN', 'THWN', 'XHHW',
                'multiconductor', 'stranded', 'solid', 'copper', 'aluminum', 'CU', 'AL',
                'current-carrying', 'neutral', 'ungrounded', 'phase'
            ],
            
            'raceway': [
                'EMT', 'IMC', 'RMC', 'PVC', 'RTRC', 'LFMC', 'LFNC', 'FMC',
                'electrical metallic tubing', 'intermediate metal conduit',
                'rigid metal conduit', 'liquidtight flexible metal conduit',
                'schedule 40', 'schedule 80', 'nipple', 'chase', 'sleeve'
            ],
            
            # Grounding and Bonding
            'grounding': [
                'ground', 'grounded', 'grounding', 'earthing', 'bonding', 'GEC',
                'equipment grounding conductor', 'ground fault', 'GFCI', 'GFPE',
                'grounding electrode', 'ground rod', 'ufer', 'made electrode',
                'ground ring', 'ground plate', 'ground bus', 'isolated ground'
            ],
            
            # Overcurrent Protection
            'overcurrent': [
                'breaker', 'circuit breaker', 'fuse', 'overcurrent', 'overload',
                'short circuit', 'AFCI', 'arc fault', 'GFCI', 'ground fault',
                'fusible', 'nonfusible', 'instantaneous trip', 'adjustable trip'
            ],
            
            # Special Occupancies
            'special_locations': [
                'hazardous', 'classified', 'class I', 'class II', 'class III',
                'division 1', 'division 2', 'zone 0', 'zone 1', 'zone 2',
                'wet location', 'damp location', 'corrosive', 'hospital',
                'healthcare', 'assembly', 'theater', 'motor fuel', 'spray booth'
            ],
            
            # Motors and Controls
            'motors': [
                'motor', 'controller', 'starter', 'VFD', 'variable frequency',
                'horsepower', 'hp', 'full-load', 'locked rotor', 'duty cycle',
                'continuous duty', 'overload', 'disconnecting means'
            ],
            
            # Equipment
            'transformers': [
                'transformer', 'xfmr', 'xfmer', 'primary', 'secondary',
                'delta', 'wye', 'impedance', 'kVA', 'step-up', 'step-down',
                'dry-type', 'liquid-filled', 'vault'
            ],
            
            'hvac': [
                'air conditioning', 'HVAC', 'heat pump', 'condenser',
                'evaporator', 'cooling', 'heating', 'disconnecting means',
                'minimum circuit ampacity', 'maximum overcurrent'
            ],
            
            # Installation Requirements
            'installation': [
                'support', 'securing', 'fastening', 'mounting', 'attachment',
                'spacing', 'interval', 'strap', 'hanger', 'bracket', 'anchor',
                'embedded', 'concealed', 'exposed'
            ],
            
            'clearance': [
                'clearance', 'spacing', 'distance', 'separation', 'depth',
                'working space', 'dedicated space', 'headroom', 'minimum depth',
                'burial depth', 'cover'
            ],
            
            # Emergency Systems
            'emergency': [
                'emergency', 'legally required standby', 'optional standby',
                'backup', 'generator', 'transfer switch', 'automatic transfer',
                'manual transfer', 'essential electrical system'
            ],
            
            # Branch Circuits and Feeders
            'branch_circuits': [
                'branch circuit', 'feeder', 'multiwire', 'general purpose',
                'dedicated circuit', 'small appliance', 'laundry', 'SABC',
                'receptacle', 'outlet', 'lighting', '15-amp', '20-amp', '30-amp'
            ],
            
            # Special Equipment
            'welding': [
                'welder', 'welding outlet', 'welding receptacle', 'electrode',
                'duty cycle', 'demand factor'
            ],
            
            'ev_charging': [
                'electric vehicle', 'EV', 'charging equipment', 'EVSE',
                'fast charging', 'level 2', 'level 3', 'charging station'
            ],
            
            # Fire Alarm Systems
            'fire_alarm': [
                'fire alarm', 'smoke detector', 'heat detector', 'initiating device',
                'notification appliance', 'fire alarm control unit', 'FACU', 'FACP'
            ],
            
            # Solar PV Systems
            'solar': [
                'photovoltaic', 'PV', 'solar', 'inverter', 'rapid shutdown',
                'module', 'array', 'combiner', 'micro-inverter', 'optimizer'
            ],
            
            # Communications
            'communications': [
                'network', 'data', 'telephone', 'coaxial', 'fiber optic',
                'category', 'CAT', 'plenum', 'riser', 'cable tray', 'J-hooks'
            ]
        }
        
        # Regex patterns
        self.article_pattern = re.compile(r'ARTICLE\s+(\d+)\s*[-—]\s*(.+?)(?=\n|$)')
        self.section_pattern = re.compile(r'(\d+\.\d+(?:\([A-Z]\))?)\s+(.+?)(?=\n|$)')
        self.reference_pattern = re.compile(r'(?:see\s+(?:Section\s+)?|with\s+)(\d+\.\d+(?:\([A-Z]\))?)')

    def _identify_context(self, text: str) -> List[str]:
        """Identify technical context tags for the text."""
        text_lower = text.lower()
        tags = []
        for context, keywords in self.context_mapping.items():
            if any(keyword in text_lower for keyword in keywords):
                tags.append(context)
        return tags

    def _find_related_sections(self, text: str) -> List[str]:
        """Find referenced sections in the text."""
        return list(set(self.reference_pattern.findall(text)))

    def analyze_chunk_with_gpt(self, chunk: str) -> Dict:
        """Use GPT to analyze or clean up chunk content."""
        if not self.client:
            return {}
        
        prompt = f"""Analyze this NFPA 70 electrical code section and extract:
        1. Key technical requirements
        2. Equipment specifications
        3. Cross-references to other sections
        4. Safety-critical elements

        Code section:
        {chunk}

        Provide response in JSON format.
        """
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",  # Example model
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                response_format={ "type": "json_object" }
            )
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            self.logger.error(f"Error in GPT analysis: {str(e)}")
            return {}

    def chunk_nfpa70_content(self, pages_text: Dict[int, str]) -> List[CodeChunk]:
        """
        Process NFPA 70 content into context-aware chunks.
        Args:
            pages_text: Dict[page_number, text]
        Returns:
            List of CodeChunk objects
        """
        chunks = []
        current_article = None
        current_article_title = None
        
        for page_num, text in pages_text.items():
            lines = text.split('\n')
            current_chunk = []
            current_section = None
            current_section_title = None
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                # Check for article header
                article_match = self.article_pattern.search(line)
                if article_match:
                    current_article = article_match.group(1)
                    current_article_title = article_match.group(2).strip()
                    continue
                
                # Check for section header
                section_match = self.section_pattern.match(line)
                if section_match:
                    # Save previous chunk if it exists
                    if current_chunk:
                        chunk_text = ' '.join(current_chunk)
                        gpt_analysis = self.analyze_chunk_with_gpt(chunk_text)
                        chunks.append(CodeChunk(
                            content=chunk_text,
                            page_number=page_num,
                            article_number=current_article,
                            article_title=current_article_title,
                            section_number=current_section,
                            section_title=current_section_title,
                            context_tags=self._identify_context(chunk_text),
                            related_sections=self._find_related_sections(chunk_text),
                            gpt_analysis=gpt_analysis
                        ))
                    
                    # Start a new chunk
                    current_section = section_match.group(1)
                    current_section_title = section_match.group(2).strip()
                    current_chunk = [line]
                else:
                    current_chunk.append(line)
            
            # Handle last chunk on page
            if current_chunk:
                chunk_text = ' '.join(current_chunk)
                gpt_analysis = self.analyze_chunk_with_gpt(chunk_text)
                chunks.append(CodeChunk(
                    content=chunk_text,
                    page_number=page_num,
                    article_number=current_article,
                    article_title=current_article_title,
                    section_number=current_section,
                    section_title=current_section_title,
                    context_tags=self._identify_context(chunk_text),
                    related_sections=self._find_related_sections(chunk_text),
                    gpt_analysis=gpt_analysis
                ))
        
        return chunks

# Compatibility function for older code calling
def chunk_nfpa70_content(text: str, openai_api_key: Optional[str] = None) -> List[Dict]:
    """Wrapper for compatibility with existing code."""
    chunker = ElectricalCodeChunker(openai_api_key=openai_api_key)
    pages_text = {1: text}
    chunks = chunker.chunk_nfpa70_content(pages_text)
    
    return [
        {
            "content": chunk.content,
            "metadata": {
                "section": chunk.section_number,
                "article": chunk.article_number,
                "page": chunk.page_number
            },
            "context_tags": chunk.context_tags,
            "related_sections": chunk.related_sections,
            "gpt_analysis": chunk.gpt_analysis
        }
        for chunk in chunks
    ]

```

File: /Users/collin/nfpa70-refactor/research.md
```md

```

File: /Users/collin/nfpa70-refactor/requirements.txt
```txt
annotated-types==0.7.0
anyio==4.8.0
azure-common==1.1.28
azure-core==1.52.0
azure-functions==1.21.3
azure-search-documents==11.5.2
azure-storage-blob==12.28.0
azure-storage-common==12.27.0
blis==1.2.0
catalogue==2.0.10
certifi==2024.12.14
cffi==1.17.1
charset-normalizer==3.4.1
click==8.1.8
cloudpathlib==0.20.0
confection==0.1.5
cryptography==44.0.0
cymem==2.0.11
distro==1.9.0
filelock==3.16.1
fsspec==2024.12.0
h11==0.14.0
httpcore==1.0.7
httpx==0.28.1
huggingface-hub==0.27.1
idna==3.10
isodate==0.7.2
Jinja2==3.1.5
jiter==0.8.2
langcodes==3.5.0
language_data==1.3.0
loguru==0.7.3
marisa-trie==1.2.1
markdown-it-py==3.0.0
MarkupSafe==3.0.2
mdurl==0.1.2
murmurhash==1.0.12
numpy==2.2.1
openai==1.59.8
packaging==24.2
preshed==3.0.9
pycparser==2.22
pydantic==2.10.5
pydantic_core==2.27.2
Pygments==2.19.1
PyMuPDF==1.25.2
python-dotenv==1.0.1
python-slugify==8.0.4
PyYAML==6.0.2
regex==2024.11.6
requests==2.32.3
rich==13.9.4
safetensors==0.5.2
shellingham==1.5.4
six==1.17.0
smart-open==7.1.0
sniffio==1.3.1
spacy==3.8.4
spacy-legacy==3.0.12
spacy-loggers==1.0.5
srsly==2.5.1
text-unidecode==1.3
thinc==8.3.4
tokenizers==0.21.0
tqdm==4.67.1
transformers==4.48.0
typer==0.15.1
typing_extensions==4.12.2
urllib3==2.3.0
wasabi==1.1.3
weasel==0.4.1
wrapt==1.17.2

```

File: /Users/collin/nfpa70-refactor/main.py
```py
import os
import logging
from dotenv import load_dotenv

from data_processing.pdf_extractor import PDFExtractor
from data_processing.text_chunker import ElectricalCodeChunker
from azure_storage.blob_manager import BlobStorageManager
from azure_search.index_creator import create_search_index

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Extracts PDF text, chunks it (including GPT analysis if openai_api_key is set),
    and saves the chunks to blob storage. Optionally creates/updates the search index schema.
    """
    try:
        load_dotenv()

        # Required environment variables
        pdf_path = os.getenv('PDF_PATH')
        search_endpoint = os.getenv('AZURE_SEARCH_SERVICE_ENDPOINT')
        search_admin_key = os.getenv('AZURE_SEARCH_ADMIN_KEY')
        index_name = os.getenv('AZURE_SEARCH_INDEX_NAME', 'nfpa70-index')
        openai_api_key = os.getenv('OPENAI_API_KEY')

        # Validate required variables
        required = [pdf_path, search_endpoint, search_admin_key, openai_api_key]
        if any(v is None or v.strip() == "" for v in required):
            raise ValueError("One or more required environment variables are missing.")

        # Step 1: Extract PDF text
        extractor = PDFExtractor()
        pages_text = extractor.extract_text_from_pdf(pdf_path)
        logger.info(f"Extracted text from {len(pages_text)} pages.")

        # Step 2: Chunk text (includes optional GPT analysis if openai_api_key is valid)
        chunker = ElectricalCodeChunker(openai_api_key=openai_api_key)
        chunks = chunker.chunk_nfpa70_content(pages_text)
        logger.info(f"Created {len(chunks)} text chunks.")

        # Step 3: Convert chunk objects into dictionaries (including new fields)
        logger.info("Creating chunk dictionaries with all relevant fields...")
        chunk_dicts = []
        for c in chunks:
            chunk_dicts.append({
                "content": c.content,
                "metadata": {
                    "article": c.article_number,
                    "section": c.section_number,
                    "page": c.page_number
                },
                "context_tags": c.context_tags,
                "related_sections": c.related_sections,
                "article_title": c.article_title or "",
                "section_title": c.section_title or "",
                # If not doing GPT analysis, this can be an empty dict
                "gpt_analysis": c.gpt_analysis or {}
            })

        # Step 4: Save processed chunks to blob storage
        try:
            blob_manager = BlobStorageManager(container_name="processed-data", blob_name="nfpa70_chunks.json")
            blob_manager.save_processed_data({"chunks": chunk_dicts})
            logger.info("Saved chunked data to blob storage for later indexing.")
        except Exception as e:
            logger.warning(f"Failed to save chunks to blob storage: {e}")

        # Optional: Create or update the search index (schema only, no docs indexed here).
        create_search_index(search_endpoint, search_admin_key, index_name)
        logger.info(f"Search index '{index_name}' created/updated successfully.")

        logger.info("Main process completed without indexing. "
                    "To index the data, run index_from_blob.py separately.")

    except Exception as e:
        logger.error(f"Error in main process: {str(e)}")
        raise

if __name__ == "__main__":
    main()

```

File: /Users/collin/nfpa70-refactor/snapshot.md
```md

```

File: /Users/collin/nfpa70-refactor/upgrade.md
```md

```

File: /Users/collin/nfpa70-refactor/test_single_chunk_processing.py
```py
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
    """
    try:
        # Set up logging
        setup_logging()
        logger.info("Starting test processing with vector validation")
        
        # 1. Validate environment
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
            logger.info(f"First page text sample: {pages_text[first_page_num][:200]}...")
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
            
            # Convert to expected dictionary format
            chunk_dict = {
                "content": first_chunk.content,
                "metadata": {
                    "article": first_chunk.article_number,
                    "section": first_chunk.section_number,
                    "page": first_chunk.page_number
                },
                "context_tags": list(first_chunk.context_tags or []),
                "related_sections": list(first_chunk.related_sections or []),
                "article_title": first_chunk.article_title or "",
                "section_title": first_chunk.section_title or "",
                "gpt_analysis": first_chunk.gpt_analysis or {}
            }
            
            # Log the chunk structure
            logger.info("Preparing to index first chunk...")
            logger.debug(f"Chunk dictionary: {json.dumps(chunk_dict, indent=2)}")
            
            # Index the chunk
            indexer.index_documents([chunk_dict])
            logger.info("Successfully indexed first chunk")

            # 7. Verify indexed content
            search_client = SearchClient(
                endpoint=search_endpoint,
                index_name=test_index_name,
                credential=AzureKeyCredential(search_key)
            )
            
            # Test basic search
            results = search_client.search(
                search_text="*",
                select=["id", "content", "page_number"],
                top=5
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
```
</file_contents>

<xml_formatting_instructions>
### Role
- You are a **code editing assistant**: You can fulfill edit requests and chat with the user about code or other questions. Provide complete instructions or code lines when replying with xml formatting.

### Capabilities
- Can create new files.
- Can rewrite entire files.
- Can delete existing files.

Avoid placeholders like `...` or `// existing code here`. Provide complete lines or code.

## Tools & Actions
1. **create** – Create a new file if it doesn’t exist.
2. **rewrite** – Replace the entire content of an existing file.
3. **delete** – Remove a file entirely (empty <content>).

### **Format to Follow for Repo Prompt's Diff Protocol**

<Plan>
Describe your approach or reasoning here.
</Plan>

<file path="path/to/example.swift" action="one_of_the_tools">
  <change>
    <description>Brief explanation of this specific change</description>
    <content>
===
// Provide the new or updated code here. Do not use placeholders
===
    </content>
  </change>
</file>

#### Tools Demonstration
1. `<file path="NewFile.swift" action="create">` – Full file in <content>
2. `<file path="DeleteMe.swift" action="delete">` – Empty <content>
3. `<file path="RewriteMe.swift" action="rewrite">` – Entire file in <content>

## Format Guidelines
1. **Plan**: Begin with a `<Plan>` block explaining your approach.
2. **<file> Tag**: e.g. `<file path="Models/User.swift" action="...">`. Must match an available tool.
3. **<change> Tag**: Provide `<description>` to clarify each change. Then `<content>` for new/modified code. Additional rules depend on your capabilities.
4. **rewrite**: Replace the entire file. This is the only way to modify existing files.
5. **create**: For new files, put the full file in <content>.
6. **delete**: Provide an empty <content>. The file is removed.

## Code Examples

-----
### Example: Full File Rewrite
<Plan>
Rewrite the entire User file to include an email property.
</Plan>

<file path="Models/User.swift" action="rewrite">
  <change>
    <description>Full file rewrite with new email field</description>
    <content>
===
import Foundation
struct User {
    let id: UUID
    var name: String
    var email: String

    init(name: String, email: String) {
        self.id = UUID()
        self.name = name
        self.email = email
    }
}
===
    </content>
  </change>
</file>

-----
### Example: Create New File
<Plan>
Create a new RoundedButton for a custom Swift UIButton subclass.
</Plan>

<file path="Views/RoundedButton.swift" action="create">
  <change>
    <description>Create custom RoundedButton class</description>
    <content>
===
import UIKit
@IBDesignable
class RoundedButton: UIButton {
    @IBInspectable var cornerRadius: CGFloat = 0
}
===
    </content>
  </change>
</file>

-----
### Example: Delete a File
<Plan>
Remove an obsolete file.
</Plan>

<file path="Obsolete/File.swift" action="delete">
  <change>
    <description>Completely remove the file from the project</description>
    <content>
===
===
    </content>
  </change>
</file>

## Final Notes
1.  **rewrite**  For rewriting an entire file, place all new content in `<content>`. No partial modifications are possible here. Avoid all use of placeholders.
2. You can always **create** new files and **delete** existing files. Provide full code for create, and empty content for delete. Avoid creating files you know exist already.
3. If a file tree is provided, place your files logically within that structure. Respect the user’s relative or absolute paths.
4. Wrap your final output in ```XML ... ``` for clarity.
5. The final output must apply cleanly with no leftover syntax errors.
</xml_formatting_instructions>
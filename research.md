<file_contents>
File: /Users/collin/nfpa70-refactor/azure_search/index_creator.py
```py
from azure.core.credentials import AzureKeyCredential
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchIndex,
    SimpleField,
    SearchableField,
    SearchField,
    SearchFieldDataType,
    VectorSearch,
    HnswAlgorithmConfiguration,
    HnswParameters,
    SemanticConfiguration,
    SemanticField,
    SemanticSettings,
    PrioritizedFields
)
from loguru import logger

def create_search_index(service_endpoint: str, admin_key: str, index_name: str) -> None:
    """
    Create an enhanced search index for electrical code content.
    
    Args:
        service_endpoint: Azure Search service endpoint
        admin_key: Azure Search admin key
        index_name: Name for the search index
    """
    try:
        credential = AzureKeyCredential(admin_key)
        index_client = SearchIndexClient(endpoint=service_endpoint, credential=credential)

        # Define fields for the index
        fields = [
            SimpleField(name="id", type=SearchFieldDataType.String, key=True),
            SearchableField(
                name="content", 
                type=SearchFieldDataType.String,
                analyzer_name="en.microsoft"
            ),
            SimpleField(
                name="page_number", 
                type=SearchFieldDataType.Int32,
                filterable=True,
                sortable=True
            ),
            SearchableField(
                name="article_number",
                type=SearchFieldDataType.String,
                filterable=True,
                facetable=True
            ),
            SearchableField(
                name="article_title",
                type=SearchFieldDataType.String,
                filterable=True
            ),
            SearchableField(
                name="section_number",
                type=SearchFieldDataType.String,
                filterable=True,
                facetable=True
            ),
            SearchableField(
                name="section_title",
                type=SearchFieldDataType.String,
                filterable=True
            ),
            SearchableField(
                name="context_tags",
                type=SearchFieldDataType.Collection(SearchFieldDataType.String),
                filterable=True,
                facetable=True,
                searchable=True
            ),
            SearchableField(
                name="related_sections",
                type=SearchFieldDataType.Collection(SearchFieldDataType.String),
                filterable=True
            ),
            # Vector field for semantic search
            SearchField(
                name="content_vector",
                type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                vector_search_dimensions=1536,
                vector_search_profile_name="my-vector-config",
            ),
        ]

        # Configure vector search
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

        # Configure semantic search for field-oriented queries
        semantic_config = SemanticConfiguration(
            name="my-semantic-config",
            prioritized_fields=PrioritizedFields(
                title_field=SemanticField(field_name="section_title"),
                keywords_fields=[
                    SemanticField(field_name="context_tags"),
                    SemanticField(field_name="article_title")
                ],
                content_fields=[SemanticField(field_name="content")]
            )
        )

        semantic_settings = SemanticSettings(
            configurations=[semantic_config]
        )

        # Create the index
        index = SearchIndex(
            name=index_name,
            fields=fields,
            vector_search=vector_search,
            semantic_settings=semantic_settings
        )

        logger.info(f"Creating {index_name} search index...")
        index_client.create_or_update_index(index)
        logger.info(f"Index {index_name} created successfully.")

    except Exception as e:
        logger.error(f"Error creating index: {str(e)}")
        raise
```

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

File: /Users/collin/nfpa70-refactor/main.py
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

File: /Users/collin/nfpa70-refactor/azure_search/data_indexer.py
```py
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from openai import OpenAI
from typing import List, Dict, Any
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
        """
        Generate embeddings for text using OpenAI's API.
        
        Args:
            text: Text to generate embeddings for
            model: OpenAI embedding model to use
            
        Returns:
            List of embedding values
        """
        try:
            response = self.openai_client.embeddings.create(
                input=[text],
                model=model
            )
            return response.data[0].embedding
        except Exception as e:
            self.logger.error(f"Error generating embeddings: {str(e)}")
            raise

    def prepare_document(self, chunk: Dict[str, Any], chunk_id: int) -> Dict[str, Any]:
        """
        Prepare a document for indexing with all necessary fields and embeddings.
        
        Args:
            chunk: Processed text chunk with metadata
            chunk_id: Unique identifier for the chunk
            
        Returns:
            Document ready for indexing
        """
        content = chunk["content"]
        metadata = chunk.get("metadata", {})
        
        # Generate embedding for the content
        content_vector = self.generate_embeddings(content)
        
        # Prepare the document with all required fields
        document = {
            "id": f"doc_{chunk_id}",
            "content": content,
            "page_number": int(metadata.get("page", 0)),  # Ensure int type
            "article_number": str(metadata.get("article", "")),  # Ensure string type
            "article_title": chunk.get("article_title", ""),
            "section_number": str(metadata.get("section", "")),  # Ensure string type
            "section_title": chunk.get("section_title", ""),
            "content_vector": content_vector,
            "context_tags": chunk.get("context_tags", []),
            "related_sections": chunk.get("related_sections", []),
            "gpt_analysis": json.dumps(chunk.get("gpt_analysis", {}))  # Add GPT analysis
        }
        
        return document

    def index_documents(self, chunks: List[Dict[str, Any]]) -> None:
        """
        Index all documents with progress tracking and error handling.
        
        Args:
            chunks: List of processed text chunks to index
        """
        try:
            documents = []
            total_chunks = len(chunks)
            
            # Process chunks with progress bar
            for i in tqdm(range(total_chunks), desc="Processing chunks"):
                doc = self.prepare_document(chunks[i], i)
                documents.append(doc)
                
                # Upload in batches of 50 to avoid timeouts
                if len(documents) >= 50 or i == total_chunks - 1:
                    try:
                        results = self.search_client.upload_documents(documents=documents)
                        self.logger.info(f"Indexed batch of {len(results)} documents")
                        documents = []
                    except Exception as e:
                        self.logger.error(f"Error uploading batch: {str(e)}")
                        raise
            
        except Exception as e:
            self.logger.error(f"Error in indexing process: {str(e)}")
            raise

# Compatibility function for existing code
def index_documents(service_endpoint: str, admin_key: str, index_name: str,
                   chunks: List[Dict[str, Any]], openai_api_key: str) -> None:
    """
    Wrapper function for compatibility with existing code.
    """
    indexer = DataIndexer(service_endpoint, admin_key, index_name, openai_api_key)
    indexer.index_documents(chunks) 
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

File: /Users/collin/nfpa70-refactor/create_project_structure.sh
```sh
mkdir -p data_processing azure_search azure_function
touch main.py requirements.txt
touch data_processing/pdf_extractor.py data_processing/text_chunker.py
touch azure_search/index_creator.py azure_search/data_indexer.py
touch azure_function/function_app.py 
```

File: /Users/collin/nfpa70-refactor/test_pipeline.py
```py
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
```

File: /Users/collin/nfpa70-refactor/upgrade.md
```md
### Basic Implementation for Blob Storage


```python
from azure.storage.blob import BlobServiceClient, ContentSettings
from azure.core.exceptions import ResourceExistsError, ResourceNotFoundError
import json
import logging
import os

class BlobStorageManager:
    def __init__(self):
        self.connect_str = os.getenv('AzureWebJobsStorage')
        if not self.connect_str:
            raise ValueError("Missing AzureWebJobsStorage connection string")
        
        self.blob_service_client = BlobServiceClient.from_connection_string(self.connect_str)
        self.container_name = "processed-data"
        self.blob_name = "processed_data.json"

    def save_processed_data(self, data: dict) -> None:
        """Save data to Azure Blob Storage with retry logic"""
        try:
            container_client = self.blob_service_client.get_container_client(self.container_name)
            blob_client = container_client.get_blob_client(self.blob_name)
            
            blob_client.upload_blob(
                json.dumps(data),
                overwrite=True,
                max_concurrency=4,
                content_settings=ContentSettings(
                    content_type='application/json',
                    cache_control='no-cache'
                )
            )
            logging.info(f"Successfully saved data to blob: {self.blob_name}")
            
        except ResourceExistsError:
            logging.warning(f"Blob {self.blob_name} already exists - overwriting")
            raise
        except Exception as e:
            logging.error(f"Error saving data to blob: {str(e)}")
            raise

    def load_processed_data(self) -> dict:
        """Load data from Azure Blob Storage with retry logic"""
        try:
            container_client = self.blob_service_client.get_container_client(self.container_name)
            blob_client = container_client.get_blob_client(self.blob_name)
            
            downloaded_blob = blob_client.download_blob()
            return json.loads(downloaded_blob.readall())
            
        except ResourceNotFoundError:
            logging.warning(f"Blob {self.blob_name} not found")
            return None
        except Exception as e:
            logging.error(f"Error loading data from blob: {str(e)}")
            return None
```

### Advanced Implementation (with Async Support)
```python
from azure.storage.blob.aio import BlobServiceClient
from azure.core.exceptions import ResourceExistsError, ResourceNotFoundError
import json
import logging
import os
import asyncio

class AsyncBlobStorageManager:
    def __init__(self):
        self.connect_str = os.getenv('AzureWebJobsStorage')
        if not self.connect_str:
            raise ValueError("Missing AzureWebJobsStorage connection string")
        
        self.container_name = "processed-data"
        self.blob_name = "processed_data.json"

    async def save_processed_data(self, data: dict) -> None:
        """Save data to Azure Blob Storage asynchronously"""
        async with BlobServiceClient.from_connection_string(self.connect_str) as blob_service_client:
            container_client = blob_service_client.get_container_client(self.container_name)
            blob_client = container_client.get_blob_client(self.blob_name)
            
            try:
                await blob_client.upload_blob(
                    json.dumps(data),
                    overwrite=True,
                    max_concurrency=4,
                    content_settings=ContentSettings(
                        content_type='application/json',
                        cache_control='no-cache'
                    )
                )
                logging.info(f"Successfully saved data to blob: {self.blob_name}")
                
            except ResourceExistsError:
                logging.warning(f"Blob {self.blob_name} already exists - overwriting")
                raise
            except Exception as e:
                logging.error(f"Error saving data to blob: {str(e)}")
                raise

    async def load_processed_data(self) -> dict:
        """Load data from Azure Blob Storage asynchronously"""
        async with BlobServiceClient.from_connection_string(self.connect_str) as blob_service_client:
            container_client = blob_service_client.get_container_client(self.container_name)
            blob_client = container_client.get_blob_client(self.blob_name)
            
            try:
                downloaded_blob = await blob_client.download_blob()
                data = await downloaded_blob.readall()
                return json.loads(data)
                
            except ResourceNotFoundError:
                logging.warning(f"Blob {self.blob_name} not found")
                return None
            except Exception as e:
                logging.error(f"Error loading data from blob: {str(e)}")
                return None
```

## 3. Usage Examples

### Basic Usage
```python
# Initialize manager
blob_manager = BlobStorageManager()

# Save data
data = {"key": "value"}
blob_manager.save_processed_data(data)

# Load data
loaded_data = blob_manager.load_processed_data()
```

### Async Usage
```python
async def main():
    # Initialize async manager
    blob_manager = AsyncBlobStorageManager()

    # Save data
    data = {"key": "value"}
    await blob_manager.save_processed_data(data)

    # Load data
    loaded_data = await blob_manager.load_processed_data()

# Run async code
asyncio.run(main())
```






ANSWERS TO YOUR LIST OF POTENTIALLY DEPRECATED OR CHANGING CLASSES/METHODS/API

Here's a code implementation guide for Azure Search document batching and vector search using the stable 11.6.0 Python SDK:
python
# Document Batching with SearchIndexingBufferedSender
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchIndexingBufferedSender

# Initialize buffered sender
endpoint = "https://{service-name}.search.windows.net"
credential = AzureKeyCredential("<api-key>")

with SearchIndexingBufferedSender(
    endpoint=endpoint,
    index_name="your-index",
    credential=credential,
    auto_flush_interval=60,  # Seconds between auto-flushes
    max_retries_per_action=3
) as batch_client:
    
    # Example document list
    documents = [
        {"id": "1", "content": "First document text..."},
        {"id": "2", "content": "Second document text..."}
    ]
    
    try:
        # Batch upload documents
        batch_client.upload_documents(documents=documents)
        
        # Manual flush if needed
        batch_client.flush()
        
    except Exception as e:
        print(f"Indexing error: {e}")
python
# Vector Search Configuration
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchField, SearchFieldDataType,
    VectorSearch, HnswAlgorithmConfiguration,
    VectorSearchProfile
)

# Create vector-enabled index
index_client = SearchIndexClient(endpoint, credential)
fields = [
    SearchField(name="id", type=SearchFieldDataType.String, key=True),
    SearchField(
        name="contentVector",
        type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
        vector_search_dimensions=1536,
        vector_search_profile="myHnswProfile"
    )
]

vector_search = VectorSearch(
    algorithms=[HnswAlgorithmConfiguration(name="myHnsw")],
    profiles=[VectorSearchProfile(name="myHnswProfile", algorithm="myHnsw")]
)

index = SearchIndex(name="vector-index", fields=fields, vector_search=vector_search)
index_client.create_index(index)
Key Components Explained:
SearchIndexingBufferedSender Features:
Automatic batch processing (default 512 docs/batch)119
Built-in retry logic (3 attempts by default)18
Event callbacks for monitoring progress17
Thread-safe document queue management3
Vector Search Implementation:
VectorSearch class configures algorithms/profiles514
HnswAlgorithmConfiguration for approximate nearest neighbors510
Vector fields require:
vector_search_dimensions (e.g., 1536 for text-embedding-ada-002)
vector_search_profile linking to algorithm1014


Here's the corrected response with only stable GA components verified through official sources:
Stable Azure Component Versions (GA Only)
Component	Version	Source	Support Until
azure-search-documents	11.5.2	Azure SDK Changelog	2025-10-31
azure-storage-blob	12.28.0	Storage Blob Changelog	2026-04-30
openai	1.12.0	Azure OpenAI Docs	2025-09-30
Production-Ready Code Examples
1. Azure Search Configuration (GA)
python
# requirements.txt
azure-search-documents==11.5.2  # Last stable GA
azure-storage-blob==12.28.0     # Storage integration

# index_creator.py
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchField, SearchIndex, SimpleField, SearchFieldDataType
)

# Stable vector configuration
fields = [
    SimpleField(name="id", type=SearchFieldDataType.String, key=True),
    SearchField(
        name="contentVector",
        type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
        vector_search_dimensions=1536  # For text-embedding-3-small
    )
]

index_client.create_index(SearchIndex(name="nfpa-index", fields=fields))
2. OpenAI Embeddings (GA Models Only)
python
# data_indexer.py
from openai import AzureOpenAI

client = AzureOpenAI(
    api_version="2024-02-01",  # Stable API version
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
)

# Stable embedding models
response = client.embeddings.create(
    input="Article 250: Grounding Requirements",
    deployment_id="text-embedding-3-small",  # GA model
    encoding_format="float"
)
3. Blob Storage Integration
python
# pdf_extractor.py
from azure.storage.blob import BlobServiceClient

blob_client = BlobServiceClient(
    account_url="https://<storage>.blob.core.windows.net",
    credential=DefaultAzureCredential()
)

# Stable blob download
blob_data = blob_client.get_blob_client("nfpa-container", "nfpa70.pdf").download_blob()
Key Production Considerations
Version Locking
bash
pip freeze | grep azure > requirements.txt
Ensures no accidental upgrades to beta versions
Stable Embedding Models
Use: text-embedding-3-small (1536D) or text-embedding-3-large (3072D)
Retire: text-embedding-ada-002 by 2025-10-03 5
Azure Search Limits (GA)
python
MAX_BATCH_SIZE = 1000  # Documents per batch
MAX_REQUEST_SIZE = 16  # MB per batch
MAX_RETRIES = 3        # Built-in retry attempts
Monitoring
python
# Add to function_app.py
from azure.monitor import AzureMonitorMetricsExporter
exporter = AzureMonitorMetricsExporter(
    connection_string=os.getenv("APPINSIGHTS_CONNECTION_STRING")
)
Approved Dependency List
python
# Verified stable versions only
azure-common==1.1.28
azure-core==1.52.0
azure-functions==1.21.3        # Last GA functions SDK
azure-search-documents==11.5.2 # Stable search
azure-storage-blob==12.28.0    # Current GA
azure-storage-common==12.27.0
openai==1.12.0                 # OpenAI GA

Here's the compatibility analysis and code adjustments for your approved dependencies (GA versions only):
Approved Dependency List
python
azure-search-documents==11.5.2  # Search
azure-storage-blob==12.28.0     # Storage
openai==1.12.0                  # Embeddings
azure-functions==1.21.3         # Functions
PyPDF2==3.0.0                   # PDF Processing
1. PyPDF2 Implementation (v3.x)
Critical Changes:
PdfFileReader → PdfReader
numPages → len(reader.pages)
getPage() → pages[] index access
Stable Code Example:
python
from PyPDF2 import PdfReader

# Approved pattern
reader = PdfReader("nfpa70.pdf")
total_pages = len(reader.pages)  # Replaces numPages
first_page = reader.pages[0]      # Replaces getPage(0)

text = first_page.extract_text(
    orientations=(0, 90),  # Electrical code optimization
    space_width=200         # Technical doc spacing
)
Deprecations to Avoid:
python
# ❌ Deprecated in 3.x
from PyPDF2 import PdfFileReader  # Removed
reader.numPages                   # AttributeError
reader.getPage(0)                 # Method removed
2. Azure Functions Configuration
host.json Requirements:
json
{
  "version": "2.0",
  "extensionBundle": {
    "id": "Microsoft.Azure.Functions.ExtensionBundle",
    "version": "[4.0.0, 5.0.0)"
  }
}
runtime.txt:
text
python-3.11
Critical Compatibility:
Confirmed working with Python 3.11 in GA bundles
Monitor worker version via Kudu:
bash
cat /proc/version
3. SearchIndexClient Operations (11.5.2)
Stable Delete Pattern:
python
from azure.search.documents.indexes import SearchIndexClient
from azure.core.credentials import AzureKeyCredential

client = SearchIndexClient(
    endpoint="https://<service>.search.windows.net",
    credential=AzureKeyCredential("<key>")
)

# Approved deletion method
client.delete_index("nfpa-index")  # Direct name input
Error Handling:
python
from azure.core.exceptions import ResourceNotFoundError

try:
    client.delete_index("nfpa-index")
except ResourceNotFoundError:
    print("Index already removed - NFPA safety check")
4. Vector Configuration (Stable API)
Approved Vector Search Setup:
python
from azure.search.documents.indexes.models import (
    VectorSearch,
    HnswAlgorithmConfiguration,
    VectorSearchAlgorithmMetric
)

# Stable COSINE reference
vector_config = VectorSearch(
    algorithms=[
        HnswAlgorithmConfiguration(
            name="hnsw-nfpa",
            parameters={
                "m": 16,
                "efConstruction": 400,
                "metric": VectorSearchAlgorithmMetric.COSINE  # ✅ GA name
            }
        )
    ]
)
Vector Query Implementation:
python
from azure.search.documents.models import VectorQuery

# Approved parameter names
vector_query = VectorQuery(
    vector=[0.1, 0.2, ...],  # Your embedding
    fields="contentVector",
    k=10,
    exhaustive=True  # Electrical code precision
)
5. OpenAI Embedding Alignment
Stable Embedding Generation:
python
from openai import AzureOpenAI

client = AzureOpenAI(
    api_version="2024-02-01",  # GA version
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
)

# text-embedding-3-small (GA)
response = client.embeddings.create(
    input="Article 250: Grounding",
    deployment_id="text-embedding-3-small-deployment",
    dimensions=512  # Optional truncation
)
Compatibility Verification Script
bash
# Run in project root
pip freeze | grep -E 'azure-search-documents|azure-storage-blob|openai|azure-functions|PyPDF2'

# Expected output:
azure-search-documents==11.5.2
azure-storage-blob==12.28.0
openai==1.12.0
azure-functions==1.21.3
PyPDF2==3.0.0
Let me know if you need specific implementation details for any NFPA 70-related functionality!
```

File: /Users/collin/nfpa70-refactor/workflow.md
```md


## Table of Contents

1. [Directory Structure](#directory-structure)
2. [File Descriptions](#file-descriptions)
   - [1. `create_project_structure.sh`](#1-create_project_structuresh)
   - [2. `main.py`](#2-mainpy)
   - [3. `requirements.txt`](#3-requirementstxt)
   - [4. `.cursorrules`](#4-cursorrules)
   - [5. `azure_function/function_app.py`](#5-azure_functionfunction_apppy)
   - [6. `azure_search/index_creator.py`](#6-azure_searchindex_creatorpy)
   - [7. `azure_search/data_indexer.py`](#7-azure_searchdata_indexerpy)
   - [8. `data_processing/pdf_extractor.py`](#8-data_processingpdf_extractorpy)
   - [9. `data_processing/text_chunker.py`](#9-data_processingtext_chunkerpy)
3. [Summary of Project Workflow](#summary-of-project-workflow)

---

## Directory Structure

```
btcelectrician-nfpa70-refactor/
├── create_project_structure.sh
├── main.py
├── requirements.txt
├── .cursorrules
├── azure_function/
│   └── function_app.py
├── azure_search/
│   ├── data_indexer.py
│   └── index_creator.py
└── data_processing/
    ├── pdf_extractor.py
    └── text_chunker.py
```

---

## File Descriptions

### 1. `create_project_structure.sh`

**Purpose:**  
A shell script to initialize the project's directory structure and create essential files. This ensures consistency and sets up the necessary environment for development.

```bash
#!/bin/bash
mkdir -p data_processing azure_search azure_function
touch main.py requirements.txt
touch data_processing/pdf_extractor.py data_processing/text_chunker.py
touch azure_search/index_creator.py azure_search/data_indexer.py
touch azure_function/function_app.py 
```

---

### 2. `main.py`

**Purpose:**  
The entry point of the project. It orchestrates the extraction of text from a PDF, processes it into chunks, creates the Azure Search index, and indexes the processed chunks.

```python
import os
from dotenv import load_dotenv
import logging
from data_processing.pdf_extractor import PDFExtractor
from data_processing.text_chunker import ElectricalCodeChunker
from azure_search.index_creator import create_search_index
from azure_search.data_indexer import DataIndexer

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Main process to extract, process, and index electrical code content."""
    try:
        # Load environment variables
        load_dotenv()
        
        # Get required environment variables
        pdf_path = os.getenv('PDF_PATH')
        search_service_endpoint = os.getenv('AZURE_SEARCH_SERVICE_ENDPOINT')
        search_admin_key = os.getenv('AZURE_SEARCH_ADMIN_KEY')
        index_name = os.getenv('AZURE_SEARCH_INDEX_NAME', 'nfpa70-index')
        openai_api_key = os.getenv('OPENAI_API_KEY')

        # Validate environment variables
        required_vars = [
            'PDF_PATH',
            'AZURE_SEARCH_SERVICE_ENDPOINT',
            'AZURE_SEARCH_ADMIN_KEY',
            'OPENAI_API_KEY'
        ]
        
        missing_vars = [var for var in required_vars if not os.getenv(var)]
        if missing_vars:
            raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")

        # Extract text from PDF with advanced cleaning
        logger.info("Extracting text from PDF...")
        pdf_extractor = PDFExtractor()
        pages_text = pdf_extractor.extract_text_from_pdf(pdf_path)
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

---

### 3. `requirements.txt`

**Purpose:**  
Lists all Python dependencies required for the project, ensuring consistent environments across different setups.

```txt
# Azure Services
azure-functions>=1.17.0  # Latest stable version for Functions runtime v4
azure-search-documents>=11.5.2  # Latest stable version
azure-storage-blob>=12.23.0  # Latest stable version

# PDF Processing
PyMuPDF>=1.25.2  # Latest stable version with improved features

# AI/ML
openai>=1.59.8  # Latest version with new features
spacy>=3.7.2  # Latest stable version
transformers>=4.48.0  # Latest version with new models

# Utilities
python-dotenv>=1.0.1  # Latest version with bug fixes
python-slugify>=8.0.4  # Latest stable version
tqdm>=4.67.1  # Latest stable version
loguru>=0.7.3  # Latest stable version with fixes

# Install spaCy model
# Run after pip install: python -m spacy download en_core_web_sm
```

---

### 4. `.cursorrules`

**Purpose:**  
Defines commit message guidelines and development workflow rules to maintain consistency and quality in version control.

```markdown
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

Documentation:
* Maintain clear README with setup instructions
* Document API interactions and data flows
* Keep manifest.json well-documented
* Don't include comments unless it's for complex logic
* Document permission requirements

Development Workflow:
* Use proper version control
* Implement proper code review process
* Test in multiple environments
* Follow semantic versioning for releases
* Maintain changelog
```

---

### 5. `azure_function/function_app.py`

**Purpose:**  
Defines an Azure Function (`search_nfpa70`) that handles HTTP requests for searching the NFPA 70 electrical code. It leverages OpenAI embeddings for semantic search and filters results based on contextual tags.

```python
import azure.functions as func
import json
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from openai import OpenAI
import os
from typing import List, Dict, Optional
from loguru import logger

app = func.FunctionApp()
logger = logger

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

---

### 6. `azure_search/index_creator.py`

**Purpose:**  
Creates or updates an Azure Search index tailored for electrical code data. It defines the schema, including vector fields for semantic search, and configures semantic settings.

```python
from azure.core.credentials import AzureKeyCredential
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchIndex,
    SimpleField,
    SearchableField,
    SearchField,
    SearchFieldDataType,
    VectorSearch,
    HnswAlgorithmConfiguration,
    SemanticConfiguration,
    SemanticField,
    SemanticSettings,
    PrioritizedFields
)
from loguru import logger

def create_search_index(service_endpoint: str, admin_key: str, index_name: str) -> None:
    """
    Create an enhanced search index for electrical code content.
    
    Args:
        service_endpoint: Azure Search service endpoint
        admin_key: Azure Search admin key
        index_name: Name for the search index
    """
    try:
        credential = AzureKeyCredential(admin_key)
        index_client = SearchIndexClient(endpoint=service_endpoint, credential=credential)

        # Define fields for the index
        fields = [
            SimpleField(name="id", type=SearchFieldDataType.String, key=True),
            SearchableField(
                name="content", 
                type=SearchFieldDataType.String,
                analyzer_name="en.microsoft"
            ),
            SimpleField(
                name="page_number", 
                type=SearchFieldDataType.Int32,
                filterable=True,
                sortable=True
            ),
            SearchableField(
                name="article_number",
                type=SearchFieldDataType.String,
                filterable=True,
                facetable=True
            ),
            SearchableField(
                name="article_title",
                type=SearchFieldDataType.String,
                filterable=True
            ),
            SearchableField(
                name="section_number",
                type=SearchFieldDataType.String,
                filterable=True,
                facetable=True
            ),
            SearchableField(
                name="section_title",
                type=SearchFieldDataType.String,
                filterable=True
            ),
            SearchableField(
                name="context_tags",
                type=SearchFieldDataType.Collection(SearchFieldDataType.String),
                filterable=True,
                facetable=True,
                searchable=True
            ),
            SearchableField(
                name="related_sections",
                type=SearchFieldDataType.Collection(SearchFieldDataType.String),
                filterable=True
            ),
            # Vector field for semantic search
            SearchField(
                name="content_vector",
                type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                vector_search_dimensions=1536,
                vector_search_profile_name="my-vector-config",
            ),
        ]

        # Configure vector search
        vector_search = VectorSearch(
            algorithms=[
                HnswAlgorithmConfiguration(
                    name="my-vector-config",
                    kind="hnsw",
                    parameters={
                        "m": 4,
                        "efConstruction": 400,
                        "efSearch": 500,
                        "metric": "cosine"
                    }
                )
            ]
        )

        # Configure semantic search for field-oriented queries
        semantic_config = SemanticConfiguration(
            name="my-semantic-config",
            prioritized_fields=PrioritizedFields(
                title_field=SemanticField(field_name="section_title"),
                keywords_fields=[
                    SemanticField(field_name="context_tags"),
                    SemanticField(field_name="article_title")
                ],
                content_fields=[SemanticField(field_name="content")]
            )
        )

        semantic_settings = SemanticSettings(
            configurations=[semantic_config]
        )

        # Create the index
        index = SearchIndex(
            name=index_name,
            fields=fields,
            vector_search=vector_search,
            semantic_settings=semantic_settings
        )

        logger.info(f"Creating {index_name} search index...")
        index_client.create_or_update_index(index)
        logger.info(f"Index {index_name} created successfully.")

    except Exception as e:
        logger.error(f"Error creating index: {str(e)}")
        raise 
```

---

### 7. `azure_search/data_indexer.py`

**Purpose:**  
Handles the indexing of processed electrical code content into Azure Search. It generates embeddings using OpenAI, prepares documents with necessary fields, and uploads them in batches to Azure Search.

```python
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from openai import OpenAI
from typing import List, Dict, Any
from loguru import logger
from tqdm import tqdm

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
        """
        Generate embeddings for text using OpenAI's API.
        
        Args:
            text: Text to generate embeddings for
            model: OpenAI embedding model to use
            
        Returns:
            List of embedding values
        """
        try:
            response = self.openai_client.embeddings.create(
                input=[text],
                model=model
            )
            return response.data[0].embedding
        except Exception as e:
            self.logger.error(f"Error generating embeddings: {str(e)}")
            raise

    def prepare_document(self, chunk: Dict[str, Any], chunk_id: int) -> Dict[str, Any]:
        """
        Prepare a document for indexing with all necessary fields and embeddings.
        
        Args:
            chunk: Processed text chunk with metadata
            chunk_id: Unique identifier for the chunk
            
        Returns:
            Document ready for indexing
        """
        # Extract main content and metadata
        content = chunk["content"]
        metadata = chunk.get("metadata", {})
        
        # Generate embedding for the content
        content_vector = self.generate_embeddings(content)
        
        # Prepare the document with all necessary fields
        document = {
            "id": f"doc_{chunk_id}",
            "content": content,
            "page_number": metadata.get("page", 0),
            "article_number": metadata.get("article", ""),
            "section_number": metadata.get("section", ""),
            "content_vector": content_vector,
            "context_tags": chunk.get("context_tags", []),
            "related_sections": chunk.get("related_sections", [])
        }
        
        return document

    def index_documents(self, chunks: List[Dict[str, Any]]) -> None:
        """
        Index all documents with progress tracking and error handling.
        
        Args:
            chunks: List of processed text chunks to index
        """
        try:
            documents = []
            total_chunks = len(chunks)
            
            # Process chunks with progress bar
            for i in tqdm(range(total_chunks), desc="Processing chunks"):
                doc = self.prepare_document(chunks[i], i)
                documents.append(doc)
                
                # Upload in batches of 50 to avoid timeouts
                if len(documents) >= 50 or i == total_chunks - 1:
                    try:
                        results = self.search_client.upload_documents(documents=documents)
                        self.logger.info(f"Indexed batch of {len(results)} documents")
                        documents = []
                    except Exception as e:
                        self.logger.error(f"Error uploading batch: {str(e)}")
                        raise
                
        except Exception as e:
            self.logger.error(f"Error in indexing process: {str(e)}")
            raise

# Compatibility function for existing code
def index_documents(service_endpoint: str, admin_key: str, index_name: str,
                   chunks: List[Dict[str, Any]], openai_api_key: str) -> None:
    """
    Wrapper function for compatibility with existing code.
    """
    indexer = DataIndexer(service_endpoint, admin_key, index_name, openai_api_key)
    indexer.index_documents(chunks) 
```

---

### 8. `data_processing/pdf_extractor.py`

**Purpose:**  
Extracts and cleans text from PDF files using PyMuPDF (`fitz`). It handles common OCR artifacts and preserves essential electrical terminology to ensure accurate data processing.

```python
from pymupdf import fitz  # PyMuPDF 1.25.2
import re
from typing import Dict, Optional
import logging

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

    def extract_text_from_pdf(self, pdf_path: str) -> Dict[int, str]:
        """
        Extract text from PDF with page numbers and enhanced cleaning.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Dictionary mapping page numbers to cleaned text
        """
        try:
            self.logger.info(f"Processing PDF: {pdf_path}")
            doc = fitz.open(pdf_path)
            pages_text = {}
            
            for page_num in range(len(doc)):
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

---

### 9. `data_processing/text_chunker.py`

**Purpose:**  
Processes extracted text from the PDF into manageable and context-aware chunks. It identifies articles, sections, and assigns relevant context tags based on comprehensive NEC terminology.

```python
import re
from collections.abc import List, Dict
from typing import Optional
from dataclasses import dataclass, field
import logging

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

class ElectricalCodeChunker:
    """Enhanced chunking for electrical code text with comprehensive NEC terminology."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
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
        
        # Compile regex patterns
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
                    # Save previous chunk if exists
                    if current_chunk:
                        chunk_text = ' '.join(current_chunk)
                        chunks.append(CodeChunk(
                            content=chunk_text,
                            page_number=page_num,
                            article_number=current_article,
                            article_title=current_article_title,
                            section_number=current_section,
                            section_title=current_section_title,
                            context_tags=self._identify_context(chunk_text),
                            related_sections=self._find_related_sections(chunk_text)
                        ))
                    
                    current_section = section_match.group(1)
                    current_section_title = section_match.group(2).strip()
                    current_chunk = [line]
                else:
                    current_chunk.append(line)
            
            # Handle last chunk on page
            if current_chunk:
                chunk_text = ' '.join(current_chunk)
                chunks.append(CodeChunk(
                    content=chunk_text,
                    page_number=page_num,
                    article_number=current_article,
                    article_title=current_article_title,
                    section_number=current_section,
                    section_title=current_section_title,
                    context_tags=self._identify_context(chunk_text),
                    related_sections=self._find_related_sections(chunk_text)
                ))
        
        return chunks

# Compatibility function for existing code
def chunk_nfpa70_content(text: str) -> List[Dict]:
    """Wrapper for compatibility with existing code."""
    chunker = ElectricalCodeChunker()
    # Convert single text to page format expected by new chunker
    pages_text = {1: text}  # Assume single page for compatibility
    chunks = chunker.chunk_nfpa70_content(pages_text)
    
    # Convert to old format for compatibility
    return [
        {
            "content": chunk.content,
            "metadata": {
                "section": chunk.section_number,
                "article": chunk.article_number,
                "page": chunk.page_number
            },
            "context_tags": chunk.context_tags,
            "related_sections": chunk.related_sections
        }
        for chunk in chunks
    ] 
```

---

## Summary of Project Workflow

1. **Project Initialization:**
   - Run `create_project_structure.sh` to set up the necessary directories and create placeholder files.

2. **Environment Setup:**
   - Install all dependencies using `requirements.txt`:
     ```bash
     pip install -r requirements.txt
     ```
   - Download the required spaCy model:
     ```bash
     python -m spacy download en_core_web_sm
     ```

3. **PDF Text Extraction:**
   - Utilize `pdf_extractor.py` to extract and clean text from the NFPA 70 PDF document.
   - The extracted text is organized by page numbers and cleaned to remove OCR artifacts while preserving critical electrical terminology.

4. **Text Chunking:**
   - Use `text_chunker.py` to divide the cleaned text into logical chunks based on articles and sections.
   - Each chunk is enriched with context tags and references to related sections, facilitating more accurate searches.

5. **Azure Search Index Creation:**
   - Execute `index_creator.py` to define and create an Azure Search index tailored for the project's needs.
   - The index includes fields for content, metadata, context tags, related sections, and vector embeddings for semantic search.

6. **Document Indexing:**
   - Deploy `data_indexer.py` to generate embeddings for each text chunk using OpenAI's API.
   - The prepared documents are then uploaded in batches to Azure Search, ensuring efficient and reliable indexing.

7. **Azure Function Deployment:**
   - Set up the Azure Function defined in `function_app.py` to handle search queries.
   - The function parses user queries, generates embeddings, performs hybrid searches (combining keyword and semantic search), and returns formatted results.

8. **Execution:**
   - Run `main.py` to orchestrate the entire workflow: extracting text, processing it, creating the search index, and indexing the documents.
     ```bash
     python main.py
     ```

9. **Version Control and Documentation:**
   - Adhere to the commit message guidelines defined in `.cursorrules` to maintain a clear and consistent version history.
   - Ensure all API interactions, data flows, and permission requirements are well-documented for future reference and maintenance.

---

*This workflow document ensures that the BTC Electrician NFPA 70 Refactor project is well-organized, thoroughly documented, and easily maintainable. It provides all the necessary information to understand, deploy, and manage the project effectively.*
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
        
        # Compile regex patterns
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
        """Use GPT to analyze code chunk content."""
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
                model="gpt-4o-mini",  # Fast, affordable small model for focused tasks
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
                    # Save previous chunk if exists
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

# Compatibility function for existing code
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
</file_contents>


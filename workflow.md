# BTC Electrician NFPA 70 Refactor Project Workflow

*This document serves as a comprehensive offline record of the BTC Electrician NFPA 70 Refactor project. It includes the complete directory structure, detailed descriptions of each file, and their respective contents. This ensures that the project can be accurately rebuilt from scratch if necessary.*

---

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
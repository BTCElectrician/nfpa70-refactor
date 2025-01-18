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
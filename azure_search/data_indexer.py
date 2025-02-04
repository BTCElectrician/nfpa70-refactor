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
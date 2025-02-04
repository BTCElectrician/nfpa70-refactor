from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from openai import OpenAI
from typing import List, Dict, Any
from loguru import logger
from tqdm import tqdm
import json
import numpy as np

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
            response = self.openai_client.embeddings.create(
                input=[text],
                model=model
            )
            return response.data[0].embedding
        except Exception as e:
            self.logger.error(f"Error generating embeddings: {str(e)}")
            raise

    def prepare_document(self, chunk: Dict[str, Any], chunk_id: int) -> Dict[str, Any]:
        """Prepare a document for indexing with proper JSON handling."""
        try:
            self.logger.debug(f"[prepare_document] Input chunk structure: {json.dumps(chunk, indent=2)}")
            
            # Generate embedding for the content
            content_vector = self.generate_embeddings(chunk["content"])
            
            # Extract metadata fields
            metadata = chunk.get("metadata", {})
            self.logger.debug(f"[prepare_document] Extracted metadata: {metadata}")
            
            # Convert gpt_analysis to string if it's a dict
            gpt_analysis = chunk.get("gpt_analysis", {})
            if isinstance(gpt_analysis, (dict, list)):
                gpt_analysis = json.dumps(gpt_analysis)
                
            self.logger.debug(f"[prepare_document] Stringified gpt_analysis type: {type(gpt_analysis)}")
            
            # Convert numpy array to list and ensure float values
            if isinstance(content_vector, (np.ndarray, np.generic)):
                content_vector = content_vector.tolist()
            content_vector = [float(x) for x in content_vector]
            
            # Create document with proper field types
            document = {
                "id": f"doc_{chunk_id}",
                "content": chunk["content"],
                "page_number": metadata.get("page", 0),
                "article_number": str(metadata.get("article") or ""),
                "section_number": str(metadata.get("section") or ""),
                "article_title": chunk.get("article_title") or "",
                "section_title": chunk.get("section_title") or "",
                "content_vector": content_vector,
                "context_tags": list(chunk.get("context_tags") or []),
                "related_sections": list(chunk.get("related_sections") or []),
                "gpt_analysis": gpt_analysis
            }
            
            # Debug log the final structure
            self.logger.debug(f"[prepare_document] Final document structure prepared")
            
            return document
            
        except Exception as e:
            self.logger.error(f"Error preparing document: {str(e)}")
            raise

    def index_documents(self, chunks: List[Dict[str, Any]]) -> None:
        """Index all documents with progress tracking and error handling."""
        try:
            self.logger.debug(f"[index_documents] Starting indexing of {len(chunks)} chunks")
            documents = []
            total_chunks = len(chunks)
            
            # Process chunks with a progress bar
            for i in tqdm(range(total_chunks), desc="Processing chunks"):
                self.logger.debug(f"[index_documents] Processing chunk {i}")
                doc = self.prepare_document(chunks[i], i)
                documents.append(doc)
                
                # Upload in batches of 50 to avoid timeouts
                if len(documents) >= 50 or i == total_chunks - 1:
                    try:
                        self.logger.debug(f"[index_documents] Uploading batch of {len(documents)} documents")
                        results = self.search_client.upload_documents(documents=documents)
                        self.logger.info(f"Indexed batch of {len(results)} documents")
                        documents = []
                    except Exception as e:
                        self.logger.error(f"Error uploading batch: {str(e)}")
                        raise
            
        except Exception as e:
            self.logger.error(f"Error in indexing process: {str(e)}")
            raise
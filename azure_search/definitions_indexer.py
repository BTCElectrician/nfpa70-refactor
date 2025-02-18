import numpy as np
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from openai import OpenAI
from typing import List, Dict, Any, Optional
from loguru import logger
from tqdm import tqdm
import json

class DefinitionsIndexer:
    """Handles indexing of processed Article 100 definitions into Azure Search."""
    
    def __init__(self, service_endpoint: str, admin_key: str, index_name: str, openai_api_key: str):
        """Initialize the indexer with necessary credentials and configuration."""
        self.search_client = SearchClient(
            endpoint=service_endpoint,
            index_name=index_name,
            credential=AzureKeyCredential(admin_key)
        )
        self.openai_client = OpenAI(api_key=openai_api_key)
        self.logger = logger.bind(context="definitions_indexer")

    def generate_embeddings(self, text: str, model: str = "text-embedding-3-small") -> List[float]:
        """Generate embeddings for text using OpenAI's API."""
        try:
            # For definitions, we combine term and definition for better semantic search
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
            self.logger.error(f"Error generating embeddings: Type: {type(e)}, Error: {str(e)}")
            self.logger.debug(f"Exception attributes: {dir(e)}")
            raise

    def generate_embeddings_batch(self, texts: List[str], model: str = "text-embedding-3-small") -> List[List[float]]:
        """Generate embeddings for multiple texts in a single API call."""
        try:
            self.logger.debug(f"[generate_embeddings_batch] Generating embeddings for {len(texts)} texts")
            response = self.openai_client.embeddings.create(
                input=texts,
                model=model
            )
            # Convert each embedding to a list of floats
            embeddings = []
            for idx, record in enumerate(response.data):
                emb = [float(x) for x in record.embedding]
                # Verify dimension
                if len(emb) != 1536:
                    raise ValueError(f"Unexpected embedding dimension for text {idx}: {len(emb)}")
                embeddings.append(emb)

            self.logger.debug("[generate_embeddings_batch] Successfully generated batch embeddings")
            return embeddings
        except Exception as e:
            self.logger.error(f"Batch embedding error: Type: {type(e)}, Error: {str(e)}")
            self.logger.debug(f"Exception attributes: {dir(e)}")
            raise

    def prepare_definition_document(self, definition: Dict[str, Any], definition_id: str) -> Dict[str, Any]:
        """
        Prepare a definition document for indexing with proper vector handling.
        
        Args:
            definition: Dictionary containing definition fields
            definition_id: Unique identifier for the definition
            
        Returns:
            Document ready for indexing
        """
        try:
            self.logger.debug(f"[prepare_definition_document] Processing definition {definition_id}")
            
            # Combine term and definition for embedding to capture full semantic meaning
            combined_text = f"{definition['term']} - {definition['definition']}"
            content_vector = self.generate_embeddings(combined_text)
            
            # Create the final document for Azure Search
            document = {
                "id": definition_id,
                "term": definition["term"],
                "definition": definition["definition"],
                "page_number": definition["page_number"],
                "context": definition.get("context", ""),
                "content_vector": content_vector,
                "cross_references": definition.get("cross_references", []),
                "info_notes": definition.get("info_notes", []),
                "committee_refs": definition.get("committee_refs", []),
                "section_refs": definition.get("section_refs", [])
            }
            
            self._validate_document(document)
            self.logger.debug(f"[prepare_definition_document] Definition {definition_id} prepared successfully")
            
            return document
            
        except Exception as e:
            self.logger.error(f"Error preparing definition document {definition_id}: {str(e)}")
            raise

    def _validate_document(self, document: Dict[str, Any]) -> None:
        """Validate document structure before indexing."""
        required_fields = [
            "id", "term", "definition", "page_number", "content_vector"
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

    def index_definitions(self, definitions: List[Dict[str, Any]], batch_size: int = 50, embed_batch_size: int = 16) -> None:
        """
        Index definitions using batched embeddings and batched upload.
        
        Args:
            definitions: List of definition dictionaries
            batch_size: How many documents to upload in one batch to Azure Search
            embed_batch_size: How many definitions to embed in one call to OpenAI embeddings
        """
        try:
            total_definitions = len(definitions)
            self.logger.info(f"Starting indexing of {total_definitions} definitions")

            documents_to_upload = []
            
            # Process embeddings in sub-batches
            for start_idx in tqdm(range(0, total_definitions, embed_batch_size), desc="Embedding definitions"):
                end_idx = min(start_idx + embed_batch_size, total_definitions)
                batch_definitions = definitions[start_idx:end_idx]
                
                # Generate combined texts for embeddings
                batch_texts = [
                    f"{d['term']} - {d['definition']}" 
                    for d in batch_definitions
                ]

                # Generate batch embeddings
                embeddings = self.generate_embeddings_batch(batch_texts)

                # Build document objects
                for i, emb in enumerate(embeddings):
                    definition = batch_definitions[i]
                    definition_id = f"def_{start_idx + i}"
                    
                    document = {
                        "id": definition_id,
                        "term": definition["term"],
                        "definition": definition["definition"],
                        "page_number": definition["page_number"],
                        "context": definition.get("context", ""),
                        "content_vector": emb,
                        "cross_references": definition.get("cross_references", []),
                        "info_notes": definition.get("info_notes", []),
                        "committee_refs": definition.get("committee_refs", []),
                        "section_refs": definition.get("section_refs", [])
                    }

                    self._validate_document(document)
                    documents_to_upload.append(document)

                    # Upload in smaller batches
                    if len(documents_to_upload) >= batch_size:
                        self.logger.debug(f"Uploading batch of {len(documents_to_upload)} documents")
                        try:
                            results = self.search_client.upload_documents(documents=documents_to_upload)
                            self.logger.info(f"Successfully indexed batch of {len(results)} documents")
                            documents_to_upload = []
                        except Exception as e:
                            self.logger.error(f"Batch upload error: Type: {type(e)}, Error: {str(e)}")
                            self.logger.debug(f"Exception attributes: {dir(e)}")
                            raise

            # Upload any remaining documents
            if documents_to_upload:
                self.logger.debug(f"Uploading final batch of {len(documents_to_upload)} documents")
                try:
                    results = self.search_client.upload_documents(documents=documents_to_upload)
                    self.logger.info(f"Successfully indexed final batch of {len(results)} documents")
                except Exception as e:
                    self.logger.error(f"Batch upload error: Type: {type(e)}, Error: {str(e)}")
                    self.logger.debug(f"Exception attributes: {dir(e)}")
                    raise

        except Exception as e:
            self.logger.error(f"Fatal indexing error: Type: {type(e)}, Error: {str(e)}")
            self.logger.debug(f"Exception attributes: {dir(e)}")
            raise
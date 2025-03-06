import numpy as np
from azure.core.credentials import AzureKeyCredential
from azure.search.documents.aio import SearchClient as AsyncSearchClient
from openai import AsyncOpenAI, APIError, RateLimitError, APITimeoutError
from typing import List, Dict, Any, Optional
from loguru import logger
from tqdm import tqdm
import httpx
from tenacity import AsyncRetrying, stop_after_attempt, wait_exponential, retry_if_exception_type
import re

class DataIndexer:
    """Handles indexing of processed electrical code content into Azure Search."""
    
    async def __aenter__(self):
        """Enhanced HTTPX client configuration."""
        timeout = httpx.Timeout(
            connect=60.0,
            read=90.0,
            write=60.0,
            pool=30.0
        )
        limits = httpx.Limits(
            max_connections=100,
            max_keepalive_connections=20,
            keepalive_expiry=30.0
        )
        self.http_client = httpx.AsyncClient(
            timeout=timeout,
            limits=limits,
            follow_redirects=True,
            http2=True
        )
        if self.openai_api_key:
            self.openai_client = AsyncOpenAI(
                api_key=self.openai_api_key,
                http_client=self.http_client,
                max_retries=5,
                timeout=90.0
            )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Close the HTTP and OpenAI clients."""
        if hasattr(self, 'openai_client') and self.openai_client is not None:
            await self.openai_client.close()  # Explicitly close OpenAI client
            self.openai_client = None
        if hasattr(self, 'http_client') and self.http_client is not None:
            await self.http_client.aclose()  # Close HTTP client
            self.http_client = None
        # Close the AsyncSearchClient to prevent unclosed session warnings
        if hasattr(self, 'search_client') and self.search_client is not None:
            await self.search_client.close()
            self.search_client = None
        self.logger.debug("Cleaned up DataIndexer resources")

    def __init__(self, service_endpoint: str, admin_key: str, index_name: str, openai_api_key: str):
        """Initialize the indexer with necessary credentials and configuration."""
        self.search_client = AsyncSearchClient(
            endpoint=service_endpoint,
            index_name=index_name,
            credential=AzureKeyCredential(admin_key)
        )
        self.openai_api_key = openai_api_key
        self.logger = logger.bind(context="indexer")

    async def generate_embeddings(self, text: str, model: str = "text-embedding-3-small") -> List[float]:
        """Generate embeddings for text using OpenAI's API."""
        try:
            self.logger.debug(f"[generate_embeddings] Generating embedding for text (length: {len(text)})")
            response = await self.openai_client.embeddings.create(
                input=[text],
                model=model
            )
            embedding = [float(x) for x in response.data[0].embedding]
            if len(embedding) != 1536:
                raise ValueError(f"Unexpected embedding dimension: {len(embedding)}")
            self.logger.debug(f"[generate_embeddings] Successfully generated embedding with dimension {len(embedding)}")
            return embedding
        except Exception as e:
            self.logger.error(f"Error generating embeddings: Type: {type(e)}, Error: {str(e)}")
            raise

    async def generate_embeddings_batch(self, texts: List[str], model: str = "text-embedding-3-small") -> List[List[float]]:
        """Generate embeddings for multiple texts in a single API call."""
        try:
            self.logger.debug(f"[generate_embeddings_batch] Generating embeddings for {len(texts)} texts")
            response = await self.openai_client.embeddings.create(
                input=texts,
                model=model
            )
            embeddings = []
            for idx, record in enumerate(response.data):
                emb = [float(x) for x in record.embedding]
                if len(emb) != 1536:
                    raise ValueError(f"Unexpected embedding dimension for text {idx}: {len(emb)}")
                embeddings.append(emb)
            self.logger.debug("[generate_embeddings_batch] Successfully generated batch embeddings")
            return embeddings
        except Exception as e:
            self.logger.error(f"Batch embedding error: Type: {type(e)}, Error: {str(e)}")
            raise

    def prepare_document(self, chunk: Dict[str, Any], chunk_id: int, blob_name: str = None) -> Dict[str, Any]:
        """Prepare a document for indexing with proper vector handling."""
        try:
            self.logger.debug(f"[prepare_document] Processing chunk {chunk_id}")
            content_vector = self.generate_embeddings(chunk["content"])
            self.logger.debug(f"[prepare_document] Generated vector with shape: {len(content_vector)}")
            
            # Create a unique ID, using blob_name if provided
            doc_id = f"{blob_name}_{chunk_id}" if blob_name else f"doc_{chunk_id}"
            
            document = {
                "id": doc_id,
                "content": chunk["content"],
                "page_number": chunk.get("page_number", 0),
                "article_number": str(chunk.get("article_number") or ""),
                "section_number": str(chunk.get("section_number") or ""),
                "article_title": chunk.get("article_title") or "",
                "section_title": chunk.get("section_title") or "",
                "content_vector": content_vector,
                "context_tags": list(chunk.get("context_tags") or []),
                "related_sections": list(chunk.get("related_sections") or [])
            }
            self._validate_document(document)
            self.logger.debug(f"[prepare_document] Document {chunk_id} prepared successfully")
            return document
        except Exception as e:
            self.logger.error(f"Error preparing document {chunk_id}: {str(e)}")
            raise

    def _validate_document(self, document: Dict[str, Any]) -> None:
        """Validate document structure before indexing."""
        required_fields = ["id", "content", "page_number", "content_vector"]
        for field in required_fields:
            if field not in document:
                raise ValueError(f"Missing required field: {field}")
        if not isinstance(document["content_vector"], list):
            raise ValueError("content_vector must be a list")
        if len(document["content_vector"]) != 1536:
            raise ValueError(f"content_vector must have 1536 dimensions, got {len(document['content_vector'])}")
        if not all(isinstance(x, float) for x in document["content_vector"]):
            raise ValueError("All vector values must be float type")

    def _sanitize_key(self, key: str) -> str:
        """
        Azure Search requires that the key only contain letters, digits, underscore (_),
        dash (-), or equal sign (=). We'll replace anything else with '_'.
        """
        return re.sub(r'[^0-9A-Za-z_\-=]', '_', key)

    async def index_documents(
        self,
        chunks: List[Dict[str, Any]],
        blob_name: Optional[str] = None,
        batch_size: int = 25,
        embed_batch_size: int = 8
    ) -> None:
        """
        Index all documents using batched embeddings and batched async upload.
        Now we accept an optional blob_name to form unique doc IDs.
        """
        try:
            total_chunks = len(chunks)
            self.logger.info(f"Starting indexing of {total_chunks} chunks{f' from {blob_name}' if blob_name else ''}")
            documents_to_upload = []

            # We'll sanitize the blob name so it's a valid partial doc ID.
            # If we have no blob_name, just default to "doc".
            base_id_prefix = "doc" if not blob_name else self._sanitize_key(blob_name)

            for start_idx in tqdm(range(0, total_chunks, embed_batch_size), desc="Embedding chunks"):
                end_idx = min(start_idx + embed_batch_size, total_chunks)
                batch_texts = [c["content"] for c in chunks[start_idx:end_idx]]

                # Generate batch embeddings with retries
                try:
                    async for attempt in AsyncRetrying(
                        stop=stop_after_attempt(5),
                        wait=wait_exponential(multiplier=2, min=4, max=30),
                        retry=retry_if_exception_type((APITimeoutError, RateLimitError, APIError))
                    ):
                        with attempt:
                            embeddings = await self.generate_embeddings_batch(batch_texts)
                except Exception as e:
                    self.logger.error(f"Failed to generate embeddings after retries: {str(e)}")
                    raise

                # Build document objects
                for i, emb in enumerate(embeddings):
                    chunk_idx = start_idx + i
                    chunk = chunks[chunk_idx]

                    # Construct a unique doc ID. No invalid characters allowed.
                    doc_id = f"{base_id_prefix}_{chunk_idx}"

                    document = {
                        "id": doc_id,
                        "content": chunk.get("content", ""),
                        "page_number": chunk.get("page_number", 0),
                        "article_number": str(chunk.get("article_number") or ""),
                        "section_number": str(chunk.get("section_number") or ""),
                        "article_title": chunk.get("article_title") or "",
                        "section_title": chunk.get("section_title") or "",
                        "content_vector": emb,
                        "context_tags": list(chunk.get("context_tags") or []),
                        "related_sections": list(chunk.get("related_sections") or [])
                    }
                    self._validate_document(document)
                    documents_to_upload.append(document)

                    # Upload in smaller batches asynchronously
                    if len(documents_to_upload) >= batch_size:
                        self.logger.debug(f"Uploading batch of {len(documents_to_upload)} documents")
                        try:
                            async for attempt in AsyncRetrying(
                                stop=stop_after_attempt(5),
                                wait=wait_exponential(multiplier=2, min=4, max=30),
                                retry=retry_if_exception_type((APITimeoutError, RateLimitError, APIError))
                            ):
                                with attempt:
                                    results = await self.search_client.upload_documents(documents=documents_to_upload)
                                    self.logger.info(f"Successfully indexed batch of {len(results)} documents")
                                    documents_to_upload = []
                        except Exception as e:
                            self.logger.error(f"Batch upload error: Type: {type(e)}, Error: {str(e)}")
                            raise

            # Upload any remaining documents
            if documents_to_upload:
                self.logger.debug(f"Uploading final batch of {len(documents_to_upload)} documents")
                try:
                    async for attempt in AsyncRetrying(
                        stop=stop_after_attempt(5),
                        wait=wait_exponential(multiplier=2, min=4, max=30),
                        retry=retry_if_exception_type((APITimeoutError, RateLimitError, APIError))
                    ):
                        with attempt:
                            results = await self.search_client.upload_documents(documents=documents_to_upload)
                            self.logger.info(f"Successfully indexed final batch of {len(results)} documents")
                except Exception as e:
                    self.logger.error(f"Final batch upload error: Type: {type(e)}, Error: {str(e)}")
                    raise

        except Exception as e:
            self.logger.error(f"Fatal indexing error: Type: {type(e)}, Error: {str(e)}")
            raise
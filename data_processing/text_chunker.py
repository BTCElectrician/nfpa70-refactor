from typing import List, Dict, Optional
from loguru import logger
import asyncio
from openai import AsyncOpenAI
import json
import httpx
from data_processing.models import NFPAChunk, ChunkBatch
from tenacity import AsyncRetrying, stop_after_attempt, stop_after_delay, retry_if_exception_type, wait_exponential

# Constants for chunk validation
MAX_CHUNK_CHARS = 2000  # Conservative size for embedding
MIN_CHUNK_CHARS = 800   # Minimum for meaningful context

class ElectricalCodeChunker:
    """Simplified chunker for electrical code text using GPT-4."""
    
    def __init__(self, openai_api_key: Optional[str] = None):
        """Initialize the chunker with OpenAI credentials."""
        self.logger = logger.bind(context="chunker")
        self.openai_api_key = openai_api_key
        self.http_client: Optional[httpx.AsyncClient] = None
        self.client: Optional[AsyncOpenAI] = None

    async def __aenter__(self):
        """Set up async resources with optimized settings."""
        timeout = httpx.Timeout(
            connect=10.0,
            read=300.0,    # Increased for larger batch processing
            write=300.0,
            pool=60.0
        )
        
        limits = httpx.Limits(
            max_connections=100,
            max_keepalive_connections=20,
            keepalive_expiry=30.0
        )
        
        self.http_client = httpx.AsyncClient(
            timeout=timeout,
            limits=limits,
            http2=True
        )
        
        if self.openai_api_key:
            self.client = AsyncOpenAI(
                api_key=self.openai_api_key,
                http_client=self.http_client,
                max_retries=5
            )
            self.logger.debug("Initialized OpenAI client with optimized configuration")
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Clean up async resources."""
        if self.http_client:
            await self.http_client.aclose()
            self.http_client = None
        if self.client:
            await self.client.close()
            self.client = None
        self.logger.debug("Cleaned up HTTP client resources")

    def _validate_chunk_size(self, content: str) -> bool:
        """
        Validate chunk size is within optimal bounds for embeddings.
        
        Args:
            content: The text content to validate
            
        Returns:
            bool: True if chunk size is valid
        """
        char_count = len(content)
        return MIN_CHUNK_CHARS <= char_count <= MAX_CHUNK_CHARS

    async def _process_content_batch(self, pages_text: Dict[int, str]) -> List[Dict]:
        """
        Process a batch of pages using GPT-4 in a single call.
        
        Args:
            pages_text: Dictionary mapping page numbers to text content
            
        Returns:
            List of structured chunks with metadata
        """
        if not self.client:
            return []
        
        # Prepare concatenated content with page markers
        formatted_content = "\n\n".join(
            f"[PAGE {page_num}]\n{content}" 
            for page_num, content in sorted(pages_text.items())
        )
        
        try:
            # Enhanced retry configuration
            async for attempt in AsyncRetrying(
                stop=(
                    stop_after_attempt(3) |
                    stop_after_delay(600)  # 10 minute total timeout
                ),
                wait=wait_exponential(multiplier=1, min=4, max=10),
                retry=retry_if_exception_type((TimeoutError, httpx.HTTPError)),
                before_sleep=lambda retry_state: self.logger.warning(
                    f"Retrying batch processing after attempt {retry_state.attempt_number}"
                )
            ):
                with attempt:
                    self.logger.debug(f"Processing batch of {len(pages_text)} pages")
                    
                    response = await self.client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[{
                            "role": "system",
                            "content": """You are a precise NFPA 70 content processor. Process the provided electrical code content to:

1. Clean up OCR errors and normalize formatting
2. Structure content into logical sections
3. Extract metadata (article numbers, section numbers, titles)
4. Identify related sections and context tags

For each section, provide:
- Clean, corrected content
- Page number (format: 70-XX)
- Article number 
- Section number
- Article title
- Section title
- Context tags
- Related sections

Ensure:
- Content is complete and accurate
- Sections are properly linked
- Cross-page content is handled
- No information is lost

Return a JSON array of chunks, each containing all required fields."""
                        }, {
                            "role": "user",
                            "content": f"Process this electrical code content:\n\n{formatted_content}"
                        }],
                        temperature=0,
                        response_format={"type": "json_object"}
                    )
                    
                    result = json.loads(response.choices[0].message.content)
                    chunks = result.get("chunks", [])
                    self.logger.debug(f"Processed {len(chunks)} chunks from batch")
                    return chunks

        except Exception as e:
            self.logger.error(f"Batch processing failed: {str(e)}")
            self.logger.exception(e)
            return []

    async def chunk_nfpa70_content(self, pages_text: Dict[int, str]) -> List[NFPAChunk]:
        """
        Process NFPA 70 content into chunks using batched GPT processing.
        
        Args:
            pages_text: Dictionary mapping page numbers to text content
            
        Returns:
            List of NFPAChunk objects
        """
        if not pages_text:
            return []

        self.logger.info(f"Processing {len(pages_text)} pages")
        chunks = []
        seen_sections = set()  # Track unique sections
        
        try:
            # Process all pages in one batch
            processed_chunks = await self._process_content_batch(pages_text)
            
            # Convert to NFPAChunk objects with deduplication
            for chunk_data in processed_chunks:
                try:
                    nfpa_chunk = NFPAChunk(
                        content=chunk_data.get('content', ''),
                        page_number=chunk_data['page_number'],
                        article_number=chunk_data.get('article_number'),
                        section_number=chunk_data.get('section_number'),
                        article_title=chunk_data.get('article_title'),
                        section_title=chunk_data.get('section_title'),
                        context_tags=chunk_data.get('context_tags', []),
                        related_sections=chunk_data.get('related_sections', [])
                    )
                    
                    # Only add if section not seen before
                    section_key = (nfpa_chunk.article_number, nfpa_chunk.section_number)
                    if section_key not in seen_sections:
                        seen_sections.add(section_key)
                        chunks.append(nfpa_chunk)
                        self.logger.debug(f"Added chunk: {section_key}")
                    else:
                        self.logger.debug(f"Skipped duplicate section: {section_key}")
                        
                except ValueError as e:
                    self.logger.warning(f"Invalid chunk data: {str(e)}")
                    continue
            
            self.logger.success(f"Successfully processed {len(chunks)} chunks")
            return chunks
            
        except Exception as e:
            self.logger.error(f"Error processing content: {str(e)}")
            self.logger.exception(e)
            raise
import asyncio
from typing import List, Dict, Optional
from loguru import logger
from tenacity import AsyncRetrying, stop_after_attempt, stop_after_delay, retry_if_exception_type, wait_exponential
import json
from openai import AsyncOpenAI, APIError
from contextlib import asynccontextmanager
import httpx
import time
from data_processing.models import NFPAChunk, ChunkBatch

class ElectricalCodeChunker:
    """Chunks electrical code text using GPT-focused approach."""
    
    def __init__(self, openai_api_key: Optional[str] = None):
        self.logger = logger.bind(context="chunker")
        self.openai_api_key = openai_api_key
        self.http_client: Optional[httpx.AsyncClient] = None
        self.client: Optional[AsyncOpenAI] = None

    async def __aenter__(self):
        # Enhanced HTTPX configuration
        timeout = httpx.Timeout(
            connect=10.0,    # Connection timeout
            read=60.0,       # Read timeout
            write=60.0,      # Write timeout
            pool=60.0        # Pool timeout
        )
        limits = httpx.Limits(
            max_connections=100,              # Increased for better concurrency
            max_keepalive_connections=20,     # Optimal keepalive connections
            keepalive_expiry=30.0            # Connection expiry time
        )
        
        self.http_client = httpx.AsyncClient(
            timeout=timeout,
            limits=limits,
            http2=True  # Enable HTTP/2 for better performance
        )
        
        if self.openai_api_key:
            self.client = AsyncOpenAI(
                api_key=self.openai_api_key,
                http_client=self.http_client,
                max_retries=5
            )
            self.logger.debug("Initialized OpenAI client with enhanced HTTPX configuration")
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.http_client:
            await self.http_client.aclose()
            self.http_client = None
        if self.client:
            await self.client.close()
            self.client = None
        self.logger.debug("Cleaned up HTTP client resources")

    async def _process_page_with_gpt(self, pdf_page_num: int, text: str) -> List[Dict]:
        """Process a single page of text using GPT to clean and chunk in one pass."""
        if not self.client:
            return []
        
        start_time = time.time()
        try:
            # Enhanced retry configuration
            async for attempt in AsyncRetrying(
                stop=(
                    stop_after_attempt(3) |    # Stop after 3 attempts
                    stop_after_delay(30)        # Or after 30 seconds total
                ),
                wait=wait_exponential(multiplier=1, min=4, max=10),
                retry=retry_if_exception_type((TimeoutError, APIError, httpx.HTTPError)),
                before_sleep=lambda retry_state: self.logger.warning(
                    f"Retrying request after attempt {retry_state.attempt_number}")
            ):
                with attempt:
                    self.logger.debug(f"Making API request for page {pdf_page_num}, attempt {attempt.retry_state.attempt_number}")
                    
                    response = await self.client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[{
                            "role": "system",
                            "content": """Process this NFPA 70 page into properly structured chunks:
                            1. Clean ALL OCR errors (e.g., "GEJ'ERAL" → "GENERAL", "ELECfRIC..AL" → "ELECTRICAL")
                            2. Extract and structure each complete section with:
                               - Full section content including subsections
                               - Article number (e.g., "110")
                               - Section number (e.g., "110.3" or "110.3(A)")
                               - Article title
                               - Section title
                            3. Important requirements:
                               - Keep sections together with their subsections
                               - Include ALL content (notes, exceptions, etc.)
                               - Add relevant context_tags
                               - Add related_sections references
                               - Handle subsections correctly (e.g., "(A)", "(B)")
                               - Reject measurements as sections (e.g., "1.2 m" stays in parent section)
                            4. Return as JSON array of chunks with consistent structure
                            """
                        }, {
                            "role": "user",
                            "content": f"Process this page (PDF page {pdf_page_num}, NFPA page 70-{pdf_page_num-3}):\n\n{text}"
                        }],
                        timeout=120.0,
                        temperature=0,
                        response_format={"type": "json_object"}
                    )
                    
                    result = json.loads(response.choices[0].message.content)
                    chunks = result.get("chunks", [])
                    
                    # Ensure page number format
                    nfpa_page = f"70-{pdf_page_num-3}"
                    for chunk in chunks:
                        chunk["page_number"] = nfpa_page
                    
                    duration = time.time() - start_time
                    self.logger.debug(f"Processed page {pdf_page_num} into {len(chunks)} chunks in {duration:.2f}s")
                    return chunks

        except asyncio.TimeoutError as e:
            duration = time.time() - start_time
            self.logger.error(f"GPT timeout for page {pdf_page_num} after {duration:.2f}s: {str(e)}")
            return []
        except Exception as e:
            duration = time.time() - start_time
            self.logger.error(f"GPT processing failed for page {pdf_page_num} after {duration:.2f}s: {str(e)}")
            self.logger.exception(e)  # Add full traceback for debugging
            return []

    async def chunk_nfpa70_content(self, pages_text: Dict[int, str]) -> List[NFPAChunk]:
        """Process NFPA 70 content into chunks."""
        chunks = []
        seen_chunks = set()  # Deduplication by page and section
        
        if not pages_text:
            return chunks

        self.logger.info(f"Processing {len(pages_text)} pages")
        
        # Process pages concurrently with enhanced timeout
        tasks = [
            asyncio.wait_for(
                self._process_page_with_gpt(page_num, text),
                timeout=300.0  # 5-minute timeout per page
            )
            for page_num, text in sorted(pages_text.items())
        ]
        
        try:
            chunk_lists = await asyncio.gather(*tasks, return_exceptions=True)
            
            for page_chunks in chunk_lists:
                if isinstance(page_chunks, Exception):
                    self.logger.error(f"Failed to process page: {str(page_chunks)}")
                    self.logger.exception(page_chunks)  # Add full traceback
                    continue
                    
                for chunk_data in page_chunks:
                    try:
                        # Convert to NFPAChunk for validation
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
                        
                        key = (nfpa_chunk.page_number, nfpa_chunk.section_number or 'Article')
                        if key not in seen_chunks:
                            seen_chunks.add(key)
                            chunks.append(nfpa_chunk)
                            self.logger.debug(f"Added chunk: {key}")
                        else:
                            self.logger.debug(f"Skipped duplicate: {key}")
                    except ValueError as e:
                        self.logger.warning(f"Invalid chunk data: {str(e)}")
                        continue
            
            self.logger.success(f"Successfully processed {len(chunks)} chunks")
            return chunks
            
        except Exception as e:
            self.logger.error(f"Error processing pages: {str(e)}")
            self.logger.exception(e)  # Add full traceback
            raise
import asyncio
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from loguru import logger
from tenacity import AsyncRetrying, stop_after_attempt, retry_if_exception_type, wait_exponential
import json
from openai import AsyncOpenAI, APIError
from contextlib import asynccontextmanager
import httpx

@dataclass
class CodeChunk:
    """Represents a chunk of electrical code with essential metadata."""
    content: str                     # The actual text content of the chunk
    page_number: int                 # NFPA page number (e.g., 63 for "70-63")
    article_number: Optional[str]    # Article number (e.g., "230")
    section_number: Optional[str]    # Section number (e.g., "230.42")
    article_title: Optional[str]     # Title of the article (e.g., "Services")
    section_title: Optional[str]     # Title of the section (e.g., "Minimum Size and Ampacity")
    context_tags: List[str]          # Technical context tags
    related_sections: List[str]      # Referenced code sections

class ElectricalCodeChunker:
    """Chunks electrical code text using GPT-4o-mini for all splitting and analysis."""
    
    def __init__(
        self,
        openai_api_key: Optional[str] = None,
        batch_size: int = 5  # Optimized for ~2500 tokens per call
    ):
        """Initialize the chunker with OpenAI API key and batch settings."""
        self.logger = logger.bind(context="chunker")
        self.openai_api_key = openai_api_key
        
        self.http_client: Optional[httpx.AsyncClient] = None
        self.client: Optional[AsyncOpenAI] = None
        
        self.batch_size = batch_size
        self.api_semaphore = asyncio.Semaphore(self.batch_size)

    async def __aenter__(self):
        """Set up async context with HTTP client."""
        self.http_client = httpx.AsyncClient(timeout=60.0)
        if self.openai_api_key:
            self.client = AsyncOpenAI(
                api_key=self.openai_api_key,
                http_client=self.http_client,
                max_retries=5
            )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Clean up async context."""
        if self.http_client:
            await self.http_client.aclose()
            self.http_client = None
        if self.client:
            await self.client.close()
            self.client = None

    @asynccontextmanager
    async def _api_limit_guard(self):
        """Manage API rate limiting."""
        async with self.api_semaphore:
            yield

    async def _process_page_batch(self, pages: List[Tuple[int, str]]) -> List[Dict]:
        """Process full page texts with GPT-4o-mini to split and analyze."""
        if not self.client or not pages:
            return []
            
        page_numbers = [page[0] for page in pages]
        page_texts = [page[1] for page in pages]
        
        try:
            async for attempt in AsyncRetrying(
                stop=stop_after_attempt(3),
                wait=wait_exponential(multiplier=1, min=4, max=10),
                retry=retry_if_exception_type((TimeoutError, APIError))
            ):
                with attempt:
                    async with self._api_limit_guard():
                        response = await self.client.chat.completions.create(
                            model="gpt-4o-mini",
                            messages=[{
                                "role": "system", 
                                "content": """You are processing NFPA 70 (NEC) text. 
                                Input is a list of full page texts with their NFPA page numbers (e.g., 63 for "70-63").
                                For each page:
                                1. Clean OCR errors (e.g., "copJJer" → "copper", "bwial" → "burial").
                                2. Split the text into logical chunks at these levels:
                                   - New articles (e.g., "ARTICLE 230 - SERVICES")
                                   - Sections (e.g., "230.42 Minimum Size and Ampacity")
                                   - Subsections (e.g., "(A) General", "(1) Continuous and Noncontinuous Loads")
                                   - Tables (e.g., "Table 230.51(C) Supports") as separate chunks
                                   - Exceptions and Informational Notes as distinct chunks if standalone
                                3. For each chunk, return a JSON object with:
                                {
                                    "content": string,         # OCR-corrected chunk text
                                    "page_number": int,        # NFPA page number (e.g., 63)
                                    "article_number": string,  # e.g., "230"
                                    "section_number": string,  # e.g., "230.42"
                                    "article_title": string,   # e.g., "Services"
                                    "section_title": string,   # e.g., "Minimum Size and Ampacity"
                                    "context_tags": string[],  # e.g., "conductors", "grounding"
                                    "related_sections": string[]  # e.g., "110.14", "300.5"
                                }
                                Aim for 4-6 chunks per page to match document structure. Ensure each chunk retains its source page number."""
                            },
                            {
                                "role": "user", 
                                "content": json.dumps({
                                    "pages": page_texts,
                                    "page_numbers": page_numbers
                                })
                            }],
                            timeout=60.0,
                            temperature=0,
                            response_format={"type": "json_object"}
                        )
                        
                        try:
                            results = json.loads(response.choices[0].message.content)
                            chunks = results.get("chunks", [])
                            self.logger.debug(f"GPT returned {len(chunks)} chunks for {len(pages)} pages")
                            return chunks
                        except json.JSONDecodeError:
                            self.logger.error("Failed to parse GPT response")
                            return []
                        
        except Exception as e:
            self.logger.error(f"Error in GPT batch processing: {str(e)}")
            return []

    async def process_pages_async(self, pages: List[Tuple[int, str]]) -> List[Dict]:
        """Process full pages asynchronously with GPT."""
        self.logger.info(f"Starting processing of {len(pages)} pages")
        results = []
        
        sub_batches = []
        for i in range(0, len(pages), self.batch_size):
            batch_slice = pages[i:i + self.batch_size]
            sub_batches.append(batch_slice)
        
        tasks = [asyncio.create_task(self._process_page_batch(batch_slice)) for batch_slice in sub_batches]
        output_all = await asyncio.gather(*tasks)
        
        for partial_batch_result in output_all:
            results.extend(partial_batch_result)
            
        return results

    async def chunk_nfpa70_content(self, pages_text: Dict[int, str]) -> List[CodeChunk]:
        """Process NFPA 70 content into chunks using GPT-4o-mini."""
        chunks = []
        if not pages_text:
            return chunks

        # Prepare pages as (page_num, text) tuples
        pages = [(page_num, text) for page_num, text in sorted(pages_text.items())]
        self.logger.info(f"Processing {len(pages)} pages with GPT-4o-mini")

        # Process pages with GPT
        chunk_analyses = await self.process_pages_async(pages)
        
        # Convert results to CodeChunk objects
        for analysis in chunk_analyses:
            if not analysis:
                continue
            chunks.append(CodeChunk(
                content=analysis.get('content', ''),
                page_number=analysis.get('page_number', 0),
                article_number=analysis.get('article_number'),
                article_title=analysis.get('article_title'),
                section_number=analysis.get('section_number'),
                section_title=analysis.get('section_title'),
                context_tags=analysis.get('context_tags', []),
                related_sections=analysis.get('related_sections', [])
            ))

        self.logger.success(f"Successfully processed {len(chunks)} chunks")
        return chunks
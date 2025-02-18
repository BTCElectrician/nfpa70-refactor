import re
from typing import List, Dict, Optional, Any, Tuple, Sequence
from dataclasses import dataclass, field
import asyncio
from loguru import logger
from tenacity import AsyncRetrying, stop_after_attempt, retry_if_exception_type, wait_exponential
import json
from openai import AsyncOpenAI, APIError
from contextlib import asynccontextmanager
import httpx

@dataclass
class Definition:
    """Represents a single electrical code definition with metadata."""
    term: str
    definition: str
    page_number: int
    context: Optional[str] = None
    cross_references: List[str] = field(default_factory=list)
    info_notes: List[str] = field(default_factory=list)
    committee_refs: List[str] = field(default_factory=list)
    section_refs: List[str] = field(default_factory=list)

class DefinitionsChunker:
    """Process Article 100 definitions with GPT-based analysis."""
    
    def __init__(
        self,
        openai_api_key: Optional[str] = None,
        batch_size: int = 10,  # Reduced from 20 to optimal batch size
        max_concurrent_batches: int = 5  # Increased from 3 to optimal concurrency
    ):
        """Initialize the definitions chunker with optimized batch settings.
        
        Based on gpt-4o-mini performance characteristics:
        - Smaller batches process faster
        - 5 parallel requests complete in ~1.7 seconds
        - Larger parallel batches cause performance degradation
        """
        self.logger = logger.bind(context="definitions_chunker")
        self.openai_api_key = openai_api_key
        
        # Initialize as None, will be set in async context
        self.http_client: Optional[httpx.AsyncClient] = None
        self.client: Optional[AsyncOpenAI] = None
        
        self.batch_size = batch_size
        self.max_concurrent_batches = max_concurrent_batches
        self.api_semaphore = asyncio.Semaphore(self.max_concurrent_batches)

    async def __aenter__(self):
        """Set up async context with HTTP client."""
        self.http_client = httpx.AsyncClient(timeout=30.0)
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

    def _split_into_definition_chunks(self, text: str) -> List[str]:
        """Split text into potential definition chunks."""
        chunks = []
        lines = text.split('\n')
        current_chunk = []
        
        for line in lines:
            if not line.strip():
                continue
                
            if re.match(r'^[A-Z][a-zA-Z\s]+[\.\(]', line.strip()):
                if current_chunk:
                    chunks.append('\n'.join(current_chunk))
                current_chunk = [line]
            else:
                current_chunk.append(line)
                
        if current_chunk:
            chunks.append('\n'.join(current_chunk))
            
        return chunks

    async def _process_definition_batch(self, chunks: Sequence[str]) -> List[Dict]:
        """Process multiple definition chunks in a single GPT call."""
        if not self.client or not chunks:
            return [{} for _ in chunks]
            
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
                                "content": """You are processing NFPA 70 Article 100 definitions.
                                For each definition chunk, return a JSON object with:
                                {
                                    "term": string,         # The term being defined
                                    "context": string,      # Any parenthetical context (e.g., "as applied to...")
                                    "definition": string,   # The actual definition text
                                    "cross_references": string[],  # Referenced sections or articles
                                    "info_notes": string[],      # Any informational notes
                                    "committee_refs": string[],  # Committee references like (CMP-4)
                                    "section_refs": string[]     # Section references like (110)
                                }"""
                            },
                            {
                                "role": "user", 
                                "content": f"Process these Article 100 definition chunks: {json.dumps(chunks)}"
                            }],
                            timeout=45.0,
                            temperature=0,
                            response_format={"type": "json_object"}
                        )
                        
                        try:
                            results = json.loads(response.choices[0].message.content)
                            definitions = results.get("definitions", [])
                            # Ensure we return the same number of results as input chunks
                            while len(definitions) < len(chunks):
                                definitions.append({})
                            return definitions
                        except json.JSONDecodeError:
                            self.logger.error("Failed to parse GPT response")
                            return [{} for _ in chunks]
                        
        except Exception as e:
            self.logger.error(f"Error in GPT batch processing: {str(e)}")
            return [{} for _ in chunks]

    async def process_definitions_async(self, chunks: List[str]) -> List[Dict]:
        """Process definition chunks with proper async session handling."""
        self.logger.info(f"Starting parallel processing of {len(chunks)} definition chunks")
        results = []
        
        # Create sub-batches
        sub_batches = []
        for i in range(0, len(chunks), self.batch_size):
            batch_slice = chunks[i:i + self.batch_size]
            sub_batches.append(batch_slice)
            
        self.logger.info(f"Created {len(sub_batches)} batches of up to {self.batch_size} chunks each")
        
        # Process in parallel with controlled concurrency
        tasks = []
        for batch_slice in sub_batches:
            tasks.append(asyncio.create_task(self._process_definition_batch(batch_slice)))
            
        # Run batches with controlled concurrency
        output_all = []
        for i in range(0, len(tasks), self.max_concurrent_batches):
            slice_of_tasks = tasks[i:i + self.max_concurrent_batches]
            partial_results = await asyncio.gather(*slice_of_tasks)
            output_all.extend(partial_results)
            
        # Flatten results
        for partial_batch_result in output_all:
            results.extend(partial_batch_result)
            
        return results

    async def process_article_100(self, pages_text: Dict[int, str]) -> List[Definition]:
        """Process Article 100 content into Definition objects."""
        definitions = []
        raw_chunks = []
        
        if not pages_text:
            return definitions

        # First pass: Split text into potential definition chunks
        for page_num, text in sorted(pages_text.items()):
            chunks = self._split_into_definition_chunks(text)
            for chunk in chunks:
                raw_chunks.append((page_num, chunk))

        # Process chunks with GPT
        self.logger.info(f"Processing {len(raw_chunks)} definition chunks")
        chunk_texts = [chunk[1] for chunk in raw_chunks]
        analyses = await self.process_definitions_async(chunk_texts)
        
        # Convert results to Definition objects with proper index validation
        for i, analysis in enumerate(analyses):
            if i >= len(raw_chunks):  # Guard against index errors
                self.logger.warning(f"Skipping analysis {i} - no matching raw chunk")
                break
                
            if not analysis:  # Skip empty results
                continue
                
            try:
                page_num = raw_chunks[i][0]
                definitions.append(Definition(
                    term=analysis.get('term', ''),
                    definition=analysis.get('definition', ''),
                    page_number=page_num,
                    context=analysis.get('context'),
                    cross_references=analysis.get('cross_references', []),
                    info_notes=analysis.get('info_notes', []),
                    committee_refs=analysis.get('committee_refs', []),
                    section_refs=analysis.get('section_refs', [])
                ))
            except Exception as e:
                self.logger.error(f"Error processing chunk {i}: {str(e)}")
                continue

        self.logger.success(f"Successfully processed {len(definitions)} definitions")
        return definitions
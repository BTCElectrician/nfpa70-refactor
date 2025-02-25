import re
import time
import asyncio
import json
from typing import List, Dict, Optional, Any, Tuple, Sequence, Set, Union
from contextlib import asynccontextmanager
from datetime import datetime

import httpx
from loguru import logger
from tenacity import (
    AsyncRetrying, stop_after_attempt, retry_if_exception_type,
    wait_exponential, RetryError
)
from openai import AsyncOpenAI, RateLimitError, APITimeoutError, APIError

# Import from our models module
from data_processing.models import (
    CodePosition, CodeChunk, ChunkBatch, 
    ProcessingStatus, TokenUsageMetrics
)

# Import performance monitoring
import sys
import os
# Add path to utils if not in path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from utils.monitoring import PerformanceMonitor


class ChunkerError(Exception):
    """Base error for chunker operations."""
    pass


class GPTProcessingError(ChunkerError):
    """Error when GPT processing fails."""
    pass


class BatchProcessingError(ChunkerError):
    """Error when batch processing fails."""
    def __init__(self, message: str, batch_index: int, retry_count: int = 0):
        super().__init__(message)
        self.batch_index = batch_index
        self.retry_count = retry_count


class NEC70TextChunker:
    """
    Enhanced chunker for NFPA 70 (NEC) text with optimized GPT-based processing.
    
    This class handles:
    1. Breaking document text into logical chunks by article/section
    2. Processing chunks with GPT to extract metadata and clean text
    3. Organizing chunks with proper context and relationships
    """
    
    # Technical context mapping for electrical code categorization
    CONTEXT_MAPPING = {
        'service_equipment': [
            'service equipment', 'service entrance', 'service drop',
            'meter', 'service disconnect', 'main disconnect'
        ],
        'conductors': [
            'conductor', 'wire', 'cable', 'AWG', 'kcmil',
            'copper', 'aluminum'
        ],
        'raceway': [
            'EMT', 'IMC', 'RMC', 'PVC', 'conduit',
            'electrical metallic tubing'
        ],
        'grounding': [
            'ground', 'grounding', 'bonding', 'GEC',
            'equipment grounding conductor'
        ],
        'overcurrent': [
            'breaker', 'circuit breaker', 'fuse', 'overcurrent',
            'AFCI', 'GFCI'
        ],
        # Additional categories for comprehensive tagging
        'hazardous_locations': [
            'classified location', 'Class I', 'Class II', 'Class III',
            'Division 1', 'Division 2', 'explosion', 'hazardous'
        ],
        'healthcare': [
            'healthcare', 'hospital', 'nursing', 'medical', 'patient care'
        ]
    }
    
    def __init__(
        self,
        openai_api_key: Optional[str] = None,
        model: str = "gpt-4o-mini",
        batch_size: int = 5,
        max_concurrent_batches: int = 2,
        timeout: float = 120.0,
        max_retries: int = 5
    ):
        """Initialize the chunker with configuration parameters."""
        self.logger = logger.bind(context="chunker")
        self.openai_api_key = openai_api_key
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries
        
        # Batch processing configuration
        self.batch_size = batch_size
        self.max_concurrent_batches = max_concurrent_batches
        self.api_semaphore = asyncio.Semaphore(self.max_concurrent_batches)
        
        # Initialize tracking components
        self.token_usage = TokenUsageMetrics()
        self.monitor = PerformanceMonitor("NEC70TextChunker")
        
        # Initialize as None, will be set in async context
        self.http_client: Optional[httpx.AsyncClient] = None
        self.client: Optional[AsyncOpenAI] = None
        
        self.logger.info(
            f"Initialized chunker with batch_size={batch_size}, "
            f"max_concurrent_batches={max_concurrent_batches}, "
            f"model={model}"
        )

    async def __aenter__(self):
        """Async context manager entry - initializes httpx client and OpenAI client."""
        # Configure timeouts for different operations
        timeout = httpx.Timeout(
            connect=30.0,     # Connection timeout
            read=self.timeout, # Read timeout
            write=60.0,       # Write timeout
            pool=30.0         # Pool timeout
        )
        
        # Configure connection limits
        limits = httpx.Limits(
            max_connections=100,
            max_keepalive_connections=20,
            keepalive_expiry=30.0
        )
        
        # Create HTTP client with resilient configuration
        self.http_client = httpx.AsyncClient(
            timeout=timeout,
            limits=limits,
            follow_redirects=True,
            http2=True  # Enable HTTP/2 for better performance
        )
        
        # Create OpenAI client if we have an API key
        if self.openai_api_key:
            self.client = AsyncOpenAI(
                api_key=self.openai_api_key,
                http_client=self.http_client,
                max_retries=5,
                timeout=self.timeout
            )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit - properly closes httpx client."""
        self.monitor.log_summary()
        
        if self.http_client:
            await self.http_client.aclose()
            self.http_client = None
        if self.client:
            await self.client.close()
            self.client = None

    @asynccontextmanager
    async def _api_limit_guard(self):
        """Context manager for API rate limiting."""
        async with self.api_semaphore:
            yield

    def _get_enhanced_system_prompt(self) -> str:
        """
        Generate an enhanced system prompt for GPT with detailed instructions
        for NEC text processing.
        """
        categories = ", ".join(self.CONTEXT_MAPPING.keys())
        
        return f"""You are a specialized electrical code analyzer for NFPA 70 (National Electrical Code).
Your task is to process raw text from the NEC and extract structured information while cleaning OCR errors.

# Important: Return your output as a JSON object with a 'chunks' array
# containing all processed chunks, like this:
{{
  "chunks": [
    {{
      "content": "cleaned text",
      "article_number": "110",
      "article_title": "title",
      ...other fields...
    }},
    {{
      ...next chunk...
    }}
  ]
}}

For each chunk of text, analyze it carefully to identify:
1. Article number and title (e.g., "ARTICLE 110 - Requirements for Electrical Installations")
2. Section number and title (e.g., "110.12 Mechanical Execution of Work")
3. Technical context - identify which categories apply from: {categories}
4. Referenced sections - any other NEC sections mentioned (e.g., "as required in 230.70")

Clean the text by:
- Correcting obvious OCR errors (e.g., "l00 amperes" should be "100 amperes")
- Maintaining proper formatting including paragraph breaks
- Preserving section numbering and lettering
- Preserving lists and indentation when present

For each text chunk, return a JSON object with:
{{
  "content": string,       # The cleaned, corrected text
  "article_number": string,  # Just the number (e.g., "110")
  "article_title": string,   # Just the title (e.g., "Requirements for Electrical Installations")
  "section_number": string,  # Full section number if present (e.g., "110.12")
  "section_title": string,   # Just the section title if present
  "context_tags": string[],  # Relevant technical categories
  "related_sections": string[]  # Referenced code sections (e.g., ["230.70", "408.36"])
}}

If an article or section cannot be identified, use null for those fields.
If the text is unintelligible or severely corrupted, do your best to reconstruct it based on context.
"""

    async def _process_chunk_batch(
        self, 
        chunks: Sequence[str], 
        batch_index: int
    ) -> Tuple[List[Dict], Dict[str, Any]]:
        """
        Process multiple chunks in a single GPT call with robust error handling
        and detailed metrics tracking.
        
        Args:
            chunks: List of text chunks to process
            batch_index: Index of this batch for tracking
            
        Returns:
            Tuple of (list of processed chunks as dicts, processing metadata)
        """
        if not self.client or not chunks:
            self.logger.warning(f"Cannot process batch {batch_index}: no client or empty chunks")
            return [{} for _ in chunks], {"success": False, "reason": "no_client_or_empty"}
            
        batch_start = time.time()
        metadata = {
            "batch_index": batch_index,
            "chunk_count": len(chunks),
            "success": False,
            "retry_count": 0,
            "tokens": {"prompt": 0, "completion": 0, "total": 0}
        }
        
        try:
            async for attempt in AsyncRetrying(
                stop=stop_after_attempt(self.max_retries),
                wait=wait_exponential(multiplier=2, min=4, max=60),
                retry=retry_if_exception_type((APITimeoutError, RateLimitError, APIError)),
                reraise=True
            ):
                attempt_start = time.time()
                metadata["retry_count"] += 1
                
                with attempt:
                    async with self._api_limit_guard():
                        self.logger.debug(
                            f"Processing batch {batch_index} (attempt {metadata['retry_count']}): "
                            f"{len(chunks)} chunks"
                        )
                        
                        # Create chat completion with enhanced system prompt
                        response = await self.client.chat.completions.create(
                            model=self.model,
                            messages=[
                                {
                                    "role": "system", 
                                    "content": self._get_enhanced_system_prompt()
                                },
                                {
                                    "role": "user", 
                                    "content": f"Process these NEC text chunks: {json.dumps(chunks)}"
                                }
                            ],
                            temperature=0,
                            response_format={"type": "json_object"},
                        )
                        
                        # Track token usage
                        if hasattr(response, 'usage'):
                            metadata["tokens"]["prompt"] = response.usage.prompt_tokens
                            metadata["tokens"]["completion"] = response.usage.completion_tokens
                            metadata["tokens"]["total"] = response.usage.total_tokens
                            
                            self.token_usage.update(
                                response.usage.prompt_tokens,
                                response.usage.completion_tokens
                            )
                        
                        # Parse response content
                        try:
                            json_content = response.choices[0].message.content
                            results = json.loads(json_content)
                            
                            # If we get a direct response with 'content' field (not in chunks array)
                            if "content" in results and isinstance(results["content"], str):
                                # Create a proper chunk structure from the single content
                                chunks_result = [{
                                    "content": results.get("content", ""),
                                    "article_number": results.get("article_number"),
                                    "section_number": results.get("section_number"),
                                    "article_title": results.get("article_title"),
                                    "section_title": results.get("section_title"),
                                    "context_tags": results.get("context_tags", []),
                                    "related_sections": results.get("related_sections", [])
                                }]
                            else:
                                # Try to get the chunks array as originally expected
                                chunks_result = results.get("chunks", [])
                                
                            # If we still don't have results in the expected format
                            if not chunks_result:
                                # Try parsing as an array directly
                                if isinstance(results, list):
                                    chunks_result = results
                                    
                            # If we still don't have results in the expected format
                            if not chunks_result:
                                # Log the issue and return the raw data for manual inspection
                                self.logger.warning(
                                    f"Unexpected GPT response format in batch {batch_index}. "
                                    f"Response: {json_content[:500]}..."
                                )
                                metadata["success"] = False
                                metadata["reason"] = "unexpected_format"
                                return [{} for _ in chunks], metadata
                            
                            # If we don't have the right number of results
                            if len(chunks_result) != len(chunks):
                                self.logger.warning(
                                    f"GPT returned {len(chunks_result)} chunks but expected {len(chunks)} "
                                    f"in batch {batch_index}"
                                )
                                # Pad with empty results if needed
                                while len(chunks_result) < len(chunks):
                                    chunks_result.append({})
                                # Truncate if too many
                                chunks_result = chunks_result[:len(chunks)]
                            
                            metadata["success"] = True
                            metadata["processing_time"] = time.time() - attempt_start
                            self.logger.debug(
                                f"Successfully processed batch {batch_index} in "
                                f"{metadata['processing_time']:.2f}s"
                            )
                            return chunks_result, metadata
                            
                        except json.JSONDecodeError as e:
                            self.logger.error(
                                f"Failed to parse JSON from GPT response in batch {batch_index}: {str(e)}"
                            )
                            metadata["success"] = False
                            metadata["reason"] = "json_decode_error"
                            metadata["error"] = str(e)
                            raise GPTProcessingError(f"JSON decode error: {str(e)}")
                        
        except RetryError as e:
            self.logger.error(f"Max retries exceeded for batch {batch_index}: {str(e)}")
            metadata["success"] = False
            metadata["reason"] = "max_retries_exceeded"
            metadata["error"] = str(e)
        except APITimeoutError as e:
            self.logger.error(f"API timeout in batch {batch_index}: {str(e)}")
            metadata["success"] = False
            metadata["reason"] = "api_timeout"
            metadata["error"] = str(e)
        except RateLimitError as e:
            self.logger.error(f"Rate limit exceeded in batch {batch_index}: {str(e)}")
            metadata["success"] = False
            metadata["reason"] = "rate_limit"
            metadata["error"] = str(e)
        except APIError as e:
            self.logger.error(f"API error in batch {batch_index}: {str(e)}")
            metadata["success"] = False
            metadata["reason"] = "api_error"
            metadata["error"] = str(e)
        except Exception as e:
            self.logger.error(
                f"Unexpected error in batch {batch_index}: {type(e).__name__}: {str(e)}"
            )
            metadata["success"] = False
            metadata["reason"] = "unexpected_error"
            metadata["error"] = f"{type(e).__name__}: {str(e)}"
            
        # If we get here, something went wrong
        metadata["processing_time"] = time.time() - batch_start
        return [{} for _ in chunks], metadata

    async def process_chunks_async(self, chunks: List[str]) -> Tuple[List[Dict], Dict[str, Any]]:
        """
        Process all chunks with parallel batch processing and comprehensive metrics.
        
        Args:
            chunks: List of text chunks to process
            
        Returns:
            Tuple of (processed chunks, processing metadata)
        """
        with self.monitor.measure("process_chunks_async", chunk_count=len(chunks)) as metrics:
            self.logger.info(f"Starting parallel processing of {len(chunks)} chunks")
            
            # Track overall processing
            process_metadata = {
                "total_chunks": len(chunks),
                "start_time": datetime.now().isoformat(),
                "batches": [],
                "success_rate": 0,
                "token_usage": {}
            }
            
            # Create sub-batches
            sub_batches = []
            for i in range(0, len(chunks), self.batch_size):
                batch_slice = chunks[i:i + self.batch_size]
                sub_batches.append(batch_slice)
                
            self.logger.info(f"Split into {len(sub_batches)} batches of up to {self.batch_size} chunks each")
            
            # Process in parallel with controlled concurrency
            all_processed_chunks = []
            
            # Process batches in sets with controlled concurrency
            for batch_set_idx in range(0, len(sub_batches), self.max_concurrent_batches):
                batch_set = sub_batches[batch_set_idx:batch_set_idx + self.max_concurrent_batches]
                batch_set_start = time.time()
                
                # Create tasks for this set of batches
                tasks = []
                for i, batch_slice in enumerate(batch_set):
                    batch_idx = batch_set_idx + i
                    tasks.append(asyncio.create_task(
                        self._process_chunk_batch(batch_slice, batch_idx)
                    ))
                
                # Wait for all tasks in this set to complete
                batch_results = await asyncio.gather(*tasks, return_exceptions=False)
                batch_set_time = time.time() - batch_set_start
                
                # Extract the results and metadata
                for i, (batch_chunks, batch_meta) in enumerate(batch_results):
                    batch_idx = batch_set_idx + i
                    process_metadata["batches"].append(batch_meta)
                    all_processed_chunks.extend(batch_chunks)
                    
                    # Log summary for this batch
                    success_status = "✓" if batch_meta["success"] else "✗"
                    self.logger.info(
                        f"Batch {batch_idx}/{len(sub_batches)} {success_status} "
                        f"({len(batch_chunks)} chunks) in {batch_meta.get('processing_time', 0):.2f}s"
                    )
                
                # Log summary for this set of batches
                self.logger.info(
                    f"Completed batch set {batch_set_idx//self.max_concurrent_batches + 1} "
                    f"({len(batch_set)}/{len(sub_batches)} batches) in {batch_set_time:.2f}s"
                )
            
            # Calculate overall success rate
            successful_batches = sum(1 for meta in process_metadata["batches"] if meta["success"])
            process_metadata["success_rate"] = (successful_batches / len(process_metadata["batches"])) * 100
            
            # Add token usage summary
            process_metadata["token_usage"] = {
                "prompt_tokens": self.token_usage.prompt_tokens,
                "completion_tokens": self.token_usage.completion_tokens,
                "total_tokens": self.token_usage.total_tokens
            }
            
            # Calculate overall processing time
            process_metadata["end_time"] = datetime.now().isoformat()
            process_metadata["duration"] = metrics.duration
            
            self.logger.info(
                f"Completed processing {len(chunks)} chunks in {metrics.duration:.2f}s "
                f"with {process_metadata['success_rate']:.1f}% batch success rate"
            )
            
            metrics.metadata.update({"success_rate": process_metadata["success_rate"]})
            return all_processed_chunks, process_metadata

    def _extract_raw_chunks(self, pages_text: Dict[int, str]) -> List[str]:
        """
        Extract raw text chunks from the input document by identifying article and section breaks.
        
        Args:
            pages_text: Dictionary mapping page numbers to text content
            
        Returns:
            List of raw text chunks
        """
        with self.monitor.measure("extract_raw_chunks", page_count=len(pages_text)) as metrics:
            raw_chunks = []
            
            # Process pages in numerical order
            min_page = min(pages_text.keys()) if pages_text else 0
            start_page = max(26, min_page)  # Start from page 26 or minimum available
            
            for page_num in sorted(k for k in pages_text if k >= start_page):
                text = pages_text[page_num]
                lines = text.split('\n')
                current_chunk = []
                
                # Extract chunks with logical breaks at articles and sections
                for line in lines:
                    line_stripped = line.strip()
                    if not line_stripped:
                        continue

                    # Start a new chunk at article or section boundary
                    if (re.search(r'ARTICLE\s+\d+', line_stripped, re.IGNORECASE) or 
                        re.search(r'^\d+\.\d+\s+[A-Z]', line_stripped)):
                        if current_chunk:
                            raw_chunks.append(' '.join(current_chunk))
                        current_chunk = [line]
                    else:
                        current_chunk.append(line)
                
                # Add the last chunk from the page
                if current_chunk:
                    raw_chunks.append(' '.join(current_chunk))
            
            metrics.metadata.update({"chunk_count": len(raw_chunks)})
            return raw_chunks

    async def chunk_nfpa70_content(self, pages_text: Dict[int, str]) -> List[CodeChunk]:
        """
        Process NFPA 70 content into context-aware chunks with comprehensive analysis.
        
        Args:
            pages_text: Dictionary mapping page numbers to text content
            
        Returns:
            List of CodeChunk objects with full metadata
        """
        with self.monitor.measure("chunk_nfpa70_content", page_count=len(pages_text)) as metrics:
            chunks = []
            
            if not pages_text:
                self.logger.warning("No pages to process")
                return chunks

            # Extract raw chunks from text
            raw_chunks = self._extract_raw_chunks(pages_text)
            self.logger.info(f"Extracted {len(raw_chunks)} raw chunks from {len(pages_text)} pages")

            # Process chunks with GPT
            self.logger.info(f"Processing {len(raw_chunks)} chunks with GPT")
            chunk_analyses, process_metadata = await self.process_chunks_async(raw_chunks)
            
            # Track pages for default page number assignment
            min_page = min(pages_text.keys()) if pages_text else 0
            
            # Convert results to CodeChunk objects
            for i, analysis in enumerate(chunk_analyses):
                if not analysis:  # Skip empty results
                    self.logger.debug(f"Skipping empty analysis for chunk {i}")
                    continue
                
                # Default page number as starting page + rough estimate based on chunk index
                default_page = min_page + (i // 10)
                
                try:
                    chunk = CodeChunk(
                        content=analysis.get('content', raw_chunks[i]),
                        page_number=analysis.get('page_number', default_page),
                        article_number=analysis.get('article_number'),
                        article_title=analysis.get('article_title'),
                        section_number=analysis.get('section_number'),
                        section_title=analysis.get('section_title'),
                        context_tags=analysis.get('context_tags', []),
                        related_sections=analysis.get('related_sections', [])
                    )
                    chunks.append(chunk)
                except Exception as e:
                    self.logger.error(f"Error creating CodeChunk for analysis {i}: {str(e)}")
                    
            self.logger.success(f"Successfully processed {len(chunks)} chunks")
            metrics.metadata.update({
                "raw_chunks": len(raw_chunks),
                "processed_chunks": len(chunks),
                "token_usage": process_metadata["token_usage"]
            })
            
            return chunks


async def chunk_nfpa70_content(text: str, openai_api_key: Optional[str] = None) -> List[Dict]:
    """
    Wrapper function for backward compatibility.
    
    Args:
        text: Text content to process
        openai_api_key: OpenAI API key
        
    Returns:
        List of dictionaries with chunk data
    """
    async with NEC70TextChunker(openai_api_key=openai_api_key) as chunker:
        pages_text = {1: text}
        chunks = await chunker.chunk_nfpa70_content(pages_text)
    
        return [chunk.to_dict() for chunk in chunks]
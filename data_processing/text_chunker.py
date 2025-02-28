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
from rich.progress import Progress, TextColumn, BarColumn, TimeElapsedColumn, TimeRemainingColumn

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


class ChunkCache:
    """
    Simple cache for storing and retrieving processed chunks to avoid redundant API calls.
    """
    def __init__(self, max_size: int = 1000):
        """Initialize the chunk cache with a maximum size."""
        self.cache: Dict[str, Dict] = {}
        self.max_size = max_size
        self.hits = 0
        self.misses = 0
        self.logger = logger.bind(context="chunk_cache")
        self.logger.info(f"Initialized chunk cache with max_size={max_size}")
        
    def get(self, chunk_text: str) -> Optional[Dict]:
        """
        Get a processed chunk from the cache.
        
        Args:
            chunk_text: The raw text of the chunk to look up
            
        Returns:
            The cached processed chunk or None if not found
        """
        # Use a hash of the chunk text as the key
        key = self._get_cache_key(chunk_text)
        result = self.cache.get(key)
        
        if result:
            self.hits += 1
            self.logger.debug(f"Cache hit ({self.hits}/{self.hits+self.misses})")
        else:
            self.misses += 1
            self.logger.debug(f"Cache miss ({self.misses}/{self.hits+self.misses})")
            
        return result
        
    def put(self, chunk_text: str, processed_chunk: Dict) -> None:
        """
        Store a processed chunk in the cache.
        
        Args:
            chunk_text: The raw text of the chunk
            processed_chunk: The processed chunk data
        """
        # Use a hash of the chunk text as the key
        key = self._get_cache_key(chunk_text)
        
        # If cache is full, remove oldest entry
        if len(self.cache) >= self.max_size:
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
            self.logger.debug(f"Cache full, removed oldest entry")
            
        self.cache[key] = processed_chunk
        self.logger.debug(f"Added entry to cache (size: {len(self.cache)})")
        
    def clear(self) -> None:
        """Clear the cache."""
        self.cache.clear()
        self.logger.info("Cache cleared")
        
    def _get_cache_key(self, chunk_text: str) -> str:
        """Generate a cache key from chunk text."""
        # Use a simple hash of the first 100 chars + length as the key
        # This is faster than hashing the entire text but still reasonably unique
        return f"{hash(chunk_text[:100])}-{len(chunk_text)}"
        
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total = self.hits + self.misses
        hit_rate = (self.hits / total) * 100 if total > 0 else 0
        
        return {
            "size": len(self.cache),
            "max_size": self.max_size,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": hit_rate
        }


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
        batch_size: int = 6,  # Reduced from 8 to 6 to minimize timeouts
        max_concurrent_batches: int = 3,  # Increased from 2 to 3 for more parallelism
        timeout: float = 90.0,  # Reduced from 120.0 for faster timeout detection
        max_retries: int = 3,  # Reduced from 5 to 3 for faster failure recovery
        enable_cache: bool = True,  # Enable caching by default
        enable_ocr_cleanup: bool = True  # Enable OCR cleanup by default
    ):
        """Initialize the chunker with configuration parameters."""
        self.logger = logger.bind(context="chunker")
        self.openai_api_key = openai_api_key
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries
        self.enable_ocr_cleanup = enable_ocr_cleanup
        
        # Batch processing configuration
        self.batch_size = batch_size
        self.max_concurrent_batches = max_concurrent_batches
        self.api_semaphore = asyncio.Semaphore(self.max_concurrent_batches)
        
        # Initialize tracking components
        self.token_usage = TokenUsageMetrics()
        self.monitor = PerformanceMonitor("NEC70TextChunker")
        
        # Initialize caching system
        self.enable_cache = enable_cache
        self.cache = ChunkCache() if enable_cache else None
        
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
            connect=10.0,     # Reduced from 30.0
            read=30.0,        # Reduced from self.timeout
            write=30.0,       # Reduced from 60.0
            pool=15.0         # Reduced from 30.0
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
        Generate an enhanced system prompt for GPT optimized for processing NFPA 70 
        electrical code text with OCR errors.
        """
        categories = ", ".join(self.CONTEXT_MAPPING.keys())
        
        return f"""You are a specialized electrical code analyzer for NFPA 70 (National Electrical Code).
Your task is to process raw text from the NEC that contains OCR errors and extract structured information
while preserving ALL technical content.

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

CRITICAL REQUIREMENTS:
1. DO NOT summarize or compress the content. Preserve ALL original text.
2. ENSURE every chunk has complete metadata (article_number, section_number, context_tags).
3. If you're uncertain about metadata, make your best guess rather than returning null values.
4. Preserve ALL measurements, values, and technical requirements exactly as they appear.

For each chunk of text, analyze it carefully to identify:
1. Article number and title (e.g., "ARTICLE 110 - Requirements for Electrical Installations")
2. Section number and title (e.g., "110.12 Mechanical Execution of Work")
3. Technical context - identify which categories apply from: {categories}
4. Referenced sections - any other NEC sections mentioned (e.g., "as required in 230.70")

Specific document structure elements to recognize:
1. EXCEPTIONS - These are important elements that begin with "Exception:" or "Exception No. X:"
2. INFORMATIONAL NOTES - These provide additional context and begin with "Informational Note:"
3. LISTS - Numbered or lettered lists of requirements

Clean the text by correcting these common OCR errors:
- Replace "l00" with "100" (lowercase L to numeral 1)
- Fix "de\\1f'.E" and similar artifacts to "DEVICE"
- Correct "Pt:LL" to "PULL"
- Fix "A.'ID" to "AND"
- Correct spacing issues between words
- Join hyphenated words split across lines

Properly format:
- Keep paragraph breaks intact
- Preserve section numbering exactly (e.g., "314.23(B)(1)")
- Maintain indentation patterns
- Keep measurements in their original format (e.g., "900 mm (3 ft)")

Special instructions for handling article/section continuations:
1. When processing a chunk that doesn't explicitly mention an article or section number,
   look for contextual clues to determine which article or section it belongs to:
   - Check for numbering patterns in the text (e.g., subsections like "(1)", "(2)", "(3)")
   - Look for references to earlier parts of the same section
   - Check for continuation language like "continued" or text that begins mid-sentence

2. When you identify a continuation chunk, use the most recent article_number and 
   section_number values, and add a note in the content field like:
   "[Continuation of Article 110, Section 110.12]" at the beginning.

3. For very long articles or sections that span multiple chunks, ensure metadata 
   consistency across all related chunks rather than leaving fields blank.

For each text chunk, return a JSON object with:
{{
  "content": string,       # The cleaned, corrected text with ALL technical content preserved
  "article_number": string,  # Just the number (e.g., "110")
  "article_title": string,   # Just the title (e.g., "Requirements for Electrical Installations")
  "section_number": string,  # Full section number if present (e.g., "110.12")
  "section_title": string,   # Just the section title if present
  "context_tags": string[],  # Relevant technical categories
  "related_sections": string[]  # Referenced code sections (e.g., ["230.70", "408.36"])
}}

IMPORTANT: 
- Never drop any content, even if it seems redundant.
- Never convert measurements to different units.
- Preserve all technical specifications exactly as written.
- If you can't determine a section number with certainty, use the most recent section number from context.
"""

    def _get_metadata_extraction_prompt(self) -> str:
        """
        Generate a metadata-focused prompt for GPT optimized for extracting structured metadata
        from NFPA 70 electrical code text.
        """
        categories = ", ".join(self.CONTEXT_MAPPING.keys())
        
        return f"""You are a specialized electrical code metadata extractor for NFPA 70 (National Electrical Code).
Your task is to extract structured metadata from NEC text while preserving the original content.

# Important: Return your output as a JSON object with a 'chunks' array
# containing all processed chunks, like this:
{{
  "chunks": [
    {{
      "content": "original text",
      "article_number": "110",
      "article_title": "title",
      ...other fields...
    }},
    {{
      ...next chunk...
    }}
  ]
}}

CRITICAL REQUIREMENTS:
1. PRESERVE the original text exactly in the "content" field.
2. FOCUS on accurate metadata extraction (article_number, section_number, context_tags).
3. If you're uncertain about metadata, make your best guess rather than returning null values.

For each chunk of text, extract the following metadata:
1. Article number and title (e.g., "ARTICLE 110 - Requirements for Electrical Installations")
2. Section number and title (e.g., "110.12 Mechanical Execution of Work")
3. Technical context - identify which categories apply from: {categories}
4. Referenced sections - any other NEC sections mentioned (e.g., "as required in 230.70")

Special instructions for handling article/section continuations:
1. When processing a chunk that doesn't explicitly mention an article or section number,
   look for contextual clues to determine which article or section it belongs to.
2. When you identify a continuation chunk, use the most recent article_number and 
   section_number values.

For each text chunk, return a JSON object with:
{{
  "content": string,       # The ORIGINAL text preserved exactly as provided
  "article_number": string,  # Just the number (e.g., "110")
  "article_title": string,   # Just the title (e.g., "Requirements for Electrical Installations")
  "section_number": string,  # Full section number if present (e.g., "110.12")
  "section_title": string,   # Just the section title if present
  "context_tags": string[],  # Relevant technical categories
  "related_sections": string[]  # Referenced code sections (e.g., ["230.70", "408.36"])
}}

IMPORTANT: 
- Do not modify the original text in the "content" field.
- Focus on accurate metadata extraction rather than text cleaning.
- If you can't determine a section number with certainty, use the most recent section number from context.
"""

    async def _process_chunk_batch_with_timeout(self, chunks: Sequence[str], batch_index: int, timeout=25.0, use_metadata_prompt=False) -> Tuple[List[Dict], Dict[str, Any]]:
        """Process batch with a specific timeout, splitting if needed."""
        try:
            return await asyncio.wait_for(
                self._process_chunk_batch(chunks, batch_index, use_metadata_prompt=use_metadata_prompt),
                timeout=timeout
            )
        except asyncio.TimeoutError:
            self.logger.warning(f"Batch {batch_index} timed out - splitting into smaller batches")
            if len(chunks) <= 2:
                # If batch is already small, retry once with extended timeout
                return await asyncio.wait_for(
                    self._process_chunk_batch(chunks, batch_index, use_metadata_prompt=use_metadata_prompt),
                    timeout=timeout * 1.5
                )
            # Split the batch and process separately
            mid = len(chunks) // 2
            results1, meta1 = await self._process_chunk_batch(chunks[:mid], f"{batch_index}_1", use_metadata_prompt=use_metadata_prompt)
            results2, meta2 = await self._process_chunk_batch(chunks[mid:], f"{batch_index}_2", use_metadata_prompt=use_metadata_prompt)
            # Combine results
            combined_results = results1 + results2
            combined_meta = {
                "batch_index": batch_index,
                "chunk_count": len(chunks),
                "success": meta1["success"] and meta2["success"],
                "processing_time": meta1.get("processing_time", 0) + meta2.get("processing_time", 0),
                "tokens": {
                    "prompt": meta1.get("tokens", {}).get("prompt", 0) + meta2.get("tokens", {}).get("prompt", 0),
                    "completion": meta1.get("tokens", {}).get("completion", 0) + meta2.get("tokens", {}).get("completion", 0),
                    "total": meta1.get("tokens", {}).get("total", 0) + meta2.get("tokens", {}).get("total", 0)
                }
            }
            return combined_results, combined_meta

    async def _process_chunk_batch(
        self, 
        chunks: Sequence[str], 
        batch_index: int,
        use_metadata_prompt: bool = False
    ) -> Tuple[List[Dict], Dict[str, Any]]:
        """
        Process multiple chunks in a single GPT call with robust error handling
        and detailed metrics tracking.
        
        Args:
            chunks: List of text chunks to process
            batch_index: Index of this batch for tracking
            use_metadata_prompt: Whether to use the metadata-focused prompt
            
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
        
        # Log first 100 chars of each chunk to track content
        chunk_previews = [f"{i}: {chunk[:100]}..." for i, chunk in enumerate(chunks)]
        self.logger.debug(f"Batch {batch_index} input previews:\n" + "\n".join(chunk_previews))
        
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
                        
                        # Choose the appropriate prompt based on the use_metadata_prompt flag
                        prompt = self._get_metadata_extraction_prompt() if use_metadata_prompt else self._get_enhanced_system_prompt()
                        
                        # Create chat completion with the selected prompt
                        response = await self.client.chat.completions.create(
                            model=self.model,
                            messages=[
                                {
                                    "role": "system", 
                                    "content": prompt
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
                                # Pad with empty dictionaries if needed
                                while len(chunks_result) < len(chunks):
                                    chunks_result.append({})
                                # Truncate if too many
                                chunks_result = chunks_result[:len(chunks)]
                            
                            # Validate results and find issues
                            issues = self._validate_chunk_results(chunks_result, chunks, batch_index)
                            
                            # If we have issues, retry problematic chunks
                            if issues:
                                self.logger.info(f"Batch {batch_index}: Found {len(issues)} problematic chunks to retry")
                                retry_results = await self._retry_problematic_chunks(issues, chunks, batch_index, use_metadata_prompt)
                                
                                # Replace problematic chunks with retry results
                                for i, idx in enumerate(issues):
                                    if i < len(retry_results) and retry_results[i]:
                                        chunks_result[idx] = retry_results[i]
                            
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

    def _validate_chunk_results(self, chunks_result: List[Dict], original_chunks: Sequence[str], batch_index: int) -> List[int]:
        """Validate chunk results and mark issues for potential reprocessing."""
        issues = []
        
        # Check for missing chunks
        if len(chunks_result) != len(original_chunks):
            self.logger.warning(f"Batch {batch_index}: Expected {len(original_chunks)} chunks, got {len(chunks_result)}")
            # Pad with empty dictionaries if needed
            while len(chunks_result) < len(original_chunks):
                chunks_result.append({})
        
        # Check for empty or invalid chunks
        for i, result in enumerate(chunks_result):
            if not result or not result.get("content"):
                self.logger.warning(f"Batch {batch_index}, Chunk {i}: Empty or missing content")
                issues.append(i)
            elif not result.get("context_tags"):
                self.logger.warning(f"Batch {batch_index}, Chunk {i}: Missing context tags")
                issues.append(i)
            elif not result.get("article_number"):
                self.logger.warning(f"Batch {batch_index}, Chunk {i}: Missing article number")
                issues.append(i)
        
        return issues

    async def _retry_problematic_chunks(
        self, 
        issues: List[int], 
        original_chunks: Sequence[str], 
        batch_index: int,
        use_metadata_prompt: bool = False
    ) -> List[Dict]:
        """Retry processing for problematic chunks individually."""
        retry_results = []
        
        for chunk_idx in issues:
            if chunk_idx >= len(original_chunks):
                continue
                
            self.logger.info(f"Retrying problematic chunk {batch_index}_{chunk_idx}")
            # Process single chunk with more detailed instructions
            retry_text = f"CRITICAL REANALYSIS NEEDED for this text chunk. Preserve ALL content and ensure metadata is complete: {original_chunks[chunk_idx]}"
            
            # Use slightly different prompt for retry, based on the prompt type
            if use_metadata_prompt:
                retry_system_prompt = self._get_metadata_extraction_prompt() + "\nThis is a RETRY of a problematic chunk. Complete metadata is REQUIRED."
            else:
                retry_system_prompt = self._get_enhanced_system_prompt() + "\nThis is a RETRY of a problematic chunk. Complete metadata is REQUIRED."
            
            # Process with extended timeout
            try:
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": retry_system_prompt},
                        {"role": "user", "content": retry_text}
                    ],
                    temperature=0,
                    response_format={"type": "json_object"},
                    timeout=45.0  # Extended timeout for retry
                )
                
                # Parse result
                json_content = response.choices[0].message.content
                result = json.loads(json_content)
                
                # Normalize to expected format
                if "chunks" in result and isinstance(result["chunks"], list) and len(result["chunks"]) > 0:
                    retry_results.append(result["chunks"][0])
                elif "content" in result:
                    retry_results.append(result)
                else:
                    self.logger.error(f"Retry for chunk {batch_index}_{chunk_idx} failed: Invalid format")
                    retry_results.append({})
                    
            except Exception as e:
                self.logger.error(f"Retry for chunk {batch_index}_{chunk_idx} failed: {str(e)}")
                retry_results.append({})
        
        return retry_results

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
            
            # Add rich progress bar
            with Progress(
                TextColumn("[bold blue]{task.description}"),
                BarColumn(),
                "[progress.percentage]{task.percentage:>3.0f}%",
                TimeElapsedColumn(),
                TimeRemainingColumn(),
            ) as progress:
                overall_task = progress.add_task(f"Processing {len(chunks)} chunks", total=len(chunks))
                
                # Process batches in sets with controlled concurrency
                for batch_set_idx in range(0, len(sub_batches), self.max_concurrent_batches):
                    batch_set = sub_batches[batch_set_idx:batch_set_idx + self.max_concurrent_batches]
                    batch_set_start = time.time()
                    
                    # Create tasks for this set of batches
                    tasks = []
                    for i, batch_slice in enumerate(batch_set):
                        batch_idx = batch_set_idx + i
                        tasks.append(asyncio.create_task(
                            self._process_chunk_batch_with_timeout(batch_slice, batch_idx)
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
                    
                    # Update progress bar
                    progress.update(overall_task, advance=len(batch_set) * self.batch_size)
                    
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
        Extract raw text chunks from the input document by identifying article and section breaks
        with improved context tracking for article and section continuations.
        
        Args:
            pages_text: Dictionary mapping page numbers to text content
            
        Returns:
            List of raw text chunks with context preservation
        """
        with self.monitor.measure("extract_raw_chunks", page_count=len(pages_text)) as metrics:
            raw_chunks = []
            current_article = None
            current_article_title = None
            current_section = None
            current_section_title = None
            max_chunk_size = 4000  # Adjust this value based on your token limits
            
            # Process pages in numerical order
            min_page = min(pages_text.keys()) if pages_text else 0
            start_page = max(26, min_page)  # Start from page 26 or minimum available
            
            for page_num in sorted(k for k in pages_text if k >= start_page):
                text = pages_text[page_num]
                lines = text.split('\n')
                current_chunk = []
                
                for line in lines:
                    line_stripped = line.strip()
                    if not line_stripped:
                        continue

                    # Detect article headers
                    article_match = re.search(r'ARTICLE\s+(\d+)(?:\s*-\s*(.+))?', line_stripped, re.IGNORECASE)
                    if article_match:
                        # Save previous chunk if exists
                        if current_chunk:
                            chunk_text = ' '.join(current_chunk)
                            raw_chunks.append(chunk_text)
                        
                        # Update article tracking
                        current_article = article_match.group(1)
                        if article_match.group(2):  # If title is captured
                            current_article_title = article_match.group(2).strip()
                        
                        current_chunk = [line]
                        self.logger.debug(f"Detected Article {current_article}: {current_article_title}")
                        continue
                    
                    # Detect section headers
                    section_match = re.search(r'^(\d+\.\d+)\s+([A-Z].*?)\.', line_stripped)
                    if section_match:
                        # Save previous chunk if exists
                        if current_chunk:
                            chunk_text = ' '.join(current_chunk)
                            raw_chunks.append(chunk_text)
                        
                        # Update section tracking
                        current_section = section_match.group(1)
                        current_section_title = section_match.group(2).strip()
                        
                        current_chunk = [line]
                        self.logger.debug(f"Detected Section {current_section}: {current_section_title}")
                        continue
                    
                    # Handle chunk size limit to avoid excessive token usage
                    if current_chunk and len(' '.join(current_chunk)) > max_chunk_size:
                        chunk_text = ' '.join(current_chunk)
                        raw_chunks.append(chunk_text)
                        
                        # Start new chunk with context prefix for continuations
                        context_prefix = ""
                        if current_article:
                            if current_section:
                                context_prefix = f"[Continuation of Article {current_article}, Section {current_section}] "
                            else:
                                context_prefix = f"[Continuation of Article {current_article}] "
                        
                        current_chunk = [context_prefix + line]
                        self.logger.debug(f"Created continuation chunk with prefix: {context_prefix}")
                    else:
                        current_chunk.append(line)
                
                # Add the last chunk from the page
                if current_chunk:
                    raw_chunks.append(' '.join(current_chunk))
            
            # Log summary
            self.logger.info(f"Extracted {len(raw_chunks)} raw chunks from {len(pages_text)} pages")
            self.logger.debug(f"Article tracking: last article={current_article}, last section={current_section}")
            
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

# For backward compatibility with existing code
ElectricalCodeChunker = NEC70TextChunker
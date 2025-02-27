import asyncio
import time
from typing import Dict, Optional, Any, List, Tuple
from loguru import logger
import httpx
from openai import AsyncOpenAI, RateLimitError, APITimeoutError, APIError
from tenacity import (
    AsyncRetrying, stop_after_attempt, retry_if_exception_type,
    wait_exponential
)
from difflib import SequenceMatcher

class OCRCleaner:
    """
    Cleans OCR text from NFPA 70 documents using GPT.
    
    This class focuses solely on text normalization and OCR error correction,
    without attempting to extract metadata or chunk the content.
    """
    
    def __init__(
        self,
        openai_api_key: Optional[str] = None,
        model: str = "gpt-4o-mini",
        timeout: float = 90.0,
        max_retries: int = 3,
        batch_size: int = 6  # Default parallel batch size
    ):
        """Initialize the OCR cleaner."""
        self.logger = logger.bind(context="ocr_cleaner")
        self.openai_api_key = openai_api_key
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries
        self.batch_size = batch_size
        
        # Initialize monitoring components
        from utils.monitoring import PerformanceMonitor
        self.monitor = PerformanceMonitor("OCRCleaner")
        
        # Initialize as None, will be set in async context
        self.http_client: Optional[httpx.AsyncClient] = None
        self.client: Optional[AsyncOpenAI] = None
        
        self.logger.info(
            f"Initialized OCR cleaner with model={model}, batch_size={batch_size}"
        )

    async def __aenter__(self):
        """Async context manager entry - initializes httpx client and OpenAI client."""
        # Configure timeouts for different operations
        timeout = httpx.Timeout(
            connect=10.0,
            read=30.0,
            write=30.0,
            pool=15.0
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
            http2=True
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
        # Log performance summary
        self.monitor.log_summary()
        
        if self.http_client:
            await self.http_client.aclose()
            self.http_client = None
        if self.client:
            await self.client.close()
            self.client = None

    def _get_cleaning_system_prompt(self) -> str:
        """Generate system prompt focused solely on cleaning OCR text."""
        return """You are a specialized OCR text corrector for NFPA 70 (National Electrical Code) documents.
Your ONLY task is to clean and normalize the text from OCR processing errors while preserving ALL technical content.

CRITICAL REQUIREMENTS:
1. DO NOT summarize, compress, or modify the content. Preserve ALL original text.
2. DO NOT attempt to extract structure, metadata, or organize the text.
3. PRESERVE ALL measurements, values, and technical requirements exactly as they appear.
4. MAINTAIN paragraph breaks and list structures in their original form.

Fix these common OCR errors:
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

IMPORTANT: 
- Your output should be ONLY the cleaned text, formatted as plaintext.
- Never add commentary or notes about your changes.
- Never drop any content, even if it seems redundant.
- Never convert measurements to different units.
- Preserve all technical specifications exactly as written.
"""

    async def clean_page_text(self, page_text: str) -> str:
        """
        Clean the OCR text from a single page.
        
        Args:
            page_text: Raw OCR text from a page
            
        Returns:
            Cleaned and normalized text
        """
        if not self.client:
            self.logger.error("OpenAI client not initialized")
            return page_text
            
        with self.monitor.measure("clean_page_text", text_length=len(page_text)) as metrics:
            try:
                async for attempt in AsyncRetrying(
                    stop=stop_after_attempt(self.max_retries),
                    wait=wait_exponential(multiplier=2, min=4, max=60),
                    retry=retry_if_exception_type((APITimeoutError, RateLimitError, APIError)),
                    reraise=True
                ):
                    with attempt:
                        self.logger.debug(f"Cleaning page text (length: {len(page_text)})")
                        
                        # Create chat completion with cleaning-focused prompt
                        response = await self.client.chat.completions.create(
                            model=self.model,
                            messages=[
                                {
                                    "role": "system", 
                                    "content": self._get_cleaning_system_prompt()
                                },
                                {
                                    "role": "user", 
                                    "content": f"Clean this NFPA 70 text from OCR errors:\n\n{page_text}"
                                }
                            ],
                            temperature=0,
                        )
                        
                        # Track token usage if available
                        if hasattr(response, 'usage'):
                            metrics.metadata["prompt_tokens"] = response.usage.prompt_tokens
                            metrics.metadata["completion_tokens"] = response.usage.completion_tokens
                            metrics.metadata["total_tokens"] = response.usage.total_tokens
                        
                        # Get cleaned text from response
                        cleaned_text = response.choices[0].message.content
                        
                        # Calculate similarity for metrics
                        similarity = SequenceMatcher(None, page_text, cleaned_text).ratio()
                        metrics.metadata["similarity"] = similarity
                        metrics.metadata["original_length"] = len(page_text)
                        metrics.metadata["cleaned_length"] = len(cleaned_text)
                        
                        self.logger.debug(f"Successfully cleaned text: {len(cleaned_text)} chars, {similarity:.2%} similarity")
                        return cleaned_text
                        
            except Exception as e:
                self.logger.error(f"Error cleaning page text: {str(e)}")
                metrics.success = False
                metrics.error = e
                # Return original text if cleaning fails
                return page_text

    async def clean_document(self, pages_text: Dict[int, str]) -> Dict[int, str]:
        """
        Clean OCR text for multiple pages with parallel processing.
        
        Args:
            pages_text: Dictionary mapping page numbers to raw OCR text
            
        Returns:
            Dictionary mapping page numbers to cleaned text
        """
        self.logger.info(f"Starting OCR cleaning for {len(pages_text)} pages with batch_size={self.batch_size}")
        cleaned_pages = {}
        
        # Setup API semaphore for rate limiting
        api_semaphore = asyncio.Semaphore(self.batch_size)
        
        async def clean_page_with_num(page_num, text):
            """Helper function to clean a page with semaphore limit."""
            async with api_semaphore:
                self.logger.info(f"Cleaning page {page_num}")
                try:
                    cleaned_text = await self.clean_page_text(text)
                    return page_num, cleaned_text
                except Exception as e:
                    self.logger.error(f"Error cleaning page {page_num}: {str(e)}")
                    return page_num, text  # Return original text on error
        
        # Process pages in batches to avoid memory issues with very large documents
        sorted_pages = sorted(pages_text.items())
        
        # Use rich progress bar if available
        try:
            from rich.progress import Progress, TextColumn, BarColumn, TimeElapsedColumn, TimeRemainingColumn
            
            with Progress(
                TextColumn("[bold blue]{task.description}"),
                BarColumn(),
                "[progress.percentage]{task.percentage:>3.0f}%",
                TimeElapsedColumn(),
                TimeRemainingColumn(),
            ) as progress:
                overall_task = progress.add_task(f"Cleaning {len(pages_text)} pages", total=len(pages_text))
                
                for batch_start in range(0, len(sorted_pages), self.batch_size * 2):
                    batch_end = min(batch_start + self.batch_size * 2, len(sorted_pages))
                    current_batch = sorted_pages[batch_start:batch_end]
                    
                    # Create tasks for current batch
                    tasks = [clean_page_with_num(page_num, text) for page_num, text in current_batch]
                    batch_results = await asyncio.gather(*tasks)
                    
                    # Update cleaned pages with results
                    for page_num, cleaned_text in batch_results:
                        cleaned_pages[page_num] = cleaned_text
                    
                    # Update progress
                    progress.update(overall_task, advance=len(current_batch))
        
        except ImportError:
            # Fallback if rich is not available
            self.logger.info("Rich progress bar not available, using simple logging")
            
            for batch_start in range(0, len(sorted_pages), self.batch_size * 2):
                batch_end = min(batch_start + self.batch_size * 2, len(sorted_pages))
                current_batch = sorted_pages[batch_start:batch_end]
                
                self.logger.info(f"Processing batch {batch_start//self.batch_size + 1}/{len(sorted_pages)//self.batch_size + 1}")
                
                # Create tasks for current batch
                tasks = [clean_page_with_num(page_num, text) for page_num, text in current_batch]
                batch_results = await asyncio.gather(*tasks)
                
                # Update cleaned pages with results
                for page_num, cleaned_text in batch_results:
                    cleaned_pages[page_num] = cleaned_text
        
        self.logger.info(f"Completed OCR cleaning for {len(pages_text)} pages")
        return cleaned_pages
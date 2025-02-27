# Two-Phase NFPA70 Processing Implementation Plan

This implementation plan is divided into three distinct phases to make it easier to implement and test each component separately.

## Phase 1: Create OCR Cleaner

In this phase, we'll create a new module for cleaning OCR text without changing any existing code.

### File: `data_processing/ocr_cleaner.py`

```python
import json
import time
import asyncio
from typing import Dict, Optional, Any, List
from loguru import logger
import httpx
from openai import AsyncOpenAI, RateLimitError, APITimeoutError, APIError
from tenacity import (
    AsyncRetrying, stop_after_attempt, retry_if_exception_type,
    wait_exponential
)

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
        batch_size: int = 3  # Default parallel batch size
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
                        from difflib import SequenceMatcher
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
```

### File: `test_pdf_processing.py` (Add New Method)

Add this method to the `PDFProcessingTester` class to test OCR cleaning separately:

```python
async def analyze_ocr_cleaning(self, pages_text: Dict[int, str]) -> Dict[int, str]:
    """
    Analyze the OCR cleaning process (Phase 1).
    """
    logger.info("Starting OCR cleaning analysis")
    try:
        # Save pre-cleaning content
        combined_text = "\n\n".join(pages_text.values())
        self._save_text_to_file(combined_text, "pre_cleaning_content.txt")
        logger.info(f"Pre-cleaning total character count: {len(combined_text)}")
        
        # Import OCR cleaner
        from data_processing.ocr_cleaner import OCRCleaner
        
        # Process cleaning
        cleaner_start = time.time()
        async with OCRCleaner(openai_api_key=self.openai_api_key) as cleaner:
            cleaned_pages = await cleaner.clean_document(pages_text)
                
        cleaner_duration = time.time() - cleaner_start
        logger.info(f"OCR cleaning completed in {cleaner_duration:.2f}s")
        
        # Analyze cleaned pages
        combined_cleaned_text = "\n\n".join(cleaned_pages.values())
        self._save_text_to_file(combined_cleaned_text, "post_cleaning_content.txt")
        logger.info(f"Post-cleaning total character count: {len(combined_cleaned_text)}")
        
        # Compare pre and post cleaning content
        similarity, diffs = self._compare_texts(
            combined_text,
            combined_cleaned_text,
            "pre-cleaning",
            "post-cleaning"
        )
        logger.info(f"Cleaning content similarity ratio: {similarity:.2%}")
        
        if diffs:
            logger.info(f"Found {len(diffs)} differences in cleaning")
            self._save_text_to_file(
                "\n".join(diffs),
                "cleaning_differences.txt"
            )
        
        # Update metrics
        self.metrics.additional_metrics["cleaning_similarity"] = similarity
        self.metrics.additional_metrics["cleaning_duration"] = cleaner_duration
        self.metrics.additional_metrics["pre_cleaning_chars"] = len(combined_text)
        self.metrics.additional_metrics["post_cleaning_chars"] = len(combined_cleaned_text)
        
        return cleaned_pages
            
    except Exception as e:
        logger.error(f"Error in OCR cleaning process: {str(e)}")
        raise
```

### File: `test_pdf_processing.py` (Modify main function)

Update the main function to test just Phase 1:

```python
async def main():
    """Main test execution function."""
    test_name = f"OCR Cleaning Test {datetime.now().strftime('%Y-%m-%d')}"
    enable_cache = True
    
    try:
        # Initialize tester
        tester = PDFProcessingTester(test_name=test_name, enable_cache=enable_cache)
        
        # Define test page range
        start_page = 66  # Can be modified for different test ranges
        end_page = 75    # start_page + 9 for 10 pages
        
        # Run tests
        logger.info(f"Starting test run '{test_name}' for pages {start_page}-{end_page}")
        
        # Step 1: Extract raw PDF content
        pages_text = tester.analyze_raw_pdf_content(start_page, end_page)
        
        # Step 2: Test OCR cleaning process (Phase 1 only)
        cleaned_pages = await tester.analyze_ocr_cleaning(pages_text)
        
        # Step 3: Finalize test with optional notes
        notes = """
        Test run for Phase 1: OCR Cleaning
        - Tests document-level OCR cleaning without chunking
        - Measures content preservation accuracy
        - Identifies differences between raw and cleaned text
        """
        tester.finalize_test(notes=notes)
        
        logger.info("Phase 1 test run completed successfully")
        
    except Exception as e:
        logger.error(f"Test run failed: {str(e)}")
        raise
```

## Phase 2: Update Text Chunker for Metadata Extraction

In this phase, we'll modify the existing chunker to work with pre-cleaned text.

### File: `data_processing/text_chunker.py` (Add New Method)

Add this method to the `NEC70TextChunker` class (or `ElectricalCodeChunker` class if you're using the original name):

```python
async def process_cleaned_text(self, cleaned_pages_text: Dict[int, str]) -> List[CodeChunk]:
    """
    Process pre-cleaned text into chunks with metadata.
    
    This method assumes the text has already been cleaned and normalized,
    and focuses on identifying logical chunks and extracting metadata.
    
    Args:
        cleaned_pages_text: Dictionary mapping page numbers to cleaned text
        
    Returns:
        List of CodeChunk objects with full metadata
    """
    with self.monitor.measure("process_cleaned_text", page_count=len(cleaned_pages_text)) as metrics:
        chunks = []
        
        if not cleaned_pages_text:
            self.logger.warning("No cleaned pages to process")
            return chunks

        # Extract raw chunks from the cleaned text
        raw_chunks = self._extract_raw_chunks(cleaned_pages_text)
        self.logger.info(f"Extracted {len(raw_chunks)} raw chunks from {len(cleaned_pages_text)} cleaned pages")

        # Process chunks with GPT focused on metadata extraction
        self.logger.info(f"Processing {len(raw_chunks)} chunks with GPT")
        chunk_analyses, process_metadata = await self.process_chunks_async(raw_chunks)
        
        # Track pages for default page number assignment
        min_page = min(cleaned_pages_text.keys()) if cleaned_pages_text else 0
        
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
```

### File: `data_processing/text_chunker.py` (Add New Prompt Method)

Add this method to provide a simplified prompt focused on metadata extraction:

```python
def _get_metadata_extraction_prompt(self) -> str:
    """
    Generate a system prompt for GPT optimized for metadata extraction 
    from pre-cleaned NFPA 70 text.
    """
    categories = ", ".join(self.CONTEXT_MAPPING.keys())
    
    return f"""You are a specialized electrical code analyzer for NFPA 70 (National Electrical Code).
Your task is to process cleaned text from the NEC and extract structured information.
The text has already been cleaned of OCR errors, so focus on identifying structure and metadata.

# Important: Return your output as a JSON object with a 'chunks' array
# containing all processed chunks, like this:
{{
  "chunks": [
    {{
      "content": "text of this chunk",
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

Specific document structure elements to recognize:
1. EXCEPTIONS - These are important elements that begin with "Exception:" or "Exception No. X:"
2. INFORMATIONAL NOTES - These provide additional context and begin with "Informational Note:"
3. LISTS - Numbered or lettered lists of requirements

For each text chunk, return a JSON object with:
{{
  "content": string,       # The exact text of the chunk (do not modify)
  "article_number": string,  # Just the number (e.g., "110")
  "article_title": string,   # Just the title (e.g., "Requirements for Electrical Installations")
  "section_number": string,  # Full section number if present (e.g., "110.12")
  "section_title": string,   # Just the section title if present
  "context_tags": string[],  # Relevant technical categories
  "related_sections": string[]  # Referenced code sections (e.g., ["230.70", "408.36"])
}}

IMPORTANT: 
- Do not modify the content field - use the exact text provided
- If you can't determine a section number with certainty, use the most recent section number from context
"""
```

### File: `data_processing/text_chunker.py` (Modify process_chunk_batch)

Modify the `_process_chunk_batch` method to use the new prompt when processing pre-cleaned text:

```python
# Add this parameter to the method signature
async def _process_chunk_batch(
    self, 
    chunks: Sequence[str], 
    batch_index: int,
    use_metadata_prompt: bool = False  # Add this parameter
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
    # Existing code...
    
    # Then in the part where you create the chat completion, modify it like this:
    prompt = self._get_metadata_extraction_prompt() if use_metadata_prompt else self._get_enhanced_system_prompt()
    
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
    
    # Rest of the existing code...
```

### File: `test_pdf_processing.py` (Add Method for Phase 2)

Add this method to test just Phase 2 with pre-cleaned text:

```python
async def analyze_metadata_extraction(self, cleaned_pages: Dict[int, str]) -> List[Dict]:
    """
    Analyze the metadata extraction process (Phase 2).
    """
    logger.info("Starting metadata extraction analysis")
    try:
        # Save cleaned content for reference
        combined_cleaned_text = "\n\n".join(cleaned_pages.values())
        self._save_text_to_file(combined_cleaned_text, "input_cleaned_content.txt")
        
        # Process chunks
        chunker_start = time.time()
        async with ElectricalCodeChunker(openai_api_key=self.openai_api_key) as chunker:
            # Use process_cleaned_text if available
            if hasattr(chunker, 'process_cleaned_text'):
                chunks = await chunker.process_cleaned_text(cleaned_pages)
            else:
                # Fallback to original method
                chunks = await chunker.chunk_nfpa70_content(cleaned_pages)
                
        chunker_duration = time.time() - chunker_start
        logger.info(f"Metadata extraction completed in {chunker_duration:.2f}s")
        
        # Analyze chunks
        combined_chunk_text = "\n\n".join(chunk.content for chunk in chunks)
        self._save_text_to_file(combined_chunk_text, "metadata_extraction_output.txt")
        logger.info(f"Output content character count: {len(combined_chunk_text)}")
        
        # Compare cleaned and chunked content
        similarity, diffs = self._compare_texts(
            combined_cleaned_text,
            combined_chunk_text,
            "cleaned-text",
            "chunked-text"
        )
        logger.info(f"Metadata extraction content preservation ratio: {similarity:.2%}")
        
        if diffs:
            logger.warning(f"Found {len(diffs)} significant differences after metadata extraction")
            self._save_text_to_file(
                "\n".join(diffs),
                "metadata_extraction_differences.txt"
            )
        
        # Update metrics
        self.metrics.total_chunks = len(chunks)
        self.metrics.character_count = len(combined_chunk_text)
        self.metrics.similarity_ratio = similarity
        self.metrics.additional_metrics["metadata_extraction_duration"] = chunker_duration
        
        # Calculate coverage metrics
        chunks_with_context_tags = sum(1 for c in chunks if c.context_tags)
        chunks_with_section_number = sum(1 for c in chunks if c.section_number)
        chunks_with_article_number = sum(1 for c in chunks if c.article_number)
        
        if len(chunks) > 0:
            self.metrics.context_tag_coverage = chunks_with_context_tags / len(chunks)
            self.metrics.section_number_coverage = chunks_with_section_number / len(chunks)
            self.metrics.article_number_coverage = chunks_with_article_number / len(chunks)
        
        return chunks
            
    except Exception as e:
        logger.error(f"Error in metadata extraction process: {str(e)}")
        raise
```

### File: `test_pdf_processing.py` (Update main for Phase 2)

Update the main function to test Phase 2:

```python
async def main():
    """Main test execution function."""
    test_name = f"Metadata Extraction Test {datetime.now().strftime('%Y-%m-%d')}"
    enable_cache = True
    
    try:
        # Initialize tester
        tester = PDFProcessingTester(test_name=test_name, enable_cache=enable_cache)
        
        # Define test page range
        start_page = 66  # Can be modified for different test ranges
        end_page = 75    # start_page + 9 for 10 pages
        
        # Run tests
        logger.info(f"Starting test run '{test_name}' for pages {start_page}-{end_page}")
        
        # Step 1: Extract raw PDF content
        pages_text = tester.analyze_raw_pdf_content(start_page, end_page)
        
        # Step 2: Run OCR cleaning (Phase 1)
        cleaned_pages = await tester.analyze_ocr_cleaning(pages_text)
        
        # Step 3: Test metadata extraction (Phase 2)
        chunks = await tester.analyze_metadata_extraction(cleaned_pages)
        
        # Step 4: Analyze final output
        tester.analyze_final_output(chunks)
        
        # Step 5: Finalize test with optional notes
        notes = """
        Test run for Phase 2: Metadata Extraction
        - Tests extraction of metadata from pre-cleaned text
        - Measures content preservation during extraction
        - Analyzes metadata coverage and quality
        """
        tester.finalize_test(notes=notes)
        
        logger.info("Phase 2 test run completed successfully")
        
    except Exception as e:
        logger.error(f"Test run failed: {str(e)}")
        raise
```

## Phase 3: Integrate Two-Phase Processing

In this phase, we'll update the main processing flow to use the two-phase approach.

### File: `test_pdf_processing.py` (Add Complete Two-Phase Method)

Add a method to test the complete two-phase pipeline:

```python
async def analyze_two_phase_processing(self, pages_text: Dict[int, str]) -> List[Dict]:
    """
    Analyze the complete two-phase processing pipeline.
    """
    logger.info("Starting two-phase processing analysis")
    try:
        # Phase 1: OCR Cleaning
        cleaned_pages = await self.analyze_ocr_cleaning(pages_text)
        
        # Phase 2: Metadata Extraction
        chunks = await self.analyze_metadata_extraction(cleaned_pages)
        
        # Calculate overall similarity (raw to final)
        combined_raw_text = "\n\n".join(pages_text.values())
        combined_chunk_text = "\n\n".join(chunk.content for chunk in chunks)
        
        overall_similarity, overall_diffs = self._compare_texts(
            combined_raw_text,
            combined_chunk_text,
            "raw-text",
            "final-text"
        )
        logger.info(f"Overall content preservation ratio: {overall_similarity:.2%}")
        
        if overall_diffs:
            logger.warning(f"Found {len(overall_diffs)} significant differences in overall process")
            self._save_text_to_file(
                "\n".join(overall_diffs),
                "overall_process_differences.txt"
            )
        
        # Update metrics with overall results
        self.metrics.similarity_ratio = overall_similarity
        
        return chunks
            
    except Exception as e:
        logger.error(f"Error in two-phase processing: {str(e)}")
        raise
```

### File: `test_pdf_processing.py` (Update main for Complete Pipeline)

Update the main function to test the complete pipeline:

```python
async def main():
    """Main test execution function."""
    test_name = f"Two-Phase Processing {datetime.now().strftime('%Y-%m-%d')}"
    enable_cache = True
    
    try:
        # Initialize tester
        tester = PDFProcessingTester(test_name=test_name, enable_cache=enable_cache)
        
        # Define test page range
        start_page = 66  # Can be modified for different test ranges
        end_page = 75    # start_page + 9 for 10 pages
        
        # Run tests
        logger.info(f"Starting test run '{test_name}' for pages {start_page}-{end_page}")
        
        # Step 1: Extract raw PDF content
        pages_text = tester.analyze_raw_pdf_content(start_page, end_page)
        
        # Step 2: Test complete two-phase pipeline
        chunks = await tester.analyze_two_phase_processing(pages_text)
        
        # Step 3: Analyze final output
        tester.analyze_final_output(chunks)
        
        # Step 4: Finalize test with optional notes
        notes = """
        Test run with complete two-phase processing approach:
        - Phase 1: Document-level OCR cleaning
        - Phase 2: Chunking and metadata extraction on cleaned text
        - Measures end-to-end content preservation
        - Analyzes metadata quality
        """
        tester.finalize_test(notes=notes)
        
        logger.info("Complete two-phase test run completed successfully")
        
    except Exception as e:
        logger.error(f"Test run failed: {str(e)}")
        raise
```

### File: `main.py` (Add Two-Phase Processing Function)

Add this function to implement the two-phase approach in the main processing flow:

```python
async def process_section_two_phase(
    extractor: PDFExtractor,
    pdf_path: str | Path,
    start_page: int,
    end_page: int,
    openai_api_key: str,
    blob_manager: BlobStorageManager
) -> None:
    """
    Process a section using the two-phase approach:
    1. Clean OCR text
    2. Extract metadata and chunk
    """
    logger.info(f"Processing PDF pages {start_page} to {end_page} with two-phase approach")
    
    # Extract text from PDF
    pages_text = extractor.extract_text_from_pdf(
        pdf_path=Path(pdf_path),
        start_page=start_page,
        end_page=end_page,
        max_pages=10
    )
    
    # Calculate NFPA document page numbers for blob name
    nfpa_start = start_page - 3  # Convert PDF page to NFPA page
    nfpa_end = end_page - 3      # Convert PDF page to NFPA page
    
    logger.info(f"Extracted text from {len(pages_text)} pages (NFPA 70-{nfpa_start} to 70-{nfpa_end})")
    
    # Phase 1: OCR Cleaning
    logger.info("Starting Phase 1: OCR Cleaning...")
    from data_processing.ocr_cleaner import OCRCleaner
    
    async with OCRCleaner(openai_api_key=openai_api_key) as cleaner:
        cleaned_pages = await cleaner.clean_document(pages_text)
    
    logger.info(f"Completed Phase 1: OCR Cleaning for {len(cleaned_pages)} pages")
    
    # Optionally save cleaned text for debugging/comparison
    try:
        cleaned_data = {str(page_num): text for page_num, text in cleaned_pages.items()}
        debug_blob_name = f"nfpa70_cleaned_{nfpa_start:03d}_{nfpa_end:03d}.json"
        debug_blob_manager = BlobStorageManager(
            container_name="nfpa70-refactor-simp",
            blob_name=debug_blob_name
        )
        debug_blob_manager.save_processed_data({"cleaned_pages": cleaned_data})
        logger.info(f"Saved cleaned text to {debug_blob_name} for debugging")
    except Exception as e:
        logger.warning(f"Could not save cleaned text for debugging: {e}")
    
    # Phase 2: Chunking and Metadata Extraction
    logger.info("Starting Phase 2: Chunking and Metadata Extraction...")
    
    # Note: If you renamed the class, use the correct name here (NEC70TextChunker or ElectricalCodeChunker)
    from data_processing.text_chunker import ElectricalCodeChunker
    
    async with ElectricalCodeChunker(openai_api_key=openai_api_key) as chunker:
        # Use the new method for pre-cleaned text if available
        if hasattr(chunker, 'process_cleaned_text'):
            chunks = await chunker.process_cleaned_text(cleaned_pages)
        else:
            # Fallback to original method for backward compatibility
            chunks = await chunker.chunk_nfpa70_content(cleaned_pages)
            
    logger.info(f"Created {len(chunks)} text chunks with metadata")
    
    # Convert chunk objects into dictionaries
    logger.info("Converting chunk objects into dictionaries...")
    chunk_dicts = []
    for c in chunks:
        chunk_dict = {
            "content": c.content,
            "page_number": c.page_number,
            "article_number": c.article_number,
            "section_number": c.section_number,
            "article_title": c.article_title or "",
            "section_title": c.section_title or "",
            "context_tags": list(c.context_tags) if c.context_tags else [],
            "related_sections": list(c.related_sections) if c.related_sections else []
        }
        chunk_dicts.append(chunk_dict)
    
    # Save this section's chunks to its own blob file
    try:
        data_to_save = {"chunks": chunk_dicts}
        blob_name = f"nfpa70_chunks_{nfpa_start:03d}_{nfpa_end:03d}.json"
        section_blob_manager = BlobStorageManager(
            container_name="nfpa70-refactor-simp",
            blob_name=blob_name
        )
        section_blob_manager.save_processed_data(data_to_save)
        logger.info(f"Saved {len(chunk_dicts)} chunks from pages {nfpa_start}-{nfpa_end} to {blob_name}")

    except Exception as e:
        logger.error(f"Failed to save chunks for pages {nfpa_start}-{nfpa_end} to blob storage: {e}")
        raise
```

### File: `main.py` (Update main function)

Update the main function to use the two-phase approach:

```python
async def main():
    """
    Process entire NFPA 70 document using the two-phase approach.
    """
    try:
        load_dotenv()

        # Required environment variables
        pdf_path = os.getenv('PDF_PATH')
        search_endpoint = os.getenv('AZURE_SEARCH_SERVICE_ENDPOINT')
        search_admin_key = os.getenv('AZURE_SEARCH_ADMIN_KEY')
        index_name = os.getenv('AZURE_SEARCH_INDEX_NAME', 'nfpa70-refactor-simp')
        openai_api_key = os.getenv('OPENAI_API_KEY')
        storage_connection_string = os.getenv('AZURE_STORAGE_CONNECTION_STRING')

        # Validate required variables
        required = [
            pdf_path,
            search_endpoint,
            search_admin_key,
            openai_api_key,
            storage_connection_string
        ]
        if any(v is None or v.strip() == "" for v in required):
            raise ValueError("One or more required environment variables are missing.")

        # Initialize services
        extractor = PDFExtractor()
        blob_manager = BlobStorageManager(container_name="nfpa70-refactor-simp")
        
        # Get all page sections to process
        sections = get_page_sections()
        logger.info(f"Starting processing of {len(sections)} sections using two-phase approach")
        
        # Process each section with two-phase approach
        for start_page, end_page in sections:
            try:
                # Use two-phase processing
                await process_section_two_phase(
                    extractor=extractor,
                    pdf_path=pdf_path,
                    start_page=start_page,
                    end_page=end_page,
                    openai_api_key=openai_api_key,
                    blob_manager=blob_manager
                )
            except Exception as e:
                logger.error(f"Error processing section {start_page}-{end_page}: {str(e)}")
                raise

        # Create or update the search index schema
        create_search_index(search_endpoint, search_admin_key, index_name)
        logger.info(f"Search index '{index_name}' created/updated successfully.")

        logger.info("Full document processing completed successfully")
        logger.info("To index the data, run 'index_from_blob.py'.")

    except Exception as e:
        logger.error(f"Error in main process: {str(e)}")
        raise
```

## Implementation Strategy

Follow this step-by-step approach to implement and test the two-phase processing:

### Step 1: Implement Phase 1 (OCR Cleaner)
1. Create `data_processing/ocr_cleaner.py` with the code provided
2. Update `test_pdf_processing.py` to add the `analyze_ocr_cleaning` method
3. Modify the main function to test only Phase 1
4. Run the test to see how well the OCR cleaning works

### Step 2: Implement Phase 2 (Metadata Extraction)
1. Modify `data_processing/text_chunker.py` to add:
   - The `process_cleaned_text` method
   - The `_get_metadata_extraction_prompt` method
   - The updated `_process_chunk_batch` method
2. Update `test_pdf_processing.py` to add the `analyze_metadata_extraction` method
3. Modify the main function to test Phase 1 and Phase 2
4. Run the test to see how well metadata extraction works with cleaned text

### Step 3: Implement Phase 3 (Integration)
1. Update `test_pdf_processing.py` to add the `analyze_two_phase_processing` method
2. Modify the main function to test the complete pipeline
3. Add the `process_section_two_phase` function to `main.py`
4. Update the main function in `main.py` to use the two-phase approach
5. Run full tests to compare the two-phase approach with the original approach

## Expected Improvements

By separating concerns into two phases, you should see:

1. **Better Content Preservation**: Cleaning the text before chunking should significantly improve content preservation
2. **Higher Metadata Quality**: Using pre-cleaned text for metadata extraction should result in more accurate article/section detection
3. **More Reliable Processing**: Each phase focuses on one task, reducing complexity and potential failures
4. **Easier Debugging**: You can examine the output of each phase independently

This approach allows you to measure and optimize each phase separately, giving you more control over the pipeline.
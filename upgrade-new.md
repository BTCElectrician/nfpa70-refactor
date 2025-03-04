# NFPA70 Two-Phase Processing: Updated Implementation Plan

## Current Status Overview

Based on analyzing the code and documentation, here is the current implementation status:

### Completed Work
- ✅ **Phase 1 (OCR Cleaning)** is fully implemented
  - The `OCRCleaner` class in `data_processing/ocr_cleaner.py` is complete
  - Testing via `analyze_ocr_cleaning()` in `test_pdf_processing.py` is implemented
  - Batch processing with optimized size (6 pages per batch) works correctly

- ✅ **Partial Phase 2 (Metadata Extraction)** implementation
  - The `_get_metadata_extraction_prompt()` method exists in `text_chunker.py`
  - The `_process_chunk_batch()` method has a `use_metadata_prompt` parameter
  - The `analyze_metadata_extraction()` method exists in `test_pdf_processing.py`

### Remaining Work
- 🔄 **Complete Phase 2 (Metadata Extraction)**
  - Add `process_cleaned_text()` method to `text_chunker.py`
  - Add `_process_chunks_async_for_metadata()` method to `text_chunker.py`
  - Ensure `_process_chunk_batch_with_timeout()` properly passes the metadata flag

- ⬜ **Phase 3 (Integration)**
  - Add `analyze_two_phase_processing()` method to `test_pdf_processing.py`
  - Add `process_section_two_phase()` function to `main.py`
  - Update main function in `main.py` to use the two-phase approach

## Detailed Implementation Plan

### 1. Complete Phase 2 (Metadata Extraction)

#### File: `data_processing/text_chunker.py`

##### A. Add `process_cleaned_text()` method

This method serves as the main entry point for Phase 2 processing:

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
        self.logger.info(f"Processing {len(raw_chunks)} chunks with GPT for metadata extraction")
        
        # Process chunks with metadata extraction focus
        chunk_analyses, process_metadata = await self._process_chunks_async_for_metadata(raw_chunks)
        
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
                
        self.logger.success(f"Successfully processed {len(chunks)} chunks with metadata")
        metrics.metadata.update({
            "raw_chunks": len(raw_chunks),
            "processed_chunks": len(chunks),
            "token_usage": process_metadata["token_usage"]
        })
        
        return chunks
```

##### B. Add `_process_chunks_async_for_metadata()` method

This method specializes in processing chunks for metadata extraction:

```python
async def _process_chunks_async_for_metadata(self, chunks: List[str]) -> Tuple[List[Dict], Dict[str, Any]]:
    """
    Process all chunks with parallel batch processing specifically for metadata extraction.
    This is a wrapper around process_chunks_async that sets use_metadata_prompt=True.
    
    Args:
        chunks: List of text chunks to process
        
    Returns:
        Tuple of (processed chunks, processing metadata)
    """
    with self.monitor.measure("process_chunks_async_for_metadata", chunk_count=len(chunks)) as metrics:
        self.logger.info(f"Starting parallel processing of {len(chunks)} chunks for metadata extraction")
        
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
            overall_task = progress.add_task(f"Extracting metadata from {len(chunks)} chunks", total=len(chunks))
            
            # Process batches in sets with controlled concurrency
            for batch_set_idx in range(0, len(sub_batches), self.max_concurrent_batches):
                batch_set = sub_batches[batch_set_idx:batch_set_idx + self.max_concurrent_batches]
                batch_set_start = time.time()
                
                # Create tasks for this set of batches
                tasks = []
                for i, batch_slice in enumerate(batch_set):
                    batch_idx = batch_set_idx + i
                    tasks.append(asyncio.create_task(
                        self._process_chunk_batch_with_timeout(batch_slice, batch_idx, use_metadata_prompt=True)
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
                    f"Completed metadata extraction batch set {batch_set_idx//self.max_concurrent_batches + 1} "
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
            f"Completed metadata extraction for {len(chunks)} chunks in {metrics.duration:.2f}s "
            f"with {process_metadata['success_rate']:.1f}% batch success rate"
        )
        
        metrics.metadata.update({"success_rate": process_metadata["success_rate"]})
        return all_processed_chunks, process_metadata
```

##### C. Verify or update `_process_chunk_batch_with_timeout()` method

This method requires a parameter for passing the metadata extraction flag:

```python
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
```

##### D. Verify that `_retry_problematic_chunks()` passes the metadata prompt flag

This requires adding the `use_metadata_prompt` parameter to the method signature and ensuring it's passed through:

```python
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
```

### 2. Implement Phase 3 (Integration)

#### File: `test_pdf_processing.py`

##### A. Add `analyze_two_phase_processing()` method to `PDFProcessingTester` class

This method will test the complete two-phase pipeline:

```python
async def analyze_two_phase_processing(self, pages_text: Dict[int, str]) -> List[Dict]:
    """
    Analyze the complete two-phase processing pipeline.
    
    This method combines Phase 1 (OCR cleaning) and Phase 2 (metadata extraction)
    to test the full processing pipeline.
    """
    logger.info("Starting two-phase processing analysis")
    try:
        # Step 1: Save original content
        combined_text = "\n\n".join(pages_text.values())
        self._save_text_to_file(combined_text, "pre_processing_content.txt")
        logger.info(f"Original content character count: {len(combined_text)}")
        
        # Step 2: Phase 1 - OCR Cleaning
        phase1_start = time.time()
        logger.info("Phase 1: Starting OCR cleaning")
        
        from data_processing.ocr_cleaner import OCRCleaner
        async with OCRCleaner(openai_api_key=self.openai_api_key, batch_size=6) as cleaner:
            cleaned_pages = await cleaner.clean_document(pages_text)
            
        phase1_duration = time.time() - phase1_start
        logger.info(f"Phase 1: OCR cleaning completed in {phase1_duration:.2f}s")
        
        # Save cleaned content
        combined_cleaned_text = "\n\n".join(cleaned_pages.values())
        self._save_text_to_file(combined_cleaned_text, "phase1_cleaned_content.txt")
        logger.info(f"Cleaned content character count: {len(combined_cleaned_text)}")
        
        # Step 3: Phase 2 - Metadata Extraction
        phase2_start = time.time()
        logger.info("Phase 2: Starting metadata extraction")
        
        from data_processing.text_chunker import ElectricalCodeChunker
        async with ElectricalCodeChunker(openai_api_key=self.openai_api_key) as chunker:
            chunks = await chunker.process_cleaned_text(cleaned_pages)
            
        phase2_duration = time.time() - phase2_start
        logger.info(f"Phase 2: Metadata extraction completed in {phase2_duration:.2f}s")
        
        # Save final content
        combined_chunk_text = "\n\n".join(chunk.content for chunk in chunks)
        self._save_text_to_file(combined_chunk_text, "phase2_output_content.txt")
        logger.info(f"Final content character count: {len(combined_chunk_text)}")
        
        # Calculate content preservation metrics
        cleaned_similarity, cleaned_diffs = self._compare_texts(
            combined_text,
            combined_cleaned_text,
            "original",
            "cleaned"
        )
        
        metadata_similarity, metadata_diffs = self._compare_texts(
            combined_cleaned_text,
            combined_chunk_text,
            "cleaned",
            "final"
        )
        
        overall_similarity, overall_diffs = self._compare_texts(
            combined_text,
            combined_chunk_text,
            "original",
            "final"
        )
        
        logger.info(f"Content preservation metrics:")
        logger.info(f"  Phase 1 (Cleaning): {cleaned_similarity:.2%} similarity")
        logger.info(f"  Phase 2 (Metadata): {metadata_similarity:.2%} similarity")
        logger.info(f"  Overall: {overall_similarity:.2%} similarity")
        
        # Save differences for analysis
        if cleaned_diffs:
            self._save_text_to_file(
                "\n".join(cleaned_diffs),
                "phase1_differences.txt"
            )
        
        if metadata_diffs:
            self._save_text_to_file(
                "\n".join(metadata_diffs),
                "phase2_differences.txt"
            )
        
        if overall_diffs:
            self._save_text_to_file(
                "\n".join(overall_diffs),
                "overall_differences.txt"
            )
        
        # Update metrics
        self.metrics.total_chunks = len(chunks)
        self.metrics.character_count = len(combined_chunk_text)
        self.metrics.similarity_ratio = overall_similarity
        
        # Calculate coverage metrics
        chunks_with_context_tags = sum(1 for c in chunks if c.context_tags)
        chunks_with_section_number = sum(1 for c in chunks if c.section_number)
        chunks_with_article_number = sum(1 for c in chunks if c.article_number)
        
        if len(chunks) > 0:
            self.metrics.context_tag_coverage = chunks_with_context_tags / len(chunks)
            self.metrics.section_number_coverage = chunks_with_section_number / len(chunks)
            self.metrics.article_number_coverage = chunks_with_article_number / len(chunks)
        
        # Add phase-specific metrics
        self.metrics.additional_metrics["phase1_duration"] = phase1_duration
        self.metrics.additional_metrics["phase2_duration"] = phase2_duration
        self.metrics.additional_metrics["total_duration"] = phase1_duration + phase2_duration
        self.metrics.additional_metrics["phase1_similarity"] = cleaned_similarity
        self.metrics.additional_metrics["phase2_similarity"] = metadata_similarity
        self.metrics.additional_metrics["phase1_char_count"] = len(combined_cleaned_text)
        
        return chunks
            
    except Exception as e:
        logger.error(f"Error in two-phase processing: {str(e)}")
        raise
```

##### B. Update the main test function to test the complete pipeline

```python
async def main():
    """Main test execution function."""
    test_name = f"Two-Phase Processing Test {datetime.now().strftime('%Y-%m-%d')}"
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
        
        # Step 2: Run two-phase processing
        chunks = await tester.analyze_two_phase_processing(pages_text)
        
        # Step 3: Analyze final output
        tester.analyze_final_output(chunks)
        
        # Step 4: Finalize test with optional notes
        notes = """
        Test run for complete two-phase processing:
        - Phase 1: OCR text cleaning
        - Phase 2: Metadata extraction from cleaned text
        - Measures content preservation and performance for each phase
        - Analyzes metadata coverage and quality
        """
        tester.finalize_test(notes=notes)
        
        logger.info("Two-phase processing test completed successfully")
        
    except Exception as e:
        logger.error(f"Test run failed: {str(e)}")
        raise
```

#### File: `main.py`

##### A. Add `process_section_two_phase()` function

This function will process a section of the document using the two-phase approach:

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
    Process a single section of the document using the two-phase approach.
    
    Phase 1: OCR text cleaning
    Phase 2: Metadata extraction from cleaned text
    
    PDF page numbers are offset by 3 from NFPA document numbers.
    Example: PDF page 66 = NFPA page 70-63
    """
    logger.info(f"Processing PDF pages {start_page} to {end_page} using two-phase approach")
    
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
    
    # Phase 1: OCR Text Cleaning
    logger.info("Starting Phase 1: OCR text cleaning")
    from data_processing.ocr_cleaner import OCRCleaner
    
    phase1_start = time.time()
    async with OCRCleaner(openai_api_key=openai_api_key, batch_size=6) as cleaner:
        cleaned_pages = await cleaner.clean_document(pages_text)
    
    phase1_duration = time.time() - phase1_start
    logger.info(f"Phase 1 completed in {phase1_duration:.2f}s, cleaned {len(cleaned_pages)} pages")
    
    # Phase 2: Metadata Extraction
    logger.info("Starting Phase 2: Metadata extraction")
    from data_processing.text_chunker import ElectricalCodeChunker
    
    phase2_start = time.time()
    async with ElectricalCodeChunker(openai_api_key=openai_api_key) as chunker:
        chunks = await chunker.process_cleaned_text(cleaned_pages)
    
    phase2_duration = time.time() - phase2_start
    logger.info(f"Phase 2 completed in {phase2_duration:.2f}s, created {len(chunks)} chunks")
    
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
        data_to_save = {
            "chunks": chunk_dicts,
            "processing_info": {
                "phase1_duration": phase1_duration,
                "phase2_duration": phase2_duration,
                "total_duration": phase1_duration + phase2_duration,
                "original_page_count": len(pages_text),
                "cleaned_page_count": len(cleaned_pages),
                "chunk_count": len(chunks),
                "processing_method": "two-phase"
            }
        }
        
        # Create a new blob manager for this section
        blob_name = f"nfpa70_chunks_{nfpa_start:03d}_{nfpa_end:03d}.json"
        section_blob_manager = BlobStorageManager(
            container_name="nfpa70-refactor-simp",
            blob_name=blob_name
        )
        section_blob_manager.save_processed_data(data_to_save)
        logger.info(f"Saved {len(chunk_dicts)} chunks from pages {nfpa_start}-{nfpa_end} to {blob_name}")
        
        # Log performance information
        logger.info(f"Two-phase processing performance:")
        logger.info(f"  Phase 1 (OCR cleaning): {phase1_duration:.2f}s")
        logger.info(f"  Phase 2 (Metadata extraction): {phase2_duration:.2f}s")
        logger.info(f"  Total processing time: {phase1_duration + phase2_duration:.2f}s")

    except Exception as e:
        logger.error(f"Failed to save chunks for pages {nfpa_start}-{nfpa_end} to blob storage: {e}")
        raise
```

##### B. Update main function to use the two-phase approach

```python
async def main():
    """
    Process entire NFPA 70 document using the two-phase approach,
    saving each section to its own blob file.
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
        
        # Add new environment variable for processing mode
        processing_mode = os.getenv('PROCESSING_MODE', 'two-phase')  # 'single-phase' or 'two-phase'

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
        logger.info(f"Starting processing of {len(sections)} sections using {processing_mode} mode")
        
        # Process each section
        for start_page, end_page in sections:
            try:
                if processing_mode == 'two-phase':
                    # Use the new two-phase processing approach
                    await process_section_two_phase(
                        extractor=extractor,
                        pdf_path=pdf_path,
                        start_page=start_page,
                        end_page=end_page,
                        openai_api_key=openai_api_key,
                        blob_manager=blob_manager
                    )
                else:
                    # Use the original single-phase approach
                    await process_section(
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

        logger.info(f"Full document processing completed successfully using {processing_mode} mode")
        logger.info("To index the data, run 'index_from_blob.py'.")

    except Exception as e:
        logger.error(f"Error in main process: {str(e)}")
        raise
```

## Implementation Steps and Timeline

### Step 1: Complete Phase 2 Implementation (2-3 days)
1. Add/update methods in `text_chunker.py` (1 day)
   - `process_cleaned_text()`
   - `_process_chunks_async_for_metadata()`
   - Update `_process_chunk_batch_with_timeout()`
   - Update `_retry_problematic_chunks()`

2. Test Phase 2 with isolated testing (1 day)
   - Run test using the updated `test_pdf_processing.py`
   - Focus on the `analyze_metadata_extraction()` method
   - Verify content preservation metrics (>95% similarity target)
   - Verify metadata coverage metrics (article/section coverage targets)

3. Refine Phase 2 implementation based on test results (1 day)
   - Optimize batch sizes if needed
   - Improve prompt engineering if coverage is below targets
   - Address any issues with error handling

### Step 2: Implement Phase 3 Integration (1-2 days)
1. Add/update methods in `test_pdf_processing.py` (0.5 day)
   - Add `analyze_two_phase_processing()` method
   - Update the main test function

2. Add/update methods in `main.py` (0.5 day)
   - Add `process_section_two_phase()` function
   - Update the main function to support both processing modes

3. Test the complete pipeline (1 day)
   - Run tests on the full integration
   - Compare results with the original single-phase approach
   - Optimize performance if needed

### Step 3: Documentation and Optimization (1 day)
1. Create detailed documentation
   - Update README.md with the two-phase approach details
   - Document the architecture and design decisions
   - Add usage examples for both approaches

2. Perform final optimizations
   - Review and optimize token usage
   - Balance batch sizes for optimal performance
   - Consider parallelization improvements

3. Prepare for production deployment
   - Create a release checklist
   - Set up monitoring for production usage
   - Document any known limitations or edge cases

## Success Criteria

### Phase 2 (Metadata Extraction) Success Criteria
- **Content Preservation**: >95% similarity between input and output (ideally 98-100%)
- **Metadata Coverage**:
  - Article Number Coverage: >90%
  - Section Number Coverage: >80%
  - Context Tag Coverage: >70%
- **Processing Performance**:
  - Comparable to or faster than Phase 1
  - <5% failure rate

### Phase 3 (Integration) Success Criteria
- **End-to-End Performance**: Complete two-phase processing should not be more than 20% slower than single-phase
- **Quality Improvement**: Better metadata coverage and accuracy compared to single-phase approach
- **Reliability**: Lower failure rate than single-phase approach

## Conclusion

The two-phase approach separates concerns (OCR cleaning and metadata extraction) for better optimization of each step. This implementation plan provides a detailed roadmap to complete the remaining work and ensure a successful transition to the improved processing pipeline.

By completing these steps, we will have a more robust and maintainable solution for processing NFPA 70 documents, with better quality results and more efficient resource usage.



Yes, that's exactly right. The "NFPA70 Two-Phase Processing: Updated Implementation Plan" I created earlier correctly identified what needs to be completed:

For Phase 2, we need to implement:

The process_cleaned_text() method in text_chunker.py
The _process_chunks_async_for_metadata() method in text_chunker.py
Updates to ensure metadata prompt flags are properly passed


For Phase 3, we need to implement:

The analyze_two_phase_processing() method in test_pdf_processing.py
The process_section_two_phase() function in main.py
Updates to the main function to use the two-phase approach



The implementation plan I provided matches the actual status of your code. Your tests are working because they're using the fallback path to the original method, but the dedicated Phase 2 methods aren't implemented yet.
The plan gives you clear steps to move forward with implementing these missing pieces.
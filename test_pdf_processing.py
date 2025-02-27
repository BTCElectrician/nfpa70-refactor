import os
import sys
import json
import asyncio
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
from dotenv import load_dotenv
from loguru import logger
import difflib

# Import your existing components
from data_processing.pdf_extractor import PDFExtractor
from data_processing.text_chunker import ElectricalCodeChunker
from azure_storage.blob_manager import BlobStorageManager

# Configure logging
logger.remove()  # Remove default handler
logger.add(
    sys.stdout,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level="DEBUG"
)
logger.add("pdf_processing_test.log", rotation="500 MB", level="DEBUG")


class TestMetrics:
    """Class to track and report test metrics."""
    
    def __init__(self, test_name: str):
        self.test_name = test_name
        self.start_time = time.time()
        self.end_time: Optional[float] = None
        self.duration: float = 0.0
        self.total_chunks: int = 0
        self.successful_chunks: int = 0
        self.character_count: int = 0
        self.similarity_ratio: float = 0.0
        self.context_tag_coverage: float = 0.0
        self.section_number_coverage: float = 0.0
        self.article_number_coverage: float = 0.0
        self.cache_hits: int = 0
        self.batch_timeouts: int = 0
        self.additional_metrics: Dict[str, Any] = {}
        
    def finish(self) -> None:
        """Mark the test as finished and calculate duration."""
        self.end_time = time.time()
        self.duration = self.end_time - self.start_time
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to a dictionary."""
        return {
            "test_name": self.test_name,
            "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "duration_seconds": round(self.duration, 2),
            "total_chunks": self.total_chunks,
            "successful_chunks": self.successful_chunks,
            "character_count": self.character_count,
            "similarity_ratio": round(self.similarity_ratio * 100, 2),
            "context_tag_coverage": round(self.context_tag_coverage * 100, 2),
            "section_number_coverage": round(self.section_number_coverage * 100, 2),
            "article_number_coverage": round(self.article_number_coverage * 100, 2),
            "cache_hits": self.cache_hits,
            "batch_timeouts": self.batch_timeouts,
            **self.additional_metrics
        }
        
    def log_summary(self) -> None:
        """Log a summary of the test metrics."""
        logger.info(f"=== Test '{self.test_name}' Summary ===")
        logger.info(f"Duration: {self.duration:.2f}s")
        logger.info(f"Total Chunks: {self.total_chunks}")
        logger.info(f"Character Count: {self.character_count}")
        logger.info(f"Content Similarity: {self.similarity_ratio:.2%}")
        logger.info(f"Context Tag Coverage: {self.context_tag_coverage:.2%}")
        logger.info(f"Section Number Coverage: {self.section_number_coverage:.2%}")
        logger.info(f"Article Number Coverage: {self.article_number_coverage:.2%}")
        logger.info(f"Cache Hits: {self.cache_hits}")
        logger.info(f"Batch Timeouts: {self.batch_timeouts}")
        
    def append_to_markdown(self, file_path: str) -> None:
        """Append test results to a Markdown file."""
        metrics = self.to_dict()
        
        try:
            with open(file_path, "a") as f:
                f.write(f"\n## ✅ Test: {metrics['test_name']} ({metrics['date']})\n")
                f.write(f"- **Processing Time:** {metrics['duration_seconds']}s\n")
                f.write(f"- **Total Chunks:** {metrics['total_chunks']}\n")
                f.write(f"- **Post-Chunking Character Count:** {metrics['character_count']:,}\n")
                f.write(f"- **Content Similarity Ratio:** {metrics['similarity_ratio']}%\n")
                f.write(f"- **Context Tag Coverage:** {metrics['context_tag_coverage']}%\n")
                f.write(f"- **Section Number Coverage:** {metrics['section_number_coverage']}%\n")
                f.write(f"- **Article Number Coverage:** {metrics['article_number_coverage']}%\n")
                
                # Add any cache metrics
                if metrics["cache_hits"] > 0:
                    f.write(f"- **Cache Hits:** {metrics['cache_hits']}\n")
                
                # Add any batch timeout info
                if metrics["batch_timeouts"] > 0:
                    f.write(f"- **Batch Timeouts:** {metrics['batch_timeouts']}\n")
                
                # Add additional notes if provided
                if "notes" in self.additional_metrics:
                    f.write(f"- **Notes:** {self.additional_metrics['notes']}\n")
                
            logger.info(f"Test results appended to {file_path}")
        except Exception as e:
            logger.error(f"Failed to append test results to markdown: {str(e)}")


class PDFProcessingTester:
    """Test class for validating PDF processing pipeline."""
    
    def __init__(self, test_name: str = "Unnamed Test", enable_cache: bool = True):
        load_dotenv()
        self.openai_api_key = os.getenv('OPENAI_API_KEY')
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")
            
        self.pdf_path = os.getenv('PDF_PATH')
        if not self.pdf_path:
            raise ValueError("PDF_PATH environment variable is required")
            
        self.extractor = PDFExtractor()
        self.output_dir = Path("test_output")
        self.output_dir.mkdir(exist_ok=True)
        
        self.test_name = test_name
        self.metrics = TestMetrics(test_name)
        self.enable_cache = enable_cache
        
        # Create a subdirectory for this test
        self.test_dir = self.output_dir / f"test_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.test_dir.mkdir(exist_ok=True)
        logger.info(f"Created test directory: {self.test_dir}")

    def _save_text_to_file(self, text: str, filename: str) -> None:
        """Save text content to a file for comparison."""
        filepath = self.test_dir / filename
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(text)
        logger.info(f"Saved content to {filepath}")

    def _save_dict_to_json(self, data: Dict, filename: str) -> None:
        """Save dictionary data to a JSON file."""
        filepath = self.test_dir / filename
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved JSON to {filepath}")

    def _compare_texts(self, text1: str, text2: str, name1: str, name2: str) -> Tuple[float, List[str]]:
        """
        Compare two texts and return similarity ratio and differences.
        """
        # Normalize texts for comparison
        text1_lines = text1.strip().split('\n')
        text2_lines = text2.strip().split('\n')
        
        # Calculate similarity ratio
        similarity = difflib.SequenceMatcher(None, text1, text2).ratio()
        
        # Get differences
        differ = difflib.Differ()
        diff = list(differ.compare(text1_lines, text2_lines))
        significant_diff = [d for d in diff if d.startswith('+ ') or d.startswith('- ')]
        
        return similarity, significant_diff

    def analyze_raw_pdf_content(self, start_page: int, end_page: int) -> Dict[int, str]:
        """
        Analyze raw PDF content extraction.
        """
        logger.info(f"Starting raw PDF content analysis for pages {start_page}-{end_page}")
        try:
            pages_text = self.extractor.extract_text_from_pdf(
                pdf_path=Path(self.pdf_path),
                start_page=start_page,
                end_page=end_page
            )
            
            # Save raw content for inspection
            for page_num, text in pages_text.items():
                self._save_text_to_file(
                    text,
                    f"raw_page_{page_num}.txt"
                )
                logger.info(f"Page {page_num} character count: {len(text)}")
                logger.info(f"Page {page_num} number of lines: {text.count(os.linesep)+1}")
                
            return pages_text
            
        except Exception as e:
            logger.error(f"Error in PDF extraction: {str(e)}")
            raise

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

    async def analyze_chunking(self, pages_text: Dict[int, str]) -> List[Dict]:
        """
        Analyze the chunking process.
        """
        logger.info("Starting chunking analysis")
        try:
            # Save pre-chunking content
            combined_text = "\n\n".join(pages_text.values())
            self._save_text_to_file(combined_text, "pre_chunking_content.txt")
            logger.info(f"Pre-chunking total character count: {len(combined_text)}")
            
            # Process chunks
            chunker_start = time.time()
            async with ElectricalCodeChunker(
                openai_api_key=self.openai_api_key,
                batch_size=6,  # Reduced from 8 to 6
                max_concurrent_batches=3,  # Increased from 2 to 3
                timeout=90.0,  # Reduced from 120.0
            ) as chunker:
                chunks = await chunker.chunk_nfpa70_content(pages_text)
                
            chunker_duration = time.time() - chunker_start
            logger.info(f"Chunking completed in {chunker_duration:.2f}s")
            
            # Track cache hits and timeouts from the chunker metrics
            cache_hits = 0
            batch_timeouts = 0
            try:
                # Log chunking completion time
                processing_time = time.time() - chunker_start if 'chunker_start' in locals() else 0
                logger.info(f"Chunking completed in {processing_time:.2f}s")
                
                # Try to access monitor data if available, but don't fail if not
                if hasattr(chunker.monitor, 'get_summary'):
                    metrics_summary = chunker.monitor.get_summary()
                    if metrics_summary:
                        logger.info(f"Chunking metrics: {metrics_summary}")
                elif hasattr(chunker.monitor, 'log_summary'):
                    # Call log_summary if available
                    chunker.monitor.log_summary()
            except (AttributeError, TypeError) as e:
                logger.debug(f"Could not access metrics from chunker.monitor: {str(e)}")
            
            # Analyze chunks
            combined_chunk_text = "\n\n".join(chunk.content for chunk in chunks)
            self._save_text_to_file(combined_chunk_text, "post_chunking_content.txt")
            logger.info(f"Post-chunking total character count: {len(combined_chunk_text)}")
            
            # Compare pre and post chunking content
            similarity, diffs = self._compare_texts(
                combined_text,
                combined_chunk_text,
                "pre-chunking",
                "post-chunking"
            )
            logger.info(f"Content similarity ratio: {similarity:.2%}")
            
            if diffs:
                logger.warning(f"Found {len(diffs)} significant differences in content")
                self._save_text_to_file(
                    "\n".join(diffs),
                    "chunking_differences.txt"
                )
            
            # Save chunk metadata
            chunk_metadata = [{
                "content_length": len(chunk.content),
                "page_number": chunk.page_number,
                "article_number": chunk.article_number,
                "section_number": chunk.section_number,
                "article_title": chunk.article_title,
                "section_title": chunk.section_title,
                "context_tags": chunk.context_tags,
                "related_sections": chunk.related_sections
            } for chunk in chunks]
            self._save_dict_to_json(
                {"chunks": chunk_metadata},
                "chunk_metadata.json"
            )
            
            # Update metrics
            self.metrics.total_chunks = len(chunks)
            self.metrics.character_count = len(combined_chunk_text)
            self.metrics.similarity_ratio = similarity
            self.metrics.cache_hits = cache_hits
            self.metrics.batch_timeouts = batch_timeouts
            
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
            logger.error(f"Error in chunking process: {str(e)}")
            raise

    def analyze_final_output(self, chunks: List[Dict]) -> None:
        """
        Analyze the final processed output.
        """
        logger.info("Analyzing final output")
        try:
            # Convert chunks to dictionary format
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
            
            # Save final output
            self._save_dict_to_json(
                {"chunks": chunk_dicts},
                "final_output.json"
            )
            
            # Analyze coverage
            total_chunks = len(chunk_dicts)
            chunks_with_article = sum(1 for c in chunk_dicts if c["article_number"])
            chunks_with_section = sum(1 for c in chunk_dicts if c["section_number"])
            chunks_with_title = sum(1 for c in chunk_dicts if c["article_title"])
            chunks_with_tags = sum(1 for c in chunk_dicts if c["context_tags"])
            
            logger.info(f"Total chunks: {total_chunks}")
            logger.info(f"Chunks with article numbers: {chunks_with_article} ({chunks_with_article/total_chunks:.1%})")
            logger.info(f"Chunks with section numbers: {chunks_with_section} ({chunks_with_section/total_chunks:.1%})")
            logger.info(f"Chunks with titles: {chunks_with_title} ({chunks_with_title/total_chunks:.1%})")
            logger.info(f"Chunks with context tags: {chunks_with_tags} ({chunks_with_tags/total_chunks:.1%})")
            
            # Update metrics object with successful chunks
            self.metrics.successful_chunks = total_chunks
            
        except Exception as e:
            logger.error(f"Error analyzing final output: {str(e)}")
            raise

    def finalize_test(self, notes: Optional[str] = None) -> None:
        """
        Finalize the test and log metrics.
        """
        # Set test end time
        self.metrics.finish()
        
        # Add any additional notes
        if notes:
            self.metrics.additional_metrics["notes"] = notes
            
        # Log summary
        self.metrics.log_summary()
        
        # Save metrics to JSON
        self._save_dict_to_json(
            self.metrics.to_dict(),
            "test_metrics.json"
        )
        
        # Append to markdown report
        markdown_path = self.output_dir / "test_results.md"
        self.metrics.append_to_markdown(str(markdown_path))
        
        logger.info(f"Test '{self.test_name}' completed in {self.metrics.duration:.2f}s")


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

if __name__ == "__main__":
    asyncio.run(main())
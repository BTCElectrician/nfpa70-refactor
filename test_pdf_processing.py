import os
import sys
import json
import asyncio
from pathlib import Path
from typing import Dict, List, Tuple
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

class PDFProcessingTester:
    """Test class for validating PDF processing pipeline."""
    
    def __init__(self):
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

    def _save_text_to_file(self, text: str, filename: str) -> None:
        """Save text content to a file for comparison."""
        filepath = self.output_dir / filename
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(text)
        logger.info(f"Saved content to {filepath}")

    def _save_dict_to_json(self, data: Dict, filename: str) -> None:
        """Save dictionary data to a JSON file."""
        filepath = self.output_dir / filename
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
            async with ElectricalCodeChunker(openai_api_key=self.openai_api_key) as chunker:
                chunks = await chunker.chunk_nfpa70_content(pages_text)
                
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
                "context_tags": chunk.context_tags,
                "related_sections": chunk.related_sections
            } for chunk in chunks]
            self._save_dict_to_json(
                {"chunks": chunk_metadata},
                "chunk_metadata.json"
            )
            
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
            
        except Exception as e:
            logger.error(f"Error analyzing final output: {str(e)}")
            raise

async def main():
    """Main test execution function."""
    try:
        # Initialize tester
        tester = PDFProcessingTester()
        
        # Define test page range
        start_page = 66  # Can be modified for different test ranges
        end_page = 75    # start_page + 9 for 10 pages
        
        # Run tests
        logger.info(f"Starting test run for pages {start_page}-{end_page}")
        
        # Step 1: Extract and analyze raw PDF content
        pages_text = tester.analyze_raw_pdf_content(start_page, end_page)
        
        # Step 2: Analyze chunking process
        chunks = await tester.analyze_chunking(pages_text)
        
        # Step 3: Analyze final output
        tester.analyze_final_output(chunks)
        
        logger.info("Test run completed successfully")
        
    except Exception as e:
        logger.error(f"Test run failed: {str(e)}")
        raise

if __name__ == "__main__":
    asyncio.run(main())
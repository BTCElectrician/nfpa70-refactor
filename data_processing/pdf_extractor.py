import pymupdf
from typing import Dict, Optional
from loguru import logger
from pathlib import Path
import os
from dotenv import load_dotenv

class PDFExtractor:
    """Simplified PDF text extraction for electrical code documents."""
    
    def __init__(self):
        load_dotenv()
        self.logger = logger.bind(context="pdf_extractor")

    def extract_text_from_pdf(
        self,
        pdf_path: Path,
        start_page: int = 1,
        end_page: Optional[int] = None,
        max_pages: Optional[int] = None
    ) -> Dict[int, str]:
        """
        Extract text from PDF using PDF page numbers as keys.
        
        Args:
            pdf_path: Path to PDF file
            start_page: First page to read (1-based index)
            end_page: Last page to read (default=None, read to end)
            max_pages: Maximum number of pages to process
        
        Returns:
            Dict mapping PDF page numbers (e.g., 66) to text
        """
        try:
            self.logger.info(f"Processing PDF: {pdf_path}")
            doc = pymupdf.open(pdf_path)
            pages_text = {}
            
            start_idx = start_page - 1
            total_pages = len(doc)
            end_idx = min(end_page or total_pages, total_pages)
            if max_pages:
                end_idx = min(start_idx + max_pages, end_idx)
            
            for page_num in range(start_idx, end_idx):
                pdf_page_num = page_num + 1  # 1-based
                page = doc[page_num]
                text = page.get_text("text")
                
                if text.strip():
                    pages_text[pdf_page_num] = text
                    self.logger.info(f"Extracted text from PDF page {pdf_page_num}")
                else:
                    self.logger.warning(f"Empty text on PDF page {pdf_page_num}, skipped")
            
            if not pages_text:
                self.logger.warning("No text extracted from PDF")
            else:
                self.logger.info(f"Successfully processed {len(pages_text)} pages")
            
            return pages_text
        
        except Exception as e:
            self.logger.error(f"Error processing PDF: {str(e)}")
            raise ValueError(f"Failed to process PDF: {str(e)}") from e
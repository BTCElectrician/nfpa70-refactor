import pymupdf 
import re
from typing import Dict, Optional
import logging
from pathlib import Path
import regex

class PDFExtractor:
    """Enhanced PDF text extraction for electrical code documents."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def _find_page_number(self, text: str) -> Optional[int]:
        """Extract NFPA page number from text."""
        match = regex.search(r'70-(\d+)', text)
        if match:
            return int(match.group(1))
        return None

    def extract_text_from_pdf(
        self,
        pdf_path: Path,
        start_page: int = 1,
        end_page: Optional[int] = None,
        max_pages: Optional[int] = None
    ) -> Dict[int, str]:
        """
        Extract text from PDF with optional page range and limit controls.
        
        Args:
            pdf_path: Path to PDF file
            start_page: The first page to read (1-based index, default=1)
            end_page: The last page to read (default=None, meaning read to end)
            max_pages: Optional maximum number of pages to process
        
        Returns:
            Dict mapping page numbers to cleaned text
        """
        try:
            self.logger.info(f"Processing PDF: {pdf_path}")
            doc = pymupdf.open(pdf_path)
            pages_text = {}
            
            # Convert start_page to 0-based index
            start_idx = start_page - 1
            
            # Calculate end page
            total_pages = len(doc)
            end_idx = min(end_page or total_pages, total_pages)
            
            # Apply max_pages limit if specified
            if max_pages:
                end_idx = min(start_idx + max_pages, end_idx)
            
            for page_num in range(start_idx, end_idx):
                page = doc[page_num]
                text = page.get_text("text")
                
                # Only store non-empty pages
                if text.strip():
                    doc_page_num = self._find_page_number(text)
                    if doc_page_num:
                        pages_text[doc_page_num] = text
                
            if not pages_text:
                self.logger.warning("No text extracted from PDF")
            else:
                self.logger.info(f"Successfully processed {len(pages_text)} pages")
                
            return pages_text
            
        except Exception as e:
            self.logger.error(f"Error processing PDF: {str(e)}")
            raise ValueError(f"Failed to process PDF: {str(e)}") from e
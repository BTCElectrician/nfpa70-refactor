import pymupdf 
import re
from typing import Dict, Optional
import logging
from pathlib import Path

class PDFExtractor:
    """Enhanced PDF text extraction for electrical code documents."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def extract_text_from_pdf(self, pdf_path: Path, max_pages: int = None) -> Dict[int, str]:
        """
        Extract text from PDF with optional page limit for testing.
        
        Args:
            pdf_path: Path to PDF file
            max_pages: Optional maximum number of pages to process
        
        Returns:
            Dict mapping page numbers to cleaned text
        """
        try:
            self.logger.info(f"Processing PDF: {pdf_path}")
            doc = pymupdf.open(pdf_path)
            pages_text = {}
            
            total_pages = len(doc)
            pages_to_process = min(max_pages, total_pages) if max_pages else total_pages
            
            for page_num in range(pages_to_process):
                page = doc[page_num]
                
                # Extract text using rawdict mode for better metadata
                raw_dict = page.get_text("rawdict")
                text_parts = []
                
                # Process each block in the rawdict
                for block in raw_dict.get("blocks", []):
                    # Skip non-text blocks
                    if "lines" not in block:
                        continue
                        
                    block_parts = []
                    for line in block["lines"]:
                        line_text = []
                        for span in line.get("spans", []):
                            # Only include text spans (skip other content types)
                            if span.get("text"):
                                # Skip text that might be headers/footers
                                if span.get("size", 0) > 20:  # Skip very large text
                                    continue
                                line_text.append(span["text"])
                                
                        if line_text:
                            block_parts.append(" ".join(line_text))
                    
                    if block_parts:
                        text_parts.append("\n".join(block_parts))
                
                text = "\n".join(text_parts)
                
                # Only store non-empty pages
                if text.strip():
                    # Store with page number (1-based)
                    pages_text[page_num + 1] = text
                
            self.logger.info(f"Successfully processed {len(pages_text)} pages")
            return pages_text
            
        except Exception as e:
            self.logger.error(f"Error processing PDF: {str(e)}")
            raise ValueError(f"Failed to process PDF: {str(e)}") from e
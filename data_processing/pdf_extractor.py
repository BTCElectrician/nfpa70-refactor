import pymupdf 
import re
from typing import Dict, Optional
import logging
from pathlib import Path

class PDFExtractor:
    """Enhanced PDF text extraction for electrical code documents."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Common OCR/PDF extraction artifacts to clean
        self.corrections = {
            r'o;': 's',
            r'\"': '"',
            r'\\"': '"',
            r'\\\'': "'",
            r'\s+': ' ',
            r'(?<=\d)\s*\.\s*(?=\d)': '.',  # Fix broken decimal points
            r'(?<=\d)\s*-\s*(?=\d)': '-',    # Fix broken ranges
        }
        
        # Important electrical terms to preserve
        self.electrical_terms = {
            r'(\d+)\s*[Vv]\b': r'\1 volts',
            r'(\d+)\s*[Aa]\b': r'\1 amperes',
            r'(\d+)\s*[Ww]\b': r'\1 watts',
            r'(\d+)\s*VAC': r'\1 VAC',
            r'(\d+)\s*VDC': r'\1 VDC',
            r'(\d+)\s*AWG': r'\1 AWG',
            r'(\d+)\s*hp\b': r'\1 horsepower',
            r'(\d+)\s*kVA\b': r'\1 kVA',
            r'(\d+)\s*Hz\b': r'\1 Hz'
        }
        
        # Common electrical terms that might be broken
        self.term_fixes = {
            'ground ing': 'grounding',
            'bond ing': 'bonding',
            'circuit breaker': 'circuit breaker',
            'race way': 'raceway',
            'load center': 'load center',
            'sub panel': 'subpanel',
            'sub circuit': 'subcircuit'
        }

    def clean_text(self, text: str) -> str:
        """
        Clean extracted text while preserving important electrical terms.
        
        Args:
            text: Raw text from PDF
            
        Returns:
            Cleaned text with preserved electrical terminology
        """
        # Basic cleaning
        text = text.strip()
        
        # Apply OCR corrections
        for pattern, replacement in self.corrections.items():
            text = re.sub(pattern, replacement, text)
        
        # Fix broken electrical terms
        for broken, fixed in self.term_fixes.items():
            text = text.replace(broken, fixed)
        
        # Preserve electrical measurements and units
        for pattern, replacement in self.electrical_terms.items():
            text = re.sub(pattern, replacement, text)
        
        # Remove multiple spaces and normalize newlines
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        return text.strip()

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
                
                # Extract text with better layout preservation
                text = page.get_text("text")
                
                # Clean the extracted text
                cleaned_text = self.clean_text(text)
                
                # Only store non-empty pages
                if cleaned_text.strip():
                    # Store with page number (1-based)
                    pages_text[page_num + 1] = cleaned_text
                
            self.logger.info(f"Successfully processed {len(pages_text)} pages")
            return pages_text
            
        except Exception as e:
            self.logger.error(f"Error processing PDF: {str(e)}")
            raise ValueError(f"Failed to process PDF: {str(e)}") from e 
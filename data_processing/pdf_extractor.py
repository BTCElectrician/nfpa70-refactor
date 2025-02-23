import pymupdf
from typing import Dict, Optional
from loguru import logger
from pathlib import Path
from openai import OpenAI
import os
from dotenv import load_dotenv

class PDFExtractor:
    """Enhanced PDF text extraction for electrical code documents using GPT-4o-mini."""
    
    def __init__(self):
        load_dotenv()
        self.logger = logger.bind(context="pdf_extractor")
        self.openai_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

    def _extract_page_number_with_gpt(self, text: str, pdf_page_num: int, prev_nfpa: Optional[int] = None) -> int:
        """Use GPT-4o-mini to extract NFPA page number, interpolating if needed."""
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{
                    "role": "system",
                    "content": """Extract the NFPA page number from this text in the format '70-XX'. 
                    Return just the XX part (e.g., 63 for '70-63'). If not found, estimate based on context or previous page."""
                }, {
                    "role": "user",
                    "content": f"Text from PDF page {pdf_page_num} (previous NFPA: {prev_nfpa or 'unknown'}):\n{text[:2000]}"
                }],
                temperature=0,
                max_tokens=10
            )
            result = response.choices[0].message.content.strip()
            try:
                nfpa_num = int(result)
                self.logger.debug(f"GPT detected NFPA page 70-{nfpa_num} for PDF page {pdf_page_num}")
                return nfpa_num
            except ValueError:
                self.logger.debug(f"GPT returned non-integer for PDF page {pdf_page_num}: {result}")
        except Exception as e:
            self.logger.error(f"GPT error for PDF page {pdf_page_num}: {str(e)}")
        
        # Fallback: Interpolate from previous NFPA number or use PDF page as base
        if prev_nfpa is not None:
            interpolated = prev_nfpa + 1
            self.logger.warning(f"No NFPA number detected for PDF page {pdf_page_num}, interpolating to 70-{interpolated}")
            return interpolated
        # If no previous page, use PDF page number as a starting point
        self.logger.warning(f"No NFPA number detected for PDF page {pdf_page_num}, using PDF number 70-{pdf_page_num}")
        return pdf_page_num

    def extract_text_from_pdf(
        self,
        pdf_path: Path,
        start_page: int = 1,
        end_page: Optional[int] = None,
        max_pages: Optional[int] = None
    ) -> Dict[int, str]:
        """
        Extract text from PDF with GPT-based page number detection and interpolation.
        
        Args:
            pdf_path: Path to PDF file
            start_page: First page to read (1-based index)
            end_page: Last page to read (default=None, read to end)
            max_pages: Maximum number of pages to process
        
        Returns:
            Dict mapping NFPA page numbers (e.g., 63) to text
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
            
            prev_nfpa = None
            for page_num in range(start_idx, end_idx):
                pdf_page_num = page_num + 1  # 1-based for logging
                page = doc[page_num]
                text = page.get_text("text")
                
                if text.strip():
                    nfpa_page_num = self._extract_page_number_with_gpt(text, pdf_page_num, prev_nfpa)
                    # Handle potential duplicates by incrementing
                    while nfpa_page_num in pages_text:
                        nfpa_page_num += 1
                        self.logger.debug(f"Incremented NFPA page to 70-{nfpa_page_num} to avoid duplicate")
                    pages_text[nfpa_page_num] = text
                    self.logger.info(f"PDF page {pdf_page_num} mapped to NFPA page 70-{nfpa_page_num}")
                    prev_nfpa = nfpa_page_num
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
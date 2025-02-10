import re
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass, field
import logging
import json
from openai import OpenAI
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

@dataclass
class CodePosition:
    """Tracks the current position within the electrical code document."""
    article_number: Optional[str] = None
    article_title: Optional[str] = None
    section_number: Optional[str] = None
    section_title: Optional[str] = None
    subsection_letter: Optional[str] = None
    hierarchy: List[str] = field(default_factory=list)
    context_before: Optional[str] = None
    context_after: Optional[str] = None

    def update_from_text(self, text: str) -> Tuple[bool, str]:
        """Updates position based on text markers."""
        position_changed = False
        change_type = ""

        article_match = re.search(r'ARTICLE\s+(\d+)\s*[-—]\s*(.+?)(?=\n|$)', text, re.IGNORECASE)
        if article_match:
            self.article_number = article_match.group(1)
            self.article_title = article_match.group(2).strip()
            self.hierarchy = [f"Article {self.article_number}"]
            self.section_number = None
            self.section_title = None
            self.subsection_letter = None
            position_changed = True
            change_type = "article"

        section_match = re.search(r'(?:^|\s)(\d+\.\d+)\s+([^.]+?)(?=\n|$)', text)
        if section_match:
            self.section_number = section_match.group(1)
            self.section_title = section_match.group(2).strip()
            if self.hierarchy:
                self.hierarchy = self.hierarchy[:1] + [f"Section {self.section_number}"]
            self.subsection_letter = None
            position_changed = True
            change_type = "section"

        subsection_match = re.search(r'\(([A-Z])\)\s+', text)
        if subsection_match:
            new_subsection = subsection_match.group(1)
            if new_subsection != self.subsection_letter:
                self.subsection_letter = new_subsection
                if len(self.hierarchy) >= 2:
                    self.hierarchy = self.hierarchy[:2] + [f"Subsection ({self.subsection_letter})"]
                position_changed = True
                change_type = "subsection"

        return position_changed, change_type

@dataclass
class CodeChunk:
    """Represents a chunk of electrical code with essential metadata."""
    content: str                     # The actual text content of the chunk
    page_number: int                 # Now represents NFPA page number (70-XX)
    article_number: Optional[str]    # Article number (e.g., "90")
    section_number: Optional[str]    # Section number (e.g., "90.2")
    article_title: Optional[str]     # Title of the article
    section_title: Optional[str]     # Title of the section
    context_tags: List[str]         # Technical context tags
    related_sections: List[str]      # Referenced code sections

class ElectricalCodeChunker:
    """Chunks electrical code text with GPT-based cleanup and analysis."""
    
    def __init__(self, openai_api_key: Optional[str] = None):
        self.logger = logger
        self.client = OpenAI(api_key=openai_api_key) if openai_api_key else None
        
        # Define technical contexts for tagging
        self.context_mapping = {
            'service_equipment': [
                'service equipment', 'service entrance', 'service drop',
                'meter', 'service disconnect', 'main disconnect'
            ],
            'conductors': [
                'conductor', 'wire', 'cable', 'AWG', 'kcmil',
                'copper', 'aluminum'
            ],
            'raceway': [
                'EMT', 'IMC', 'RMC', 'PVC', 'conduit',
                'electrical metallic tubing'
            ],
            'grounding': [
                'ground', 'grounding', 'bonding', 'GEC',
                'equipment grounding conductor'
            ],
            'overcurrent': [
                'breaker', 'circuit breaker', 'fuse', 'overcurrent',
                'AFCI', 'GFCI'
            ]
        }

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def process_chunk_with_gpt(self, text: str) -> Dict:
        """Single GPT call to clean and analyze text."""
        if not self.client:
            return {}
            
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{
                    "role": "system", 
                    "content": """You are processing NFPA 70 (NEC) text. Clean any OCR errors and analyze the content.
                    Return a JSON object with:
                    {
                        "content": string,         # OCR-corrected text
                        "article_number": string,
                        "section_number": string,
                        "article_title": string,
                        "section_title": string,
                        "context_tags": string[],  # Technical categories
                        "related_sections": string[]  # Referenced sections
                    }"""
                },
                {
                    "role": "user", 
                    "content": f"Process this NEC text: {text}"
                }],
                temperature=0,
                response_format={"type": "json_object"}
            )
            
            return json.loads(response.choices[0].message.content)
                
        except Exception as e:
            self.logger.error(f"Error in GPT processing: {str(e)}")
            return {}

    def chunk_nfpa70_content(self, pages_text: Dict[int, str]) -> List[CodeChunk]:
        """Process NFPA 70 content into context-aware chunks."""
        chunks = []
        
        if not pages_text:
            return chunks

        start_page = min(26, min(pages_text.keys()))

        for page_num in sorted(k for k in pages_text if k >= start_page):
            text = pages_text[page_num]
            lines = text.split('\n')
            current_chunk = []
            
            for line in lines:
                if not line.strip():
                    continue

                if re.search(r'ARTICLE\s+\d+|^\d+\.\d+\s+[A-Z]', line):
                    if current_chunk:
                        chunk_text = ' '.join(current_chunk)
                        analysis = self.process_chunk_with_gpt(chunk_text)
                        
                        chunks.append(CodeChunk(
                            content=analysis.get('content', chunk_text),
                            page_number=page_num,
                            article_number=analysis.get('article_number'),
                            article_title=analysis.get('article_title'),
                            section_number=analysis.get('section_number'),
                            section_title=analysis.get('section_title'),
                            context_tags=analysis.get('context_tags', []),
                            related_sections=analysis.get('related_sections', [])
                        ))
                    current_chunk = [line]
                else:
                    current_chunk.append(line)
            
            if current_chunk:
                chunk_text = ' '.join(current_chunk)
                analysis = self.process_chunk_with_gpt(chunk_text)
                
                chunks.append(CodeChunk(
                    content=analysis.get('content', chunk_text),
                    page_number=page_num,
                    article_number=analysis.get('article_number'),
                    article_title=analysis.get('article_title'),
                    section_number=analysis.get('section_number'),
                    section_title=analysis.get('section_title'),
                    context_tags=analysis.get('context_tags', []),
                    related_sections=analysis.get('related_sections', [])
                ))

        return chunks

def chunk_nfpa70_content(text: str, openai_api_key: Optional[str] = None) -> List[Dict]:
    """Wrapper for older code if needed."""
    chunker = ElectricalCodeChunker(openai_api_key=openai_api_key)
    pages_text = {1: text}
    chunks = chunker.chunk_nfpa70_content(pages_text)
    
    return [
        {
            "content": chunk.content,
            "page_number": chunk.page_number,
            "article_number": chunk.article_number,
            "section_number": chunk.section_number,
            "article_title": chunk.article_title or "",
            "section_title": chunk.section_title or "",
            "context_tags": chunk.context_tags,
            "related_sections": chunk.related_sections,
            "gpt_analysis": {},
            "cleaned_text": chunk.content,
            "ocr_confidence": 1.0
        }
        for chunk in chunks
    ]
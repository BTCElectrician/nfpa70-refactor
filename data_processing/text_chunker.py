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
    """Represents a chunk of electrical code with enhanced context."""
    content: str
    page_number: int
    article_number: Optional[str] = None
    article_title: Optional[str] = None
    section_number: Optional[str] = None
    section_title: Optional[str] = None
    context_tags: List[str] = field(default_factory=list)
    related_sections: List[str] = field(default_factory=list)
    gpt_analysis: Dict = field(default_factory=dict)
    hierarchy: List[str] = field(default_factory=list)
    context_before: Optional[str] = None
    context_after: Optional[str] = None
    cleaned_text: Optional[str] = None
    ocr_confidence: float = 1.0

class ElectricalCodeChunker:
    """Enhanced chunking for electrical code text with GPT-based OCR cleanup."""
    
    def __init__(self, openai_api_key: Optional[str] = None):
        """Initialize the chunker with optional OpenAI integration."""
        self.logger = logger
        self.client = OpenAI(api_key=openai_api_key) if openai_api_key else None
        self.position = CodePosition()
        
        self.context_mapping = {
            'service_equipment': [
                'service equipment', 'service entrance', 'service drop', 'service lateral',
                'meter', 'metering', 'service disconnect', 'main disconnect', 'service panel',
                'switchboard', 'switchgear', 'panelboard'
            ],
            'conductors': [
                'conductor', 'wire', 'cable', 'AWG', 'kcmil', 'THHN', 'THWN', 'XHHW',
                'multiconductor', 'stranded', 'solid', 'copper', 'aluminum', 'CU', 'AL'
            ],
            'raceway': [
                'EMT', 'IMC', 'RMC', 'PVC', 'RTRC', 'LFMC', 'LFNC', 'FMC',
                'electrical metallic tubing', 'intermediate metal conduit',
                'rigid metal conduit', 'liquidtight flexible metal conduit'
            ],
            'grounding': [
                'ground', 'grounded', 'grounding', 'earthing', 'bonding', 'GEC',
                'equipment grounding conductor', 'ground fault', 'GFCI', 'GFPE'
            ],
            'overcurrent': [
                'breaker', 'circuit breaker', 'fuse', 'overcurrent', 'overload',
                'short circuit', 'AFCI', 'arc fault', 'GFCI', 'ground fault'
            ],
            'special_locations': [
                'hazardous', 'classified', 'class I', 'class II', 'class III',
                'division 1', 'division 2', 'zone 0', 'zone 1', 'zone 2'
            ],
            'installation': [
                'support', 'securing', 'fastening', 'mounting', 'attachment',
                'spacing', 'interval', 'strap', 'hanger', 'bracket', 'anchor'
            ]
        }

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def correct_line_with_gpt(self, line: str) -> Tuple[str, float]:
        """Use GPT to correct potential OCR errors in a single line of text."""
        if not self.client or not line.strip():
            return line, 1.0
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert electrical code proofreader. Return a JSON object with: {'corrected_text': string, 'confidence': float (0-1)}"
                    },
                    {
                        "role": "user",
                        "content": f"Fix OCR errors in this line: {line}"
                    }
                ],
                temperature=0,
                response_format={"type": "json_object"}
            )
            
            result = json.loads(response.choices[0].message.content)
            return result.get("corrected_text", line), result.get("confidence", 1.0)
            
        except Exception as e:
            self.logger.error(f"Error during GPT OCR correction: {str(e)}")
            return line, 1.0

    def _identify_context(self, text: str) -> List[str]:
        """Identify technical context tags for the text."""
        text_lower = text.lower()
        tags = []
        for context, keywords in self.context_mapping.items():
            if any(keyword in text_lower for keyword in keywords):
                tags.append(context)
        return list(set(tags))

    def _find_related_sections(self, text: str) -> List[str]:
        """Find referenced sections in the text."""
        patterns = [
            r'(?:see\s+(?:Section\s+)?|with\s+)(\d+\.\d+(?:\([A-Z]\))?)',
            r'(?:in accordance with|as specified in)\s+(?:Section\s+)?(\d+\.\d+(?:\([A-Z]\))?)',
            r'(?:refer to|referenced in)\s+(?:Section\s+)?(\d+\.\d+(?:\([A-Z]\))?)'
        ]
        
        references = []
        for pattern in patterns:
            references.extend(re.findall(pattern, text, re.IGNORECASE))
        
        return list(set(references))

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def analyze_chunk_with_gpt(self, chunk: str) -> Dict:
        """Use GPT to clean text and analyze content."""
        if not self.client:
            return {}
            
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{
                    "role": "system", 
                    "content": "You are an expert electrical code analyzer processing NFPA 70 (NEC) text chunks. Return a JSON object with the following structure:\n\n\
                    article_number: string,\n\
                    article_title: string,\n\
                    section_number: string,\n\
                    section_title: string,\n\
                    content: string,              # Original content for reference\n\
                    cleaned_text: string,         # OCR-corrected but technically accurate text\n\
                    page_number: number,          # Preserve original page number\n\
                    context_tags: string[],       # Electrical context tags\n\
                    related_sections: string[],   # Referenced code sections\n\n\
                    gpt_analysis: {\n\
                        subsection_info: {\n\
                            letter: string,           # A, B, C, etc.\n\
                            continues_from: string,   # Previous section reference if mid-section\n\
                            continues_to: string      # Next section reference if continues\n\
                        },\n\
                        requirements: string[],     # Specific code requirements\n\
                        safety_elements: string[],  # Safety-related items\n\
                        equipment: string[]         # Equipment/components mentioned\n\
                    }"
                },
                {
                    "role": "user", 
                    "content": f"Clean and analyze this NEC text: {chunk}"
                }],
                temperature=0,
                response_format={"type": "json_object"}
            )
            
            try:
                result = json.loads(response.choices[0].message.content)
                return result
            except json.JSONDecodeError:
                self.logger.error("Failed to parse GPT response as JSON")
                return {}
                
        except Exception as e:
            self.logger.error(f"Error in GPT analysis: {str(e)}")
            return {}

    def chunk_nfpa70_content(self, pages_text: Dict[int, str]) -> List[CodeChunk]:
        """Process NFPA 70 content into context-aware chunks."""
        chunks = []
        self.position = CodePosition()
        
        if not pages_text:
            return chunks

        start_page = min(26, min(pages_text.keys()))

        for page_num in sorted(k for k in pages_text if k >= start_page):
            text = pages_text[page_num]
            lines = text.split('\n')
            current_chunk = []
            
            for line in lines:
                corrected_line, confidence = self.correct_line_with_gpt(line.strip())
                if not corrected_line:
                    continue

                self.position.update_from_text(corrected_line)
                
                if re.search(r'ARTICLE\s+\d+|^\d+\.\d+\s+[A-Z]', corrected_line):
                    if current_chunk:
                        chunk_text = ' '.join(current_chunk)
                        analysis_result = self.analyze_chunk_with_gpt(chunk_text)
                        
                        chunks.append(CodeChunk(
                            content=chunk_text,
                            page_number=page_num,
                            article_number=self.position.article_number,
                            article_title=self.position.article_title,
                            section_number=self.position.section_number,
                            section_title=self.position.section_title,
                            context_tags=self._identify_context(chunk_text),
                            related_sections=self._find_related_sections(chunk_text),
                            gpt_analysis=analysis_result,
                            hierarchy=self.position.hierarchy.copy(),
                            context_before=self.position.context_before,
                            context_after=self.position.context_after,
                            cleaned_text=analysis_result.get('cleaned_text'),
                            ocr_confidence=confidence
                        ))
                    current_chunk = [corrected_line]
                else:
                    current_chunk.append(corrected_line)
            
            if current_chunk:
                chunk_text = ' '.join(current_chunk)
                analysis_result = self.analyze_chunk_with_gpt(chunk_text)
                
                chunks.append(CodeChunk(
                    content=chunk_text,
                    page_number=page_num,
                    article_number=self.position.article_number,
                    article_title=self.position.article_title,
                    section_number=self.position.section_number,
                    section_title=self.position.section_title,
                    context_tags=self._identify_context(chunk_text),
                    related_sections=self._find_related_sections(chunk_text),
                    gpt_analysis=analysis_result,
                    hierarchy=self.position.hierarchy.copy(),
                    context_before=self.position.context_before,
                    context_after=self.position.context_after,
                    cleaned_text=analysis_result.get('cleaned_text'),
                    ocr_confidence=1.0
                ))

        # Link context before/after each chunk
        for i in range(len(chunks)):
            if i > 0:
                chunks[i].context_before = chunks[i-1].content[:200]
            if i < len(chunks) - 1:
                chunks[i].context_after = chunks[i+1].content[:200]

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
            "gpt_analysis": chunk.gpt_analysis or {},
            "cleaned_text": chunk.cleaned_text,
            "ocr_confidence": chunk.ocr_confidence
        }
        for chunk in chunks
    ]
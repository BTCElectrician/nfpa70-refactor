import re
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field
import logging
import json
from openai import OpenAI
from loguru import logger

@dataclass
class CodePosition:
    """Tracks the current position within the electrical code document.
    
    This class maintains hierarchical context as we process the document,
    ensuring we understand the relationship between chunks even when they
    don't contain explicit section numbers.
    """
    article_number: Optional[str] = None
    article_title: Optional[str] = None
    section_number: Optional[str] = None
    section_title: Optional[str] = None
    subsection_letter: Optional[str] = None
    hierarchy: List[str] = field(default_factory=list)
    context_before: Optional[str] = None
    context_after: Optional[str] = None

    def update_from_text(self, text: str) -> None:
        """Updates position based on text markers."""
        # Check for article headers
        article_match = re.search(r'ARTICLE\s+(\d+)\s*[-—]\s*(.+?)(?=\n|$)', text)
        if article_match:
            self.article_number = article_match.group(1)
            self.article_title = article_match.group(2).strip()
            self.hierarchy = [f"Article {self.article_number}"]

        # Check for section numbers
        section_match = re.search(r'(\d+\.\d+)\s+(.+?)(?=\n|$)', text)
        if section_match:
            self.section_number = section_match.group(1)
            self.section_title = section_match.group(2).strip()
            if self.hierarchy:
                self.hierarchy = self.hierarchy[:1] + [f"Section {self.section_number}"]

        # Check for subsections
        subsection_match = re.search(r'\(([A-Z])\)\s+', text)
        if subsection_match:
            self.subsection_letter = subsection_match.group(1)
            if len(self.hierarchy) >= 2:
                self.hierarchy = self.hierarchy[:2] + [f"Subsection ({self.subsection_letter})"]

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

class ElectricalCodeChunker:
    """Enhanced chunking for electrical code text with comprehensive NEC terminology."""
    
    def __init__(self, openai_api_key: Optional[str] = None):
        """Initialize the chunker with optional OpenAI integration."""
        self.logger = logger
        self.client = OpenAI(api_key=openai_api_key) if openai_api_key else None
        self.position = CodePosition()
        
        # Comprehensive NEC terminology mapping
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

    def _identify_context(self, text: str) -> List[str]:
        """Identify technical context tags for the text."""
        text_lower = text.lower()
        tags = []
        for context, keywords in self.context_mapping.items():
            if any(keyword in text_lower for keyword in keywords):
                tags.append(context)
        return tags

    def _find_related_sections(self, text: str) -> List[str]:
        """Find referenced sections in the text."""
        reference_pattern = re.compile(r'(?:see\s+(?:Section\s+)?|with\s+)(\d+\.\d+(?:\([A-Z]\))?)')
        return list(set(reference_pattern.findall(text)))

    def analyze_chunk_with_gpt(self, chunk: str) -> Dict:
        """Use GPT to analyze chunk content."""
        if not self.client:
            return {}
            
        # Define prompt parts separately to avoid f-string formatting issues
        prompt_prefix = "Analyze this section of the National Electrical Code carefully.\n\n"
        prompt_content = f"Code text:\n{chunk}\n\n"
        prompt_instructions = """Analyze and extract:
1. Technical Requirements
2. Safety Implications
3. Related References
4. Key Equipment

Return the analysis in this exact JSON structure:
{
    "requirements": [],
    "safety_elements": [],
    "related_sections": [],
    "equipment": []
}"""

        # Combine parts into final prompt
        prompt = prompt_prefix + prompt_content + prompt_instructions

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                response_format={ "type": "json_object" }
            )
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            self.logger.error(f"Error in GPT analysis: {str(e)}")
            return {}

    def chunk_nfpa70_content(self, pages_text: Dict[int, str]) -> List[CodeChunk]:
        """Process NFPA 70 content into context-aware chunks.
        
        Args:
            pages_text: Dict mapping page numbers to text content
            
        Returns:
            List of CodeChunk objects with enhanced context
        """
        chunks = []
        self.position = CodePosition()  # Reset position tracker
        
        # Skip initial pages that don't contain actual code content
        start_page = min(26, min(pages_text.keys()))  # Start at page 26 or first available
        
        for page_num in sorted(key for key in pages_text.keys() if key >= start_page):
            text = pages_text[page_num]
            lines = text.split('\n')
            current_chunk = []
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                # Update our position in the document
                self.position.update_from_text(line)
                
                # Check for new article or section
                if re.search(r'ARTICLE\s+\d+|^\d+\.\d+\s+[A-Z]', line):
                    # Save previous chunk if it exists
                    if current_chunk:
                        chunk_text = ' '.join(current_chunk)
                        gpt_analysis = self.analyze_chunk_with_gpt(chunk_text)
                        
                        chunks.append(CodeChunk(
                            content=chunk_text,
                            page_number=page_num,
                            article_number=self.position.article_number,
                            article_title=self.position.article_title,
                            section_number=self.position.section_number,
                            section_title=self.position.section_title,
                            context_tags=self._identify_context(chunk_text),
                            related_sections=self._find_related_sections(chunk_text),
                            gpt_analysis=gpt_analysis,
                            hierarchy=self.position.hierarchy.copy(),
                            context_before=self.position.context_before,
                            context_after=self.position.context_after
                        ))
                    current_chunk = [line]
                else:
                    current_chunk.append(line)
            
            # Handle last chunk on page
            if current_chunk:
                chunk_text = ' '.join(current_chunk)
                gpt_analysis = self.analyze_chunk_with_gpt(chunk_text)
                
                chunks.append(CodeChunk(
                    content=chunk_text,
                    page_number=page_num,
                    article_number=self.position.article_number,
                    article_title=self.position.article_title,
                    section_number=self.position.section_number,
                    section_title=self.position.section_title,
                    context_tags=self._identify_context(chunk_text),
                    related_sections=self._find_related_sections(chunk_text),
                    gpt_analysis=gpt_analysis,
                    hierarchy=self.position.hierarchy.copy(),
                    context_before=self.position.context_before,
                    context_after=self.position.context_after
                ))
        
        # Add context links between chunks
        for i in range(len(chunks)):
            if i > 0:
                chunks[i].context_before = chunks[i-1].content[:200]  # First 200 chars of previous chunk
            if i < len(chunks) - 1:
                chunks[i].context_after = chunks[i+1].content[:200]  # First 200 chars of next chunk
        
        return chunks


# Compatibility function for older code
def chunk_nfpa70_content(text: str, openai_api_key: Optional[str] = None) -> List[Dict]:
    """Wrapper for compatibility with existing code."""
    chunker = ElectricalCodeChunker(openai_api_key=openai_api_key)
    pages_text = {1: text}  # Wrap in dict for compatibility
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
            "gpt_analysis": chunk.gpt_analysis or {}
        }
        for chunk in chunks
    ]
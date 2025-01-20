import re
from typing import List, Dict, Optional
from dataclasses import dataclass, field
import logging
import json
from openai import OpenAI

@dataclass
class CodeChunk:
    """Represents a chunk of electrical code with context."""
    content: str
    page_number: int
    article_number: Optional[str] = None
    article_title: Optional[str] = None
    section_number: Optional[str] = None
    section_title: Optional[str] = None
    context_tags: List[str] = field(default_factory=list)
    related_sections: List[str] = field(default_factory=list)
    gpt_analysis: Dict = field(default_factory=dict)

class ElectricalCodeChunker:
    """Enhanced chunking for electrical code text with comprehensive NEC terminology."""
    
    def __init__(self, openai_api_key: Optional[str] = None):
        self.logger = logging.getLogger(__name__)
        self.client = OpenAI(api_key=openai_api_key) if openai_api_key else None
        
        # Comprehensive NEC terminology mapping
        self.context_mapping = {
            # Power Distribution
            'service_equipment': [
                'service equipment', 'service entrance', 'service drop', 'service lateral',
                'meter', 'metering', 'service disconnect', 'main disconnect', 'service panel',
                'switchboard', 'switchgear', 'panelboard'
            ],
            
            # Conductors and Raceways
            'conductors': [
                'conductor', 'wire', 'cable', 'AWG', 'kcmil', 'THHN', 'THWN', 'XHHW',
                'multiconductor', 'stranded', 'solid', 'copper', 'aluminum', 'CU', 'AL',
                'current-carrying', 'neutral', 'ungrounded', 'phase'
            ],
            
            'raceway': [
                'EMT', 'IMC', 'RMC', 'PVC', 'RTRC', 'LFMC', 'LFNC', 'FMC',
                'electrical metallic tubing', 'intermediate metal conduit', 
                'rigid metal conduit', 'liquidtight flexible metal conduit',
                'schedule 40', 'schedule 80', 'nipple', 'chase', 'sleeve'
            ],
            
            # Grounding and Bonding
            'grounding': [
                'ground', 'grounded', 'grounding', 'earthing', 'bonding', 'GEC',
                'equipment grounding conductor', 'ground fault', 'GFCI', 'GFPE',
                'grounding electrode', 'ground rod', 'ufer', 'made electrode',
                'ground ring', 'ground plate', 'ground bus', 'isolated ground'
            ],
            
            # Overcurrent Protection
            'overcurrent': [
                'breaker', 'circuit breaker', 'fuse', 'overcurrent', 'overload',
                'short circuit', 'AFCI', 'arc fault', 'GFCI', 'ground fault',
                'fusible', 'nonfusible', 'instantaneous trip', 'adjustable trip'
            ],
            
            # Special Occupancies
            'special_locations': [
                'hazardous', 'classified', 'class I', 'class II', 'class III',
                'division 1', 'division 2', 'zone 0', 'zone 1', 'zone 2',
                'wet location', 'damp location', 'corrosive', 'hospital',
                'healthcare', 'assembly', 'theater', 'motor fuel', 'spray booth'
            ],
            
            # Motors and Controls
            'motors': [
                'motor', 'controller', 'starter', 'VFD', 'variable frequency',
                'horsepower', 'hp', 'full-load', 'locked rotor', 'duty cycle',
                'continuous duty', 'overload', 'disconnecting means'
            ],
            
            # Equipment
            'transformers': [
                'transformer', 'xfmr', 'xfmer', 'primary', 'secondary',
                'delta', 'wye', 'impedance', 'kVA', 'step-up', 'step-down',
                'dry-type', 'liquid-filled', 'vault'
            ],
            
            'hvac': [
                'air conditioning', 'HVAC', 'heat pump', 'condenser',
                'evaporator', 'cooling', 'heating', 'disconnecting means',
                'minimum circuit ampacity', 'maximum overcurrent'
            ],
            
            # Installation Requirements
            'installation': [
                'support', 'securing', 'fastening', 'mounting', 'attachment',
                'spacing', 'interval', 'strap', 'hanger', 'bracket', 'anchor',
                'embedded', 'concealed', 'exposed'
            ],
            
            'clearance': [
                'clearance', 'spacing', 'distance', 'separation', 'depth',
                'working space', 'dedicated space', 'headroom', 'minimum depth',
                'burial depth', 'cover'
            ],
            
            # Emergency Systems
            'emergency': [
                'emergency', 'legally required standby', 'optional standby',
                'backup', 'generator', 'transfer switch', 'automatic transfer',
                'manual transfer', 'essential electrical system'
            ],
            
            # Branch Circuits and Feeders
            'branch_circuits': [
                'branch circuit', 'feeder', 'multiwire', 'general purpose',
                'dedicated circuit', 'small appliance', 'laundry', 'SABC',
                'receptacle', 'outlet', 'lighting', '15-amp', '20-amp', '30-amp'
            ],
            
            # Special Equipment
            'welding': [
                'welder', 'welding outlet', 'welding receptacle', 'electrode',
                'duty cycle', 'demand factor'
            ],
            
            'ev_charging': [
                'electric vehicle', 'EV', 'charging equipment', 'EVSE',
                'fast charging', 'level 2', 'level 3', 'charging station'
            ],
            
            # Fire Alarm Systems
            'fire_alarm': [
                'fire alarm', 'smoke detector', 'heat detector', 'initiating device',
                'notification appliance', 'fire alarm control unit', 'FACU', 'FACP'
            ],
            
            # Solar PV Systems
            'solar': [
                'photovoltaic', 'PV', 'solar', 'inverter', 'rapid shutdown',
                'module', 'array', 'combiner', 'micro-inverter', 'optimizer'
            ],
            
            # Communications
            'communications': [
                'network', 'data', 'telephone', 'coaxial', 'fiber optic',
                'category', 'CAT', 'plenum', 'riser', 'cable tray', 'J-hooks'
            ]
        }
        
        # Compile regex patterns
        self.article_pattern = re.compile(r'ARTICLE\s+(\d+)\s*[-—]\s*(.+?)(?=\n|$)')
        self.section_pattern = re.compile(r'(\d+\.\d+(?:\([A-Z]\))?)\s+(.+?)(?=\n|$)')
        self.reference_pattern = re.compile(r'(?:see\s+(?:Section\s+)?|with\s+)(\d+\.\d+(?:\([A-Z]\))?)')

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
        return list(set(self.reference_pattern.findall(text)))

    def analyze_chunk_with_gpt(self, chunk: str) -> Dict:
        """Use GPT to analyze code chunk content."""
        if not self.client:
            return {}
            
        prompt = f"""Analyze this NFPA 70 electrical code section and extract:
        1. Key technical requirements
        2. Equipment specifications
        3. Cross-references to other sections
        4. Safety-critical elements

        Code section:
        {chunk}

        Provide response in JSON format.
        """
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",  # Fast, affordable small model for focused tasks
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                response_format={ "type": "json_object" }
            )
            
            return json.loads(response.choices[0].message.content)
            
        except Exception as e:
            self.logger.error(f"Error in GPT analysis: {str(e)}")
            return {}

    def chunk_nfpa70_content(self, pages_text: Dict[int, str]) -> List[CodeChunk]:
        """
        Process NFPA 70 content into context-aware chunks.
        Args:
            pages_text: Dict[page_number, text]
        Returns:
            List of CodeChunk objects
        """
        chunks = []
        current_article = None
        current_article_title = None
        
        for page_num, text in pages_text.items():
            lines = text.split('\n')
            current_chunk = []
            current_section = None
            current_section_title = None
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                # Check for article header
                article_match = self.article_pattern.search(line)
                if article_match:
                    current_article = article_match.group(1)
                    current_article_title = article_match.group(2).strip()
                    continue
                
                # Check for section header
                section_match = self.section_pattern.match(line)
                if section_match:
                    # Save previous chunk if exists
                    if current_chunk:
                        chunk_text = ' '.join(current_chunk)
                        gpt_analysis = self.analyze_chunk_with_gpt(chunk_text)
                        chunks.append(CodeChunk(
                            content=chunk_text,
                            page_number=page_num,
                            article_number=current_article,
                            article_title=current_article_title,
                            section_number=current_section,
                            section_title=current_section_title,
                            context_tags=self._identify_context(chunk_text),
                            related_sections=self._find_related_sections(chunk_text),
                            gpt_analysis=gpt_analysis
                        ))
                    
                    current_section = section_match.group(1)
                    current_section_title = section_match.group(2).strip()
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
                    article_number=current_article,
                    article_title=current_article_title,
                    section_number=current_section,
                    section_title=current_section_title,
                    context_tags=self._identify_context(chunk_text),
                    related_sections=self._find_related_sections(chunk_text),
                    gpt_analysis=gpt_analysis
                ))
        
        return chunks

# Compatibility function for existing code
def chunk_nfpa70_content(text: str, openai_api_key: Optional[str] = None) -> List[Dict]:
    """Wrapper for compatibility with existing code."""
    chunker = ElectricalCodeChunker(openai_api_key=openai_api_key)
    pages_text = {1: text}
    chunks = chunker.chunk_nfpa70_content(pages_text)
    
    return [
        {
            "content": chunk.content,
            "metadata": {
                "section": chunk.section_number,
                "article": chunk.article_number,
                "page": chunk.page_number
            },
            "context_tags": chunk.context_tags,
            "related_sections": chunk.related_sections,
            "gpt_analysis": chunk.gpt_analysis
        }
        for chunk in chunks
    ] 
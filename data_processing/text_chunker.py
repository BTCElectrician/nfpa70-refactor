import re
from typing import List, Dict, Optional, Any, Tuple, Sequence
from dataclasses import dataclass, field
import asyncio
from loguru import logger
from tenacity import AsyncRetrying, stop_after_attempt, retry_if_exception_type, wait_exponential
import json
from openai import AsyncOpenAI, APIError
from contextlib import asynccontextmanager

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
    context_tags: List[str]          # Technical context tags
    related_sections: List[str]      # Referenced code sections

class ElectricalCodeChunker:
    """Chunks electrical code text with batched GPT-based cleanup and analysis."""
    
    def __init__(self, 
                 openai_api_key: Optional[str] = None,
                 batch_size: int = 50,
                 max_concurrent_requests: int = 5):
        """
        Initialize the chunker with configurable batch parameters.
        
        Args:
            openai_api_key: API key for OpenAI
            batch_size: Number of chunks to process in one batch
            max_concurrent_requests: Maximum number of concurrent API requests
        """
        self.logger = logger.bind(context="chunker")
        self.client = AsyncOpenAI(
            api_key=openai_api_key,
            timeout=30.0,
            max_retries=5
        ) if openai_api_key else None
        self.batch_size = batch_size
        self.semaphore = asyncio.Semaphore(max_concurrent_requests)
        
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

    @asynccontextmanager
    async def _api_semaphore(self):
        """Context manager for API rate limiting."""
        async with self.semaphore:
            yield

    async def _process_chunk_batch(self, chunks: Sequence[str]) -> List[Dict]:
        """Process multiple chunks in a single GPT call with proper async retry."""
        if not self.client or not chunks:
            return [{} for _ in chunks]
            
        try:
            async for attempt in AsyncRetrying(
                stop=stop_after_attempt(3),
                wait=wait_exponential(multiplier=1, min=4, max=10),
                retry=retry_if_exception_type((TimeoutError, APIError))
            ):
                with attempt:
                    async with self._api_semaphore():
                        response = await self.client.chat.completions.create(
                            model="gpt-4o-mini",
                            messages=[{
                                "role": "system", 
                                "content": """You are processing NFPA 70 (NEC) text. Clean any OCR errors and analyze the content.
                                For each text chunk, return a JSON object with:
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
                                "content": f"Process these NEC text chunks: {json.dumps(chunks)}"
                            }],
                            timeout=45.0,  # Longer timeout for batch processing
                            temperature=0,
                            response_format={"type": "json_object"}
                        )
                        
                        results = json.loads(response.choices[0].message.content)
                        return results.get("chunks", [{} for _ in chunks])
                        
        except Exception as e:
            self.logger.error(f"Error in GPT batch processing: {str(e)}")
            return [{} for _ in chunks]

    async def process_chunks_async(self, chunks: List[str]) -> List[Dict]:
        """Process all chunks with batching."""
        results = []
        for i in range(0, len(chunks), self.batch_size):
            batch = chunks[i:i + self.batch_size]
            self.logger.info(f"Processing batch {i//self.batch_size + 1}, size: {len(batch)}")
            
            batch_results = await self._process_chunk_batch(batch)
            results.extend(batch_results)
            
            self.logger.debug(f"Completed batch {i//self.batch_size + 1}")
            
        return results

    def chunk_nfpa70_content(self, pages_text: Dict[int, str]) -> List[CodeChunk]:
        """Process NFPA 70 content into context-aware chunks."""
        chunks = []
        raw_chunks = []
        
        if not pages_text:
            return chunks

        # Extract chunks first
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
                        raw_chunks.append(' '.join(current_chunk))
                    current_chunk = [line]
                else:
                    current_chunk.append(line)
            
            if current_chunk:
                raw_chunks.append(' '.join(current_chunk))

        # Process chunks in batches asynchronously
        self.logger.info(f"Processing {len(raw_chunks)} chunks in batches of {self.batch_size}")
        
        # Run async processing in event loop
        loop = asyncio.get_event_loop()
        chunk_analyses = loop.run_until_complete(self.process_chunks_async(raw_chunks))
        
        # Convert results to CodeChunk objects
        for i, analysis in enumerate(chunk_analyses):
            if not analysis:  # Skip empty results
                continue
                
            chunks.append(CodeChunk(
                content=analysis.get('content', raw_chunks[i]),
                page_number=start_page + (i // 10),  # Approximate page number
                article_number=analysis.get('article_number'),
                article_title=analysis.get('article_title'),
                section_number=analysis.get('section_number'),
                section_title=analysis.get('section_title'),
                context_tags=analysis.get('context_tags', []),
                related_sections=analysis.get('related_sections', [])
            ))

        self.logger.success(f"Successfully processed {len(chunks)} chunks")
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

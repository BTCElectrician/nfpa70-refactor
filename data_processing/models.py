from __future__ import annotations
from typing import List, Optional, Dict, Any, TYPE_CHECKING
from pydantic import BaseModel, Field, validator
from datetime import datetime

if TYPE_CHECKING:
    from .text_chunker import CodeChunk  # Relative import since we're in the same package

class NFPAChunk(BaseModel):
    """Represents a chunk of NFPA 70 content with metadata."""
    content: str
    page_number: str  # Using string to maintain "70-XX" format
    article_number: Optional[str] = None
    section_number: Optional[str] = None
    article_title: Optional[str] = None
    section_title: Optional[str] = None
    context_tags: List[str] = Field(default_factory=list)
    related_sections: List[str] = Field(default_factory=list)

    @validator('page_number')
    def validate_page_number(cls, v: str) -> str:
        """Validate NFPA page number format (70-XX)."""
        if not v.startswith('70-'):
            raise ValueError(f'Page number {v} must start with "70-"')
        try:
            page_num = int(v.split('-')[1])
            if not (1 <= page_num <= 1000):  # Adjust range as needed
                raise ValueError(f'Page number {page_num} outside valid range')
        except (IndexError, ValueError) as e:
            raise ValueError(f'Invalid page number format: {v}') from e
        return v

    def to_dict(self) -> Dict[str, Any]:
        """Convert to your existing storage format."""
        return {
            "content": self.content,
            "page_number": self.page_number,
            "article_number": self.article_number or "",
            "section_number": self.section_number or "",
            "article_title": self.article_title or "",
            "section_title": self.section_title or "",
            "context_tags": list(self.context_tags),
            "related_sections": list(self.related_sections)
        }

    @classmethod
    def from_code_chunk(cls, chunk: 'CodeChunk') -> 'NFPAChunk':
        """Convert from your existing CodeChunk to NFPAChunk."""
        return cls(
            content=chunk.content,
            page_number=chunk.page_number,  # Already in "70-XX" format
            article_number=chunk.article_number,
            section_number=chunk.section_number,
            article_title=chunk.article_title,
            section_title=chunk.section_title,
            context_tags=chunk.context_tags,
            related_sections=chunk.related_sections
        )

class ChunkBatch(BaseModel):
    """Represents a batch of NFPA chunks."""
    chunks: List[NFPAChunk]
    processed_at: datetime = Field(default_factory=datetime.utcnow)
    
    def to_storage_format(self) -> Dict[str, Any]:
        """Convert to format compatible with existing storage."""
        return {
            "chunks": [chunk.to_dict() for chunk in self.chunks]
        }
from typing import List, Dict, Optional, Any, Union
from enum import Enum
from datetime import datetime
from dataclasses import dataclass, field


class ProcessingStatus(Enum):
    """Status of chunk processing steps"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    ERROR = "error"


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


@dataclass
class CodeChunk:
    """Represents a chunk of electrical code with essential metadata."""
    content: str                     # The actual text content of the chunk
    page_number: int                 # NFPA page number (70-XX)
    article_number: Optional[str] = None    # Article number (e.g., "90")
    section_number: Optional[str] = None    # Section number (e.g., "90.2")
    article_title: Optional[str] = None     # Title of the article
    section_title: Optional[str] = None     # Title of the section
    context_tags: List[str] = field(default_factory=list)  # Technical context tags
    related_sections: List[str] = field(default_factory=list)  # Referenced code sections
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert CodeChunk to dictionary for storage or indexing."""
        return {
            "content": self.content,
            "page_number": self.page_number,
            "article_number": self.article_number,
            "section_number": self.section_number,
            "article_title": self.article_title or "",
            "section_title": self.section_title or "",
            "context_tags": list(self.context_tags) if self.context_tags else [],
            "related_sections": list(self.related_sections) if self.related_sections else []
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CodeChunk":
        """Create CodeChunk from dictionary data."""
        return cls(
            content=data.get("content", ""),
            page_number=data.get("page_number", 0),
            article_number=data.get("article_number"),
            section_number=data.get("section_number"),
            article_title=data.get("article_title", ""),
            section_title=data.get("section_title", ""),
            context_tags=data.get("context_tags", []),
            related_sections=data.get("related_sections", [])
        )


@dataclass
class ChunkBatch:
    """Collection of code chunks for batch processing."""
    chunks: List[CodeChunk] = field(default_factory=list)
    status: ProcessingStatus = ProcessingStatus.PENDING
    error_message: Optional[str] = None
    
    def to_storage_format(self) -> Dict[str, Any]:
        """Prepare batch data for storage."""
        return {
            "chunks": [chunk.to_dict() for chunk in self.chunks],
            "status": self.status.value,
            "error_message": self.error_message
        }
    
    @classmethod
    def from_storage_format(cls, data: Dict[str, Any]) -> "ChunkBatch":
        """Create batch from storage data."""
        chunks = [CodeChunk.from_dict(chunk_data) for chunk_data in data.get("chunks", [])]
        status_str = data.get("status", ProcessingStatus.PENDING.value)
        
        try:
            status = ProcessingStatus(status_str)
        except ValueError:
            status = ProcessingStatus.PENDING
            
        return cls(
            chunks=chunks,
            status=status,
            error_message=data.get("error_message")
        )


@dataclass
class ProcessingMetrics:
    """Metrics for monitoring processing performance."""
    total_chunks: int = 0
    processed_chunks: int = 0
    successful_chunks: int = 0
    failed_chunks: int = 0
    total_processing_time: float = 0.0
    avg_processing_time: float = 0.0
    
    def update_from_batch(self, 
                         batch_size: int, 
                         successful: int, 
                         failed: int, 
                         processing_time: float) -> None:
        """Update metrics with batch results."""
        self.total_chunks += batch_size
        self.processed_chunks += (successful + failed)
        self.successful_chunks += successful
        self.failed_chunks += failed
        self.total_processing_time += processing_time
        
        if self.processed_chunks > 0:
            self.avg_processing_time = self.total_processing_time / self.processed_chunks


class TokenUsageMetrics:
    """Tracks token usage for OpenAI API calls."""
    def __init__(self):
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0
        
    def update(self, prompt_tokens: int, completion_tokens: int) -> None:
        """Update token metrics."""
        self.prompt_tokens += prompt_tokens
        self.completion_tokens += completion_tokens
        self.total_tokens = self.prompt_tokens + self.completion_tokens
    
    def __str__(self) -> str:
        """String representation of token usage."""
        return (f"TokenUsage(prompt={self.prompt_tokens}, "
                f"completion={self.completion_tokens}, "
                f"total={self.total_tokens})")
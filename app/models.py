from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Union
from datetime import datetime
from enum import Enum


class Question(BaseModel):
    """A question with an optional answer"""
    question: str
    answer: Optional[str] = None


class Questions(BaseModel):
    """A collection of questions"""
    questions: List[Question] = Field(default_factory=list, description="List of questions")


class LegalQueryRequest(BaseModel):
    """Request model for legal query API"""
    question: str


class LegalQueryResponse(BaseModel):
    """Response model for legal query API"""
    original_question: str
    decomposed_questions: List[Question] = Field(default_factory=list)
    final_answer: str

    # Optional metadata about the documents used
    document_metadata: Optional[List[Dict[str, Any]]] = None


class DocumentMetadata(BaseModel):
    """Metadata about a document returned in search results"""
    id: str
    score: float
    case_title: Optional[str] = None
    court: Optional[str] = None
    year: Optional[int] = None
    source: Optional[str] = None


# ========================================
# Chat Models for Legal Query Functionality
# ========================================

class SourceType(str, Enum):
    """Enumeration for different legal source types"""
    DECOMPOSITION = "decomposition"
    LEGAL_DOCUMENT = "legal_document"
    STATUTE = "statute"
    CASE = "case"
    REGULATION = "regulation"
    EXTERNAL_RESEARCH = "external_research"
    CONTEXT = "context"


class ConfidenceLevel(str, Enum):
    """Confidence levels for legal sources"""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class LegalSource(BaseModel):
    """
    Unified source model for legal information tracking.
    Combines document metadata with legal-specific fields.
    """
    title: str = Field(..., description="Source title")
    content_preview: str = Field(..., description="Brief content preview")
    relevance_score: float = Field(..., ge=0.0, le=1.0, description="Relevance score")
    source_type: SourceType = Field(..., description="Type of legal source")

    # Optional fields for different source types
    document_id: Optional[str] = Field(None, description="Document identifier")
    citation: Optional[str] = Field(None, description="Legal citation")
    jurisdiction: Optional[str] = Field(None, description="Legal jurisdiction")
    year: Optional[int] = Field(None, description="Year of source")
    court: Optional[str] = Field(None, description="Court name for cases")
    url: Optional[str] = Field(None, description="Source URL if available")

    # Metadata and display information
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    display_priority: int = Field(default=1, description="Display order priority")
    confidence_level: Optional[ConfidenceLevel] = Field(None, description="Source confidence")

    class Config:
        use_enum_values = True


class ToolCallResult(BaseModel):
    """Model for tracking external tool calls and their results"""
    tool_name: str = Field(..., description="Name of the tool called")
    arguments: Dict[str, Any] = Field(..., description="Arguments passed to tool")
    result: Dict[str, Any] = Field(..., description="Result from tool call")
    success: bool = Field(..., description="Whether tool call succeeded")
    timestamp: datetime = Field(..., description="When tool was called")
    sources_generated: List[LegalSource] = Field(default_factory=list, description="Sources from this tool")


class ChatMessage(BaseModel):
    """Chat message model for conversation tracking"""
    role: str = Field(..., description="Message role: user, assistant, system")
    content: str = Field(..., description="Message content")
    timestamp: Optional[datetime] = Field(None, description="Message timestamp")


class LegalChatResponse(BaseModel):
    """Response model for legal chat with comprehensive source tracking"""
    response: str = Field(..., description="Chat response content")
    sources: List[LegalSource] = Field(default_factory=list, description="Sources used in response")
    conversation_id: str = Field(..., description="Conversation identifier")
    timestamp: datetime = Field(default_factory=datetime.now, description="Response timestamp")

    # Research tracking
    external_research_used: bool = Field(default=False, description="Whether external research was used")
    tools_called: List[ToolCallResult] = Field(default_factory=list, description="Tools called during research")
    processing_time_seconds: Optional[float] = Field(None, description="Processing time")

    # Context information
    previous_decomposition_used: bool = Field(default=False, description="Used previous decomposition results")


class LegalQueryRequestWithChat(BaseModel):
    """Request model for legal queries with chat support"""
    question: str = Field(..., description="Legal question to answer")
    enable_followup: bool = Field(default=True, description="Enable follow-up chat")
    conversation_id: Optional[str] = Field(None, description="Existing conversation ID")


class LegalQueryResponseWithChat(BaseModel):
    """Enhanced legal query response that supports chat follow-up"""
    # Original fields from LegalQueryResponse
    original_question: str
    decomposed_questions: List[Question] = Field(default_factory=list)
    final_answer: str
    document_metadata: Optional[List[Dict[str, Any]]] = None

    # Chat integration fields
    supports_followup: bool = Field(default=True, description="Whether follow-up is supported")
    conversation_id: Optional[str] = Field(None, description="Conversation ID for follow-up")
    processing_time: float = Field(..., description="Total processing time")
    cache_hit: bool = Field(default=False, description="Whether result was cached")

    # Enhanced source information
    sources: List[LegalSource] = Field(default_factory=list, description="All sources used")

    def create_chat_context(self) -> Dict[str, Any]:
        """Create context for follow-up chat conversations"""
        return {
            "original_decomposition": {
                "question": self.original_question,
                "sub_questions": [q.model_dump() for q in self.decomposed_questions],
                "answer": self.final_answer,
                "document_metadata": self.document_metadata,
                "sources": [source.model_dump() for source in self.sources]
            },
            "conversation_id": self.conversation_id,
            "timestamp": datetime.now().isoformat()
        }


# Factory methods for creating sources from different origins
class SourceFactory:
    """Factory for creating LegalSource instances from different data sources"""

    @staticmethod
    def from_decomposition_result(doc_meta: Dict[str, Any]) -> LegalSource:
        """Create source from decomposition pipeline result"""
        title = (
            doc_meta.get("case_title") or
            doc_meta.get("article_title") or
            doc_meta.get("legislation_title") or
            "Legal Document"
        )

        return LegalSource(
            title=title,
            content_preview=f"Relevance Score: {doc_meta.get('score', 0):.3f}",
            relevance_score=doc_meta.get("score", 0),
            source_type=SourceType.LEGAL_DOCUMENT,
            document_id=doc_meta.get("document_id"),
            citation=doc_meta.get("citation"),
            jurisdiction=doc_meta.get("jurisdiction", "Nigeria"),
            year=doc_meta.get("year"),
            court=doc_meta.get("court"),
            metadata=doc_meta,
            display_priority=1,
            confidence_level=ConfidenceLevel.HIGH
        )

    @staticmethod
    def from_external_research(research_result: Dict[str, Any], source_type: SourceType) -> LegalSource:
        """Create source from external legal research"""
        return LegalSource(
            title=research_result.get("title", "Legal Research"),
            content_preview=research_result.get("summary", "")[:200],
            relevance_score=research_result.get("relevance", 0.8),
            source_type=source_type,
            citation=research_result.get("citation"),
            jurisdiction=research_result.get("jurisdiction", "Nigeria"),
            year=research_result.get("year"),
            court=research_result.get("court"),
            url=research_result.get("url"),
            metadata={
                "tool_used": research_result.get("tool_used"),
                "research_timestamp": research_result.get("timestamp")
            },
            display_priority=2,
            confidence_level=ConfidenceLevel.MEDIUM
        )

    @staticmethod
    def from_decomposition_question(question: Question, index: int) -> LegalSource:
        """Create source from decomposed question"""
        return LegalSource(
            title=f"Legal Sub-Question {index + 1}",
            content_preview=question.question,
            relevance_score=1.0,
            source_type=SourceType.DECOMPOSITION,
            metadata={
                "question_index": index,
                "original_answer": question.answer,
                "question_type": "decomposed_query"
            },
            display_priority=0,  # Highest priority
            confidence_level=ConfidenceLevel.HIGH
        )

    @staticmethod
    def from_conversation_context(context: str, confidence: float = 0.5) -> LegalSource:
        """Create source from previous conversation context"""
        return LegalSource(
            title="Previous Conversation Context",
            content_preview=context[:200],
            relevance_score=confidence,
            source_type=SourceType.CONTEXT,
            metadata={
                "context_type": "conversation_history",
                "created_at": datetime.now().isoformat()
            },
            display_priority=3,
            confidence_level=ConfidenceLevel.LOW
        )
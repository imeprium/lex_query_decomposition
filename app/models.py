from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any


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
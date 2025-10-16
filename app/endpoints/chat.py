"""
Chat API endpoints for legal query conversational functionality.
Follows RESTful principles and maintains clean separation from existing endpoints.
"""

import logging
from typing import Optional, Dict, Any

from fastapi import APIRouter, HTTPException, Query, Depends
from fastapi.responses import Response
from pydantic import BaseModel, Field

from app.models import (
    Question, LegalQueryRequest, LegalQueryResponse, DocumentMetadata,
    LegalChatResponse, ChatMessage, LegalSource, ToolCallResult,
    SourceType, ConfidenceLevel, SourceFactory, LegalQueryResponseWithChat,
    LegalQueryRequestWithChat
)
from app.services.unified_chat_service import get_unified_chat_service
from app.utils.sanitizer import sanitize_legal_query_body
from app.auth import (
    require_chat_access,
    require_legal_research_access,
    get_user_context
)

logger = logging.getLogger("chat_api")
router = APIRouter(prefix="/api/chat", tags=["Chat"])


# Request models
class FollowupQuestionRequest(BaseModel):
    """Request model for follow-up questions"""
    question: str = Field(..., description="Follow-up legal question")


@router.post("/start", response_model=Dict[str, Any])
async def start_legal_chat(
    request: LegalQueryRequestWithChat,
    user_context: dict = Depends(get_user_context),
    _: None = Depends(require_chat_access),
    __: None = Depends(require_legal_research_access)
):
    """
    Start a new legal chat session, optionally with initial decomposition.

    Args:
        request: Legal query request with chat options

    Returns:
        Chat session with initial response and conversation ID
    """
    # Sanitize the question from the request body
    sanitized_question = sanitize_legal_query_body(request.question)
    logger.info(f"Starting new chat session: {sanitized_question[:100]}...")

    try:
        unified_service = get_unified_chat_service()

        # Start chat session using the working unified service
        chat_response = await unified_service.start_chat(
            initial_question=sanitized_question,
            enable_decomposition=request.enable_followup  # Respect the request parameter
        )

        return {
            "response": chat_response.response,
            "conversation_id": chat_response.conversation_id,
            "sources": [source.model_dump() for source in chat_response.sources],
            "supports_followup": True,
            "processing_time": chat_response.processing_time_seconds,
            "timestamp": chat_response.timestamp.isoformat(),
            "external_research_used": chat_response.external_research_used,
            "tools_called": [tool.model_dump() for tool in chat_response.tools_called],
            "previous_decomposition_used": chat_response.previous_decomposition_used
        }

    except Exception as e:
        logger.error(f"Failed to start chat session: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to start chat session: {str(e)}"
        )


@router.post("/continue/{conversation_id}", response_model=Dict[str, Any])
async def continue_legal_chat(
    conversation_id: str,
    request: FollowupQuestionRequest,
    user_context: dict = Depends(get_user_context),
    _: None = Depends(require_chat_access)
):
    """
    Continue an existing legal chat conversation with a follow-up question.

    Args:
        conversation_id: Existing conversation ID
        request: Follow-up question request

    Returns:
        Chat response with sources and updated conversation context
    """
    # Sanitize the follow-up question
    sanitized_question = sanitize_legal_query_body(request.question)
    logger.info(f"Continuing chat {conversation_id}: {sanitized_question[:50]}...")

    try:
        unified_service = get_unified_chat_service()

        # Continue chat conversation using the working unified service
        chat_response = await unified_service.continue_chat(
            question=sanitized_question,
            conversation_id=conversation_id
        )

        return {
            "response": chat_response.response,
            "conversation_id": chat_response.conversation_id,
            "sources": [source.model_dump() for source in chat_response.sources],
            "processing_time": chat_response.processing_time_seconds,
            "timestamp": chat_response.timestamp.isoformat(),
            "external_research_used": chat_response.external_research_used,
            "tools_called": [tool.model_dump() for tool in chat_response.tools_called],
            "previous_decomposition_used": chat_response.previous_decomposition_used
        }

    except Exception as e:
        logger.error(f"Failed to continue chat: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to continue chat: {str(e)}"
        )


@router.get("/conversations/{conversation_id}/history")
async def get_conversation_history(
    conversation_id: str,
    user_context: dict = Depends(get_user_context),
    _: None = Depends(require_chat_access)
):
    """
    Get the complete conversation history for a given conversation ID.

    Args:
        conversation_id: Conversation ID

    Returns:
        Complete conversation history with metadata
    """
    logger.info(f"Retrieving conversation history: {conversation_id}")

    try:
        unified_service = get_unified_chat_service()
        history = unified_service.get_conversation_history(conversation_id)

        if not history:
            raise HTTPException(
                status_code=404,
                detail="Conversation not found"
            )

        return {
            "conversation_id": conversation_id,
            "messages": [
                {
                    "role": msg.role,
                    "content": msg.content,
                    "timestamp": msg.timestamp.isoformat() if msg.timestamp else None
                }
                for msg in history
            ],
            "message_count": len(history),
            "retrieved_at": unified_service.memory.conversation_metadata.get(
                conversation_id, {}
            ).get("last_updated")
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get conversation history: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve conversation history: {str(e)}"
        )


@router.delete("/conversations/{conversation_id}")
async def clear_conversation(
    conversation_id: str,
    user_context: dict = Depends(get_user_context),
    _: None = Depends(require_chat_access)
):
    """
    Clear/delete a conversation and its history.

    Args:
        conversation_id: Conversation ID to clear

    Returns:
        Confirmation of conversation deletion
    """
    logger.info(f"Clearing conversation: {conversation_id}")

    try:
        unified_service = get_unified_chat_service()
        success = unified_service.clear_conversation(conversation_id)

        if not success:
            raise HTTPException(
                status_code=404,
                detail="Conversation not found"
            )

        return {
            "message": f"Conversation {conversation_id} cleared successfully",
            "conversation_id": conversation_id
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to clear conversation: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to clear conversation: {str(e)}"
        )


@router.post("/ask-with-followup", response_model=Dict[str, Any])
async def ask_legal_question_with_followup(
    request: LegalQueryRequestWithChat,
    user_context: dict = Depends(get_user_context),
    _: None = Depends(require_chat_access),
    __: None = Depends(require_legal_research_access)
):
    """
    Enhanced version of the original ask endpoint that supports follow-up chat.
    Uses the existing decomposition pipeline but adds chat capabilities.

    Args:
        request: Legal query request with chat options

    Returns:
        Enhanced legal query response with chat support
    """
    # Sanitize the question from the request body
    sanitized_question = sanitize_legal_query_body(request.question)
    logger.info(f"Processing question with follow-up support: {sanitized_question[:100]}...")

    try:
        unified_service = get_unified_chat_service()

        # Process with chat support using the working unified service
        response = await unified_service.start_chat(
            initial_question=sanitized_question,
            enable_decomposition=request.enable_followup
        )

        # Return JSON response (default format)
        return {
            "original_question": sanitized_question,
            "decomposed_questions": [],  # This is handled internally by the unified service
            "final_answer": response.response,
            "document_metadata": {},  # This is handled internally by the unified service
            "sources": [source.model_dump() for source in response.sources],
            "supports_followup": True,
            "conversation_id": response.conversation_id,
            "processing_time": response.processing_time_seconds,
            "cache_hit": False,  # This is handled internally by the unified service
            "external_research_used": response.external_research_used,
            "tools_called": [tool.model_dump() for tool in response.tools_called]
        }

    except Exception as e:
        logger.error(f"Failed to process question with follow-up: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process question: {str(e)}"
        )


@router.post("/followup/{conversation_id}", response_model=Dict[str, Any])
async def ask_followup_question(
    conversation_id: str,
    request: FollowupQuestionRequest,
    user_context: dict = Depends(get_user_context),
    _: None = Depends(require_chat_access)
):
    """
    Ask a follow-up question in an existing conversation.

    Args:
        conversation_id: Existing conversation ID
        request: Follow-up question request

    Returns:
        Chat response with comprehensive source tracking
    """
    # Sanitize the follow-up question
    sanitized_question = sanitize_legal_query_body(request.question)
    logger.info(f"Processing follow-up for {conversation_id}: {sanitized_question[:50]}...")

    try:
        unified_service = get_unified_chat_service()

        # Process follow-up question using the working unified service
        response = await unified_service.continue_chat(
            question=sanitized_question,
            conversation_id=conversation_id
        )

        return {
            "response": response.response,
            "conversation_id": response.conversation_id,
            "sources": [source.model_dump() for source in response.sources],
            "processing_time": response.processing_time_seconds,
            "timestamp": response.timestamp.isoformat(),
            "external_research_used": response.external_research_used,
            "tools_called": [tool.model_dump() for tool in response.tools_called],
            "previous_decomposition_used": response.previous_decomposition_used
        }

    except Exception as e:
        logger.error(f"Failed to process follow-up question: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process follow-up question: {str(e)}"
        )


def _format_response_as_markdown(response) -> str:
    """Format enhanced response as markdown"""
    markdown = f"""# Legal Analysis

**Original Question:** {response.original_question}

## Final Answer
{response.final_answer}

## Decomposed Questions
"""

    for i, q in enumerate(response.decomposed_questions, 1):
        markdown += f"""
### {i}. {q.question}
{q.answer or "No answer available"}
"""

    if response.sources:
        markdown += "\n## Sources\n"
        for i, source in enumerate(response.sources[:5], 1):
            markdown += f"""
### {i}. {source.title}
- **Type:** {source.source_type.value}
- **Relevance:** {source.relevance_score:.3f}
"""
            if source.citation:
                markdown += f"- **Citation:** {source.citation}\n"
            if source.jurisdiction:
                markdown += f"- **Jurisdiction:** {source.jurisdiction}\n"
            if source.year:
                markdown += f"- **Year:** {source.year}\n"
            markdown += f"- **Content:** {source.content_preview}\n"

    if response.supports_followup and response.conversation_id:
        markdown += f"""
## Follow-up Questions
You can ask follow-up questions using conversation ID: `{response.conversation_id}`

Example: `POST /api/chat/followup/{response.conversation_id}?question=What are the specific penalties?`
"""

    markdown += f"""
---
*Processing time: {response.processing_time:.2f} seconds*
*Cache hit: {'Yes' if response.cache_hit else 'No'}*
"""

    return markdown


# Health check endpoint for chat service
@router.get("/health")
async def chat_health_check():
    """Health check for chat service"""
    try:
        unified_service = get_unified_chat_service()

        return {
            "status": "healthy",
            "unified_chat_service": "available",
            "research_tools": list(unified_service.research_manager.get_all_tools().keys())
        }
    except Exception as e:
        logger.error(f"Chat health check failed: {str(e)}")
        return {
            "status": "unhealthy",
            "error": str(e)
        }
"""
Fixed Chat API endpoints using the unified chat service
Eliminates the 'get' method bug and provides robust chat functionality
"""

import logging
from typing import Optional, Dict, Any

from fastapi import APIRouter, HTTPException, Query, Depends
from pydantic import BaseModel, Field

from app.models import (
    LegalQueryRequestWithChat, ChatMessage, LegalSource, ToolCallResult
)
from app.services.unified_chat_service import get_unified_chat_service
from app.utils.sanitizer import sanitize_legal_query_body
from app.auth import (
    require_chat_access,
    require_legal_research_access,
    get_user_context
)

logger = logging.getLogger("chat_api_fixed")
router = APIRouter(prefix="/api/chat", tags=["Chat"])


# Request models
class FollowupQuestionRequest(BaseModel):
    """Request model for follow-up questions"""
    question: str = Field(..., description="Follow-up legal question")


@router.post("/start", response_model=Dict[str, Any])
async def start_legal_chat_fixed(
    request: LegalQueryRequestWithChat,
    user_context: dict = Depends(get_user_context),
    _: None = Depends(require_chat_access),
    __: None = Depends(require_legal_research_access)
):
    """
    Start a new legal chat session using the unified chat service
    Fixed version that eliminates the 'get' method bug
    """
    # Sanitize the question from the request body
    sanitized_question = sanitize_legal_query_body(request.question)
    logger.info(f"Starting new chat session (fixed): {sanitized_question[:100]}...")

    try:
        chat_service = get_unified_chat_service()

        # Start chat session with unified service
        chat_response = await chat_service.start_chat(
            initial_question=sanitized_question,
            enable_decomposition=True
        )

        return {
            "success": True,
            "response": chat_response.response,
            "conversation_id": chat_response.conversation_id,
            "sources": [source.model_dump() for source in chat_response.sources],
            "supports_followup": True,
            "processing_time": chat_response.processing_time_seconds,
            "timestamp": chat_response.timestamp.isoformat(),
            "external_research_used": chat_response.external_research_used,
            "tools_called": [tool.model_dump() for tool in chat_response.tools_called],
            "previous_decomposition_used": chat_response.previous_decomposition_used,
            "message": "Chat session started successfully with unified service"
        }

    except Exception as e:
        logger.error(f"Failed to start chat session (fixed): {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to start chat session: {str(e)}"
        )


@router.post("/continue/{conversation_id}", response_model=Dict[str, Any])
async def continue_legal_chat_fixed(
    conversation_id: str,
    request: FollowupQuestionRequest
):
    """
    Continue an existing legal chat conversation using the unified chat service
    Fixed version that properly handles conversation context
    """
    # Sanitize the follow-up question
    sanitized_question = sanitize_legal_query_body(request.question)
    logger.info(f"Continuing chat {conversation_id} (fixed): {sanitized_question[:50]}...")

    try:
        chat_service = get_unified_chat_service()

        # Continue chat conversation with unified service
        chat_response = await chat_service.continue_chat(
            question=sanitized_question,
            conversation_id=conversation_id
        )

        return {
            "success": True,
            "response": chat_response.response,
            "conversation_id": chat_response.conversation_id,
            "sources": [source.model_dump() for source in chat_response.sources],
            "processing_time": chat_response.processing_time_seconds,
            "timestamp": chat_response.timestamp.isoformat(),
            "external_research_used": chat_response.external_research_used,
            "tools_called": [tool.model_dump() for tool in chat_response.tools_called],
            "previous_decomposition_used": chat_response.previous_decomposition_used,
            "message": f"Chat continued successfully for conversation {conversation_id}"
        }

    except Exception as e:
        logger.error(f"Failed to continue chat (fixed): {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to continue chat: {str(e)}"
        )


@router.get("/conversations/{conversation_id}/history")
async def get_conversation_history_fixed(conversation_id: str):
    """
    Get the complete conversation history using the unified chat service
    Fixed version with proper error handling
    """
    logger.info(f"Retrieving conversation history (fixed): {conversation_id}")

    try:
        chat_service = get_unified_chat_service()
        history = chat_service.get_conversation_history(conversation_id)

        if not history:
            raise HTTPException(
                status_code=404,
                detail="Conversation not found"
            )

        return {
            "success": True,
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
            "retrieved_at": chat_service.memory.conversation_metadata.get(
                conversation_id, {}
            ).get("last_updated"),
            "message": "Conversation history retrieved successfully"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get conversation history (fixed): {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve conversation history: {str(e)}"
        )


@router.delete("/conversations/{conversation_id}")
async def clear_conversation_fixed(conversation_id: str):
    """
    Clear/delete a conversation using the unified chat service
    Fixed version with proper cleanup
    """
    logger.info(f"Clearing conversation (fixed): {conversation_id}")

    try:
        chat_service = get_unified_chat_service()
        success = chat_service.clear_conversation(conversation_id)

        if not success:
            raise HTTPException(
                status_code=404,
                detail="Conversation not found"
            )

        return {
            "success": True,
            "message": f"Conversation {conversation_id} cleared successfully",
            "conversation_id": conversation_id,
            "cleared_at": chat_service.memory.conversation_metadata.get(
                conversation_id, {}
            ).get("last_updated")
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to clear conversation (fixed): {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to clear conversation: {str(e)}"
        )


@router.post("/ask-with-followup", response_model=Dict[str, Any])
async def ask_legal_question_with_followup_fixed(request: LegalQueryRequestWithChat):
    """
    Enhanced version of the original ask endpoint using unified chat service
    Fixed version that properly integrates decomposition with chat
    """
    # Sanitize the question from the request body
    sanitized_question = sanitize_legal_query_body(request.question)
    logger.info(f"Processing question with follow-up support (fixed): {sanitized_question[:100]}...")

    try:
        chat_service = get_unified_chat_service()

        # Process with unified chat service
        chat_response = await chat_service.start_chat(
            initial_question=sanitized_question,
            enable_decomposition=request.enable_followup
        )

        # Convert to the expected response format
        return {
            "success": True,
            "original_question": sanitized_question,
            "final_answer": chat_response.response,
            "sources": [source.model_dump() for source in chat_response.sources],
            "supports_followup": True,
            "conversation_id": chat_response.conversation_id,
            "processing_time": chat_response.processing_time_seconds,
            "previous_decomposition_used": chat_response.previous_decomposition_used,
            "message": "Question processed successfully with unified chat service"
        }

    except Exception as e:
        logger.error(f"Failed to process question with follow-up (fixed): {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process question: {str(e)}"
        )


@router.post("/followup/{conversation_id}", response_model=Dict[str, Any])
async def ask_followup_question_fixed(
    conversation_id: str,
    request: FollowupQuestionRequest
):
    """
    Ask a follow-up question using the unified chat service
    Fixed version that maintains proper conversation context
    """
    # Sanitize the follow-up question
    sanitized_question = sanitize_legal_query_body(request.question)
    logger.info(f"Processing follow-up for {conversation_id} (fixed): {sanitized_question[:50]}...")

    try:
        chat_service = get_unified_chat_service()

        # Process follow-up question with unified service
        chat_response = await chat_service.continue_chat(
            question=sanitized_question,
            conversation_id=conversation_id
        )

        return {
            "success": True,
            "response": chat_response.response,
            "conversation_id": chat_response.conversation_id,
            "sources": [source.model_dump() for source in chat_response.sources],
            "processing_time": chat_response.processing_time_seconds,
            "timestamp": chat_response.timestamp.isoformat(),
            "previous_decomposition_used": chat_response.previous_decomposition_used,
            "message": f"Follow-up question processed successfully for conversation {conversation_id}"
        }

    except Exception as e:
        logger.error(f"Failed to process follow-up question (fixed): {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process follow-up question: {str(e)}"
        )


# Health check endpoint for fixed chat service
@router.get("/health")
async def chat_health_check_fixed():
    """Health check for fixed chat service"""
    try:
        chat_service = get_unified_chat_service()

        return {
            "status": "healthy",
            "chat_service": "unified_chat_service",
            "service_version": "fixed_v1.0",
            "active_conversations": len(chat_service.memory.conversations),
            "research_tools": list(chat_service.research_manager.get_all_tools().keys()) if hasattr(chat_service, 'research_manager') else []
        }
    except Exception as e:
        logger.error(f"Chat health check failed (fixed): {str(e)}", exc_info=True)
        return {
            "status": "unhealthy",
            "error": str(e),
            "service_version": "fixed_v1.0"
        }


# Test endpoint for the fixed service
@router.post("/test")
async def test_chat_functionality_fixed(
    test_message: str = "What are the elements of criminal breach of trust?"
):
    """
    Test endpoint for the fixed chat functionality
    """
    try:
        chat_service = get_unified_chat_service()

        # Start a new chat session
        result = await chat_service.start_chat(
            initial_question=test_message,
            enable_decomposition=True
        )

        return {
            "test_status": "success",
            "service_version": "fixed_v1.0",
            "test_message": test_message,
            "response_preview": result.response[:200] + "..." if len(result.response) > 200 else result.response,
            "conversation_id": result.conversation_id,
            "sources_found": len(result.sources),
            "decomposition_used": result.previous_decomposition_used,
            "processing_time": result.processing_time_seconds,
            "timestamp": result.timestamp.isoformat(),
            "message": "Fixed chat service test completed successfully"
        }

    except Exception as e:
        return {
            "test_status": "failed",
            "service_version": "fixed_v1.0",
            "test_message": test_message,
            "error": str(e),
            "timestamp": None,
            "message": "Fixed chat service test failed"
        }
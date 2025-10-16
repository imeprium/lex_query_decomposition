"""
Enhanced pipeline service integrating chat functionality with existing decomposition pipeline.
Follows SOLID principles by extending rather than modifying existing functionality.
"""

import logging
import time
from typing import Dict, Any, Optional

from app.pipelines.legal_decomposition_pipeline import process_question
from app.models import (
    LegalQueryResponseWithChat, LegalSource, SourceFactory, SourceType
)
from app.services.legal_chat_service import get_legal_chat_service

logger = logging.getLogger("enhanced_pipeline_service")


class EnhancedLegalPipelineService:
    """
    Enhanced pipeline service that adds chat capabilities to the existing decomposition pipeline.
    This follows the Open/Closed Principle by extending functionality without modifying existing code.
    """

    def __init__(self):
        self.chat_service = get_legal_chat_service()
        logger.info("EnhancedLegalPipelineService initialized with chat capabilities")

    async def process_with_chat_support(
        self,
        question: str,
        enable_followup: bool = True,
        conversation_id: Optional[str] = None
    ) -> LegalQueryResponseWithChat:
        """
        Process a legal question with optional chat follow-up support.

        Args:
            question: The legal question to process
            enable_followup: Whether to enable chat follow-up capabilities
            conversation_id: Existing conversation ID for continuing chat

        Returns:
            Enhanced response with chat support information
        """
        start_time = time.time()
        logger.info(f"Processing question with chat support: {question[:100]}...")

        try:
            # Step 1: Process through existing decomposition pipeline
            decomp_result = await process_question(question)

            # Step 2: Create sources from decomposition results
            sources = self._create_sources_from_result(decomp_result)

            # Step 3: Handle conversation setup
            if enable_followup:
                if conversation_id:
                    # Use existing conversation
                    logger.info(f"Using existing conversation: {conversation_id}")
                else:
                    # Start new conversation and provide initial context
                    chat_response = await self.chat_service.start_chat(
                        initial_question=question,
                        enable_decomposition=False  # We already have decomposition
                    )
                    conversation_id = chat_response.conversation_id

                    # Store decomposition context for follow-up
                    await self._store_decomposition_context(
                        conversation_id, decomp_result, sources
                    )
            else:
                conversation_id = None

            # Step 4: Extract decomposed questions properly
            sub_questions_obj = decomp_result.get("sub_questions")
            decomposed_questions = []

            logger.info(f"Type of sub_questions_obj: {type(sub_questions_obj)}")

            # Handle different types of sub_questions objects
            if sub_questions_obj is None:
                logger.warning("sub_questions_obj is None")
                decomposed_questions = []
            elif isinstance(sub_questions_obj, tuple):
                # This is the problematic case - it's a tuple instead of Questions object
                logger.warning(f"sub_questions_obj is a tuple: {sub_questions_obj}")
                if len(sub_questions_obj) > 1 and isinstance(sub_questions_obj[1], list):
                    # Extract questions list from tuple
                    decomposed_questions = sub_questions_obj[1]
                    logger.info(f"Extracted {len(decomposed_questions)} questions from tuple")
                else:
                    decomposed_questions = []
            elif hasattr(sub_questions_obj, "questions"):
                # Normal case - it's a Questions object
                decomposed_questions = sub_questions_obj.questions
                logger.info(f"Extracted {len(decomposed_questions)} questions from Questions object")
            elif hasattr(sub_questions_obj, 'content') and hasattr(sub_questions_obj, 'role'):
                # It's a ChatMessage object - this shouldn't happen but handle it gracefully
                logger.warning(f"sub_questions_obj is a ChatMessage object - this is unexpected")
                decomposed_questions = []
            else:
                logger.warning(f"sub_questions_obj has unexpected type: {type(sub_questions_obj)}")
                decomposed_questions = []

            # Step 5: Create enhanced response
            processing_time = time.time() - start_time

            # Validate decomposed_questions before passing to Pydantic
            logger.info(f"About to create LegalQueryResponseWithChat with {len(decomposed_questions)} questions")
            logger.info(f"Type of decomposed_questions[0]: {type(decomposed_questions[0]) if decomposed_questions else 'None'}")

            try:
                enhanced_response = LegalQueryResponseWithChat(
                    original_question=question,
                    decomposed_questions=decomposed_questions,
                    final_answer=decomp_result.get("answer", ""),
                    document_metadata=decomp_result.get("document_metadata", []),
                    supports_followup=enable_followup,
                    conversation_id=conversation_id,
                    processing_time=processing_time,
                    cache_hit=decomp_result.get("cache_hit", False),
                    sources=sources
                )
                logger.info("Successfully created LegalQueryResponseWithChat")
            except Exception as model_error:
                logger.error(f"Pydantic validation error: {model_error}")
                logger.error(f"decomposed_questions details: {decomposed_questions}")
                # Try to create with empty questions as fallback
                enhanced_response = LegalQueryResponseWithChat(
                    original_question=question,
                    decomposed_questions=[],
                    final_answer=f"Error creating response: {str(model_error)}",
                    document_metadata=[],
                    supports_followup=False,
                    conversation_id=None,
                    processing_time=processing_time,
                    cache_hit=False,
                    sources=[]
                )

            logger.info(f"Enhanced processing completed in {processing_time:.2f}s")
            return enhanced_response

        except Exception as e:
            logger.error(f"Enhanced pipeline processing failed: {str(e)}")
            processing_time = time.time() - start_time

            # Return error response
            return LegalQueryResponseWithChat(
                original_question=question,
                decomposed_questions=[],
                final_answer=f"An error occurred while processing your question: {str(e)}",
                document_metadata=[],
                supports_followup=False,
                conversation_id=None,
                processing_time=processing_time,
                cache_hit=False,
                sources=[]
            )

    async def process_followup_question(
        self,
        question: str,
        conversation_id: str,
        original_decomposition_context: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Process a follow-up question using chat service.

        Args:
            question: The follow-up question
            conversation_id: Existing conversation ID
            original_decomposition_context: Context from original decomposition

        Returns:
            Chat response with sources
        """
        logger.info(f"Processing follow-up question for conversation {conversation_id}: {question[:50]}...")

        try:
            # Use chat service for follow-up
            chat_response = await self.chat_service.continue_chat(
                question=question,
                conversation_id=conversation_id,
                previous_decomposition_context=original_decomposition_context
            )

            return {
                "response": chat_response.response,
                "sources": [source.model_dump() for source in chat_response.sources],
                "conversation_id": chat_response.conversation_id,
                "external_research_used": chat_response.external_research_used,
                "tools_called": [tool.model_dump() for tool in chat_response.tools_called],
                "processing_time": chat_response.processing_time_seconds,
                "timestamp": chat_response.timestamp.isoformat()
            }

        except Exception as e:
            logger.error(f"Follow-up question processing failed: {str(e)}")
            return {
                "response": f"I apologize, but I encountered an error with your follow-up question: {str(e)}",
                "sources": [],
                "conversation_id": conversation_id,
                "external_research_used": False,
                "tools_called": [],
                "processing_time": 0,
                "timestamp": None,
                "error": str(e)
            }

    def _create_sources_from_result(self, decomp_result: Dict[str, Any]) -> list:
        """Create LegalSource objects from decomposition results"""
        sources = []

        # Add decomposed questions as sources
        sub_questions = decomp_result.get("sub_questions")
        questions_list = []

        # Handle different types of sub_questions objects
        if sub_questions is None:
            questions_list = []
        elif isinstance(sub_questions, tuple):
            # Problematic case - it's a tuple instead of Questions object
            logger.warning(f"sub_questions is a tuple in _create_sources_from_result: {type(sub_questions)}")
            if len(sub_questions) > 1 and isinstance(sub_questions[1], list):
                questions_list = sub_questions[1]
            else:
                questions_list = []
        elif hasattr(sub_questions, "questions"):
            # Normal case - it's a Questions object
            questions_list = sub_questions.questions
        elif hasattr(sub_questions, 'content') and hasattr(sub_questions, 'role'):
            # It's a ChatMessage object - this shouldn't happen but handle it gracefully
            logger.warning(f"sub_questions is a ChatMessage object in _create_sources_from_result - this is unexpected")
            questions_list = []
        else:
            logger.warning(f"sub_questions has unexpected type in _create_sources_from_result: {type(sub_questions)}")
            questions_list = []

        # Create sources from questions
        for idx, question in enumerate(questions_list):
            source = SourceFactory.from_decomposition_question(question, idx)
            sources.append(source)

        # Add document sources
        document_metadata = decomp_result.get("document_metadata", [])
        for doc_meta in document_metadata:
            source = SourceFactory.from_decomposition_result(doc_meta)
            sources.append(source)

        return sources

    async def _store_decomposition_context(
        self,
        conversation_id: str,
        decomp_result: Dict[str, Any],
        sources: list
    ):
        """Store decomposition context in conversation memory for follow-up reference"""
        try:
            # Create context that can be used for follow-up questions
            context = {
                "original_question": decomp_result.get("original_question", ""),
                "final_answer": decomp_result.get("answer", ""),
                "document_metadata": decomp_result.get("document_metadata", []),
                "sources": [source.model_dump() for source in sources],
                "timestamp": time.time()
            }

            # Store in conversation memory metadata
            if hasattr(self.chat_service.memory, 'conversation_metadata'):
                self.chat_service.memory.conversation_metadata[conversation_id].update({
                    "decomposition_context": context
                })

            logger.info(f"Stored decomposition context for conversation {conversation_id}")

        except Exception as e:
            logger.error(f"Failed to store decomposition context: {str(e)}")

    def get_conversation_history(self, conversation_id: str) -> list:
        """Get conversation history for a given conversation ID"""
        try:
            messages = self.chat_service.get_conversation_history(conversation_id)
            return [
                {
                    "role": msg.role,
                    "content": msg.content,
                    "timestamp": msg.timestamp.isoformat() if msg.timestamp else None
                }
                for msg in messages
            ]
        except Exception as e:
            logger.error(f"Failed to get conversation history: {str(e)}")
            return []

    def clear_conversation(self, conversation_id: str) -> bool:
        """Clear a conversation"""
        try:
            return self.chat_service.clear_conversation(conversation_id)
        except Exception as e:
            logger.error(f"Failed to clear conversation: {str(e)}")
            return False


# Singleton instance for dependency injection
_enhanced_pipeline_service = None

def get_enhanced_pipeline_service() -> EnhancedLegalPipelineService:
    """Get singleton instance of enhanced pipeline service"""
    global _enhanced_pipeline_service
    if _enhanced_pipeline_service is None:
        _enhanced_pipeline_service = EnhancedLegalPipelineService()
    return _enhanced_pipeline_service
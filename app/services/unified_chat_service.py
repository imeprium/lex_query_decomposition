"""
Unified Chat Service for Legal Query Decomposition
Fixes the 'get' method bug by eliminating type confusion and implementing clean data flow
"""

import asyncio
import logging
import uuid
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple, Union

from haystack.dataclasses import ChatMessage as HaystackChatMessage
from haystack_integrations.components.generators.cohere import CohereChatGenerator
from haystack.utils import Secret

from app.models import (
    LegalSource, LegalChatResponse, ToolCallResult, ChatMessage,
    SourceType, SourceFactory, ConfidenceLevel, Question, Questions
)
from app.services.legal_research_tools import get_legal_research_manager
from app.pipelines.legal_decomposition_pipeline import process_question
from app.config.settings import COHERE_API_KEY, COHERE_MODEL
from app.components.embedders import get_dense_embedder, get_sparse_embedder
from app.components.retrievers import get_hybrid_retriever
from app.document_store.store import get_document_store
from haystack_integrations.components.retrievers.qdrant import QdrantHybridRetriever
from app.config.settings import DEFAULT_TOP_K, DEFAULT_SCORE_THRESHOLD

logger = logging.getLogger("unified_chat_service")


class ConversationMemory:
    """Manages conversation state and history with robust data handling"""

    def __init__(self):
        self.conversations: Dict[str, List[ChatMessage]] = {}
        self.conversation_metadata: Dict[str, Dict[str, Any]] = {}

    def create_conversation(self) -> str:
        """Create a new conversation and return its ID"""
        conversation_id = str(uuid.uuid4())
        self.conversations[conversation_id] = []
        self.conversation_metadata[conversation_id] = {
            "created_at": datetime.now(),
            "message_count": 0,
            "last_updated": datetime.now(),
            "context_sources": [],
            "decomposition_context": None
        }
        logger.info(f"Created new conversation: {conversation_id}")
        return conversation_id

    def add_message(self, conversation_id: str, role: str, content: str, metadata: Dict = None):
        """Add a message to the conversation"""
        if conversation_id not in self.conversations:
            logger.warning(f"Conversation {conversation_id} not found")
            return

        message = ChatMessage(
            role=role,
            content=content,
            timestamp=datetime.now()
        )

        self.conversations[conversation_id].append(message)
        self.conversation_metadata[conversation_id]["last_updated"] = datetime.now()
        self.conversation_metadata[conversation_id]["message_count"] += 1

        if metadata:
            self.conversation_metadata[conversation_id].update(metadata)

    def get_conversation(self, conversation_id: str) -> List[ChatMessage]:
        """Get conversation history"""
        return self.conversations.get(conversation_id, [])

    def get_haystack_messages(self, conversation_id: str) -> List[HaystackChatMessage]:
        """Convert conversation to Haystack ChatMessage format"""
        chat_messages = self.get_conversation(conversation_id)
        haystack_messages = []

        for msg in chat_messages:
            if msg.role == "user":
                haystack_messages.append(HaystackChatMessage.from_user(msg.content))
            elif msg.role == "assistant":
                haystack_messages.append(HaystackChatMessage.from_assistant(msg.content))
            elif msg.role == "system":
                haystack_messages.append(HaystackChatMessage.from_system(msg.content))

        return haystack_messages

    def set_decomposition_context(self, conversation_id: str, context: Dict[str, Any]):
        """Store decomposition context for follow-up questions"""
        if conversation_id in self.conversation_metadata:
            self.conversation_metadata[conversation_id]["decomposition_context"] = context
            logger.info(f"Stored decomposition context for conversation {conversation_id}")

    def get_decomposition_context(self, conversation_id: str) -> Optional[Dict[str, Any]]:
        """Get stored decomposition context"""
        if conversation_id in self.conversation_metadata:
            return self.conversation_metadata[conversation_id].get("decomposition_context")
        return None

    def clear_conversation(self, conversation_id: str) -> bool:
        """Clear a conversation"""
        if conversation_id in self.conversations:
            del self.conversations[conversation_id]
            if conversation_id in self.conversation_metadata:
                del self.conversation_metadata[conversation_id]
            return True
        return False


class UnifiedChatService:
    """
    Unified chat service that eliminates type confusion and integrates cleanly with decomposition
    """

    def __init__(self):
        self.memory = ConversationMemory()
        self.research_manager = get_legal_research_manager()
        self.chat_generator = self._create_chat_generator()
        logger.info("UnifiedChatService initialized")

    def _create_chat_generator(self) -> CohereChatGenerator:
        """Create Cohere chat generator"""
        api_key = Secret.from_token(COHERE_API_KEY) if COHERE_API_KEY else None

        chat_generator = CohereChatGenerator(
            model=COHERE_MODEL,
            api_key=api_key
        )

        return chat_generator

    def _normalize_questions_data(self, questions_data: Any) -> List[Question]:
        """
        Normalize questions data to always return a list of Question objects
        This is the key fix for the 'get' method bug
        """
        if not questions_data:
            return []

        # Case 1: It's already a Questions object with a questions attribute
        if hasattr(questions_data, 'questions') and hasattr(questions_data.questions, '__iter__'):
            logger.debug("Normalizing Questions object")
            return questions_data.questions

        # Case 2: It's a list of Question objects
        elif isinstance(questions_data, list) and all(hasattr(q, 'question') for q in questions_data):
            logger.debug("Normalizing list of Question objects")
            return questions_data

        # Case 3: It's a list of dictionaries
        elif isinstance(questions_data, list) and all(isinstance(q, dict) for q in questions_data):
            logger.debug("Normalizing list of question dictionaries")
            questions = []
            for q_data in questions_data:
                if "question" in q_data:
                    questions.append(Question(
                        question=q_data.get("question", ""),
                        answer=q_data.get("answer")
                    ))
            return questions

        # Case 4: It's a dictionary with questions key
        elif isinstance(questions_data, dict) and "questions" in questions_data:
            logger.debug("Normalizing dictionary with questions key")
            return self._normalize_questions_data(questions_data["questions"])

        # Case 5: It's a single Question object
        elif hasattr(questions_data, 'question'):
            logger.debug("Normalizing single Question object")
            return [questions_data]

        # Case 6: It's a single dictionary
        elif isinstance(questions_data, dict) and "question" in questions_data:
            logger.debug("Normalizing single question dictionary")
            return [Question(
                question=questions_data.get("question", ""),
                answer=questions_data.get("answer")
            )]

        # Case 7: It's a tuple (problematic case from cache)
        elif isinstance(questions_data, tuple):
            logger.warning(f"Normalizing tuple data: {type(questions_data)}")
            if len(questions_data) > 1 and isinstance(questions_data[1], list):
                return self._normalize_questions_data(questions_data[1])
            else:
                return []

        # Case 8: Unknown type - try to extract questions
        else:
            logger.warning(f"Unknown questions data type: {type(questions_data)}")
            return []

    def _create_sources_from_normalized_data(
        self,
        questions: List[Question],
        document_metadata: List[Dict[str, Any]]
    ) -> List[LegalSource]:
        """Create unified source list from normalized questions and documents with proper relevance scoring"""
        sources = []

        # Add question sources (highest priority from decomposition)
        for idx, question in enumerate(questions):
            source = SourceFactory.from_decomposition_question(question, idx)
            # Set high relevance for decomposition sources
            source.relevance_score = 1.0 - (idx * 0.1)  # 1.0, 0.9, 0.8, etc.
            sources.append(source)

        # Add document sources with proper relevance scoring
        for doc_idx, doc_meta in enumerate(document_metadata):
            # Use score from metadata or calculate based on position
            base_score = doc_meta.get("score", 0.5)
            if isinstance(base_score, (int, float)):
                relevance_score = base_score
            else:
                # Fallback scoring based on position
                relevance_score = 0.8 - (doc_idx * 0.1)

            source = SourceFactory.from_decomposition_result(doc_meta)
            source.relevance_score = relevance_score
            sources.append(source)

        return sources

    async def start_chat(self, initial_question: str, enable_decomposition: bool = True) -> LegalChatResponse:
        """
        Start a new chat session with robust data handling
        """
        start_time = datetime.now()
        conversation_id = self.memory.create_conversation()

        try:
            # Add user message to conversation
            self.memory.add_message(conversation_id, "user", initial_question)

            sources = []
            final_answer = ""
            decomposition_used = False

            if enable_decomposition and self._should_use_decomposition(initial_question):
                logger.info(f"Using decomposition for initial question: {initial_question[:50]}...")

                # Use decomposition pipeline
                decomp_result = await process_question(initial_question)

                # CRITICAL FIX: Normalize the questions data properly
                questions = self._normalize_questions_data(decomp_result.get("sub_questions"))
                document_metadata = decomp_result.get("document_metadata", [])
                final_answer = decomp_result.get("answer", "No answer generated")

                # Create unified sources
                sources = self._create_sources_from_normalized_data(questions, document_metadata)

                # Store decomposition context for follow-up
                decomposition_context = {
                    "original_question": initial_question,
                    "questions": [{"question": q.question, "answer": q.answer} for q in questions],
                    "document_metadata": document_metadata,
                    "final_answer": final_answer,
                    "timestamp": datetime.now().isoformat()
                }
                self.memory.set_decomposition_context(conversation_id, decomposition_context)
                decomposition_used = True

                logger.info(f"Decomposition completed with {len(questions)} sub-questions")

            else:
                # Direct chat without decomposition - retrieve documents directly
                logger.info(f"Using direct chat for question: {initial_question[:50]}...")
                sources = await self._retrieve_documents_directly(initial_question)
                final_answer = await self._generate_direct_response(initial_question, sources)

            processing_time = (datetime.now() - start_time).total_seconds()

            # Add assistant response to conversation
            self.memory.add_message(
                conversation_id, "assistant", final_answer,
                {"sources_used": len(sources), "processing_time": processing_time}
            )

            return LegalChatResponse(
                response=final_answer,
                sources=sources,
                conversation_id=conversation_id,
                timestamp=datetime.now(),
                processing_time_seconds=processing_time,
                previous_decomposition_used=decomposition_used
            )

        except Exception as e:
            logger.error(f"Chat session failed: {str(e)}", exc_info=True)
            error_response = LegalChatResponse(
                response=f"I apologize, but I encountered an error: {str(e)}",
                sources=[],
                conversation_id=conversation_id,
                timestamp=datetime.now(),
                processing_time_seconds=(datetime.now() - start_time).total_seconds()
            )
            return error_response

    async def continue_chat(
        self,
        question: str,
        conversation_id: str
    ) -> LegalChatResponse:
        """
        Continue an existing chat conversation with proper context handling
        """
        start_time = datetime.now()

        if conversation_id not in self.memory.conversations:
            logger.warning(f"Conversation {conversation_id} not found, creating new one")
            return await self.start_chat(question)

        try:
            # Add user message
            self.memory.add_message(conversation_id, "user", question)

            # Get conversation history
            hay_messages = self.memory.get_haystack_messages(conversation_id)

            # Collect all sources
            all_sources = []

            # Step 1: Use stored decomposition context if available
            decomposition_context = self.memory.get_decomposition_context(conversation_id)
            if decomposition_context:
                questions = self._normalize_questions_data(decomposition_context.get("questions", []))
                document_metadata = decomposition_context.get("document_metadata", [])
                context_sources = self._create_sources_from_normalized_data(questions, document_metadata)
                all_sources.extend(context_sources)
                logger.info(f"Using decomposition context with {len(context_sources)} sources")

            # Step 2: Retrieve additional documents for the follow-up question
            additional_sources = await self._retrieve_documents_directly(question)
            all_sources.extend(additional_sources)

            # Step 3: Generate response with all sources
            final_answer, used_sources = await self._generate_contextual_response(
                question, all_sources, hay_messages
            )

            processing_time = (datetime.now() - start_time).total_seconds()

            # Add assistant response to conversation
            self.memory.add_message(
                conversation_id, "assistant", final_answer,
                {
                    "sources_used": len(used_sources),
                    "processing_time": processing_time,
                    "available_sources": len(all_sources)
                }
            )

            return LegalChatResponse(
                response=final_answer,
                sources=used_sources,  # Only return sources that were actually used
                conversation_id=conversation_id,
                timestamp=datetime.now(),
                processing_time_seconds=processing_time,
                previous_decomposition_used=bool(decomposition_context)
            )

        except Exception as e:
            logger.error(f"Continue chat failed: {str(e)}", exc_info=True)
            error_response = LegalChatResponse(
                response=f"I apologize, but I encountered an error: {str(e)}",
                sources=[],
                conversation_id=conversation_id,
                timestamp=datetime.now(),
                processing_time_seconds=(datetime.now() - start_time).total_seconds()
            )
            return error_response

    def _should_use_decomposition(self, question: str) -> bool:
        """Determine if question should be decomposed"""
        question_lower = question.lower()

        decomp_keywords = [
            "what constitutes", "elements of", "requirements for",
            "how does", "process for", "procedure to", "what are the",
            "legal framework", "comprehensive analysis", "detailed explanation"
        ]

        for keyword in decomp_keywords:
            if keyword in question_lower:
                return True

        if len(question) > 100:
            return True

        return False

    async def _retrieve_documents_directly(self, question: str) -> List[LegalSource]:
        """Retrieve documents directly without decomposition with proper relevance scoring"""
        try:
            logger.info(f"Retrieving documents for: {question[:50]}...")

            # Create document store and retriever
            document_store = get_document_store()
            qdrant_retriever = QdrantHybridRetriever(
                document_store=document_store,
                top_k=DEFAULT_TOP_K,
                score_threshold=DEFAULT_SCORE_THRESHOLD
            )
            hybrid_retriever = get_hybrid_retriever(qdrant_retriever)

            # Create a single-question wrapper
            single_question = Questions(questions=[Question(question=question)])

            # Get embedders
            dense_embedder = get_dense_embedder()
            sparse_embedder = get_sparse_embedder()

            # Generate embeddings
            dense_result = await dense_embedder.run_async(single_question)
            sparse_result = await sparse_embedder.run_async(single_question)

            # Perform hybrid retrieval
            retrieval_result = await hybrid_retriever.run_async(
                queries=single_question,
                dense_embeddings=dense_result["embeddings"],
                sparse_embeddings=sparse_result["sparse_embeddings"],
                top_k=DEFAULT_TOP_K
            )

            # Convert to LegalSource objects with proper relevance scoring
            sources = []
            question_context_pairs = retrieval_result.get("question_context_pairs", [])

            for pair in question_context_pairs:
                if "documents" in pair and pair["documents"]:
                    for doc_idx, doc in enumerate(pair["documents"]):
                        metadata = doc.get("metadata", {})
                        content = doc.get("content", "")

                        # Get document score from Haystack Document object
                        doc_score = getattr(doc, 'score', 0.5)
                        if doc_score is None:
                            doc_score = 0.8 - (doc_idx * 0.1)  # Fallback scoring

                        enhanced_metadata = metadata.copy()
                        enhanced_metadata.update({
                            "content_preview": content[:200] if content else "",
                            "retrieval_score": doc_score,
                            "original_score": doc_score
                        })

                        source = SourceFactory.from_decomposition_result(enhanced_metadata)
                        # Ensure the source has the proper relevance score
                        source.relevance_score = float(doc_score)
                        sources.append(source)

            logger.info(f"Retrieved {len(sources)} sources for: {question[:50]}...")
            return sources

        except Exception as e:
            logger.error(f"Document retrieval failed: {str(e)}", exc_info=True)
            return []

    async def _generate_direct_response(self, question: str, sources: List[LegalSource]) -> str:
        """Generate direct response using sources"""
        system_message = HaystackChatMessage.from_system(
            "You are an expert legal assistant specializing in Nigerian jurisprudence. "
            "Provide accurate, well-researched legal information with proper citations. "
            "Always specify when information is general guidance and recommend consulting with qualified legal counsel."
        )

        context_message = self._create_context_prompt(question, sources)
        user_message = HaystackChatMessage.from_user(context_message)

        messages = [system_message, user_message]
        result = self.chat_generator.run(messages=messages)

        # Handle different response formats from Cohere
        if hasattr(result, 'replies'):
            # Result is a ChatGenerator response object
            if result.replies and len(result.replies) > 0:
                return result.replies[0].text
        elif isinstance(result, dict):
            # Result is a dictionary
            if "replies" in result and result["replies"]:
                if isinstance(result["replies"], list) and len(result["replies"]) > 0:
                    reply = result["replies"][0]
                    if hasattr(reply, 'text'):
                        return reply.text
                    elif isinstance(reply, dict) and "text" in reply:
                        return reply["text"]

        return "I'm unable to provide a response at this time."

    async def _generate_contextual_response(
        self,
        question: str,
        sources: List[LegalSource],
        conversation_history: List[HaystackChatMessage]
    ) -> Tuple[str, List[LegalSource]]:
        """Generate response with conversation context and sources, returning both response and used sources"""

        # Sort sources by relevance and display priority
        sorted_sources = sorted(
            sources,
            key=lambda x: (x.display_priority, -x.relevance_score)
        )

        # Only use top relevant sources (relevance threshold)
        relevant_sources = [s for s in sorted_sources if s.relevance_score > -2.0][:5]

        if not relevant_sources:
            # No relevant sources found
            context_message = f"USER QUESTION: {question}\n\nNo relevant legal sources were found for this query. Please provide general legal guidance while acknowledging the limitation in available sources."
        else:
            context_message = self._create_context_prompt(question, relevant_sources)

        prompt_message = HaystackChatMessage.from_user(context_message)

        # Add system message if not in conversation history
        if not any(msg.role == "system" for msg in conversation_history):
            system_message = HaystackChatMessage.from_system(
                "You are an expert legal assistant specializing in Nigerian jurisprudence. "
                "Use the provided sources to give accurate, well-cited legal information. "
                "Always cite your sources using the format [Source X] and distinguish between different types of legal authorities."
            )
            conversation_history = [system_message] + conversation_history

        messages = conversation_history + [prompt_message]
        result = self.chat_generator.run(messages=messages)

        # Handle different response formats from Cohere
        if hasattr(result, 'replies'):
            # Result is a ChatGenerator response object
            if result.replies and len(result.replies) > 0:
                response_text = result.replies[0].text
            else:
                response_text = "I'm unable to provide a response at this time."
        elif isinstance(result, dict):
            # Result is a dictionary
            if "replies" in result and result["replies"]:
                if isinstance(result["replies"], list) and len(result["replies"]) > 0:
                    reply = result["replies"][0]
                    if hasattr(reply, 'text'):
                        response_text = reply.text
                    elif isinstance(reply, dict) and "text" in reply:
                        response_text = reply["text"]
                    else:
                        response_text = "I'm unable to provide a response at this time."
                else:
                    response_text = "I'm unable to provide a response at this time."
            else:
                response_text = "I'm unable to provide a response at this time."
        else:
            response_text = "I'm unable to provide a response at this time."

        return response_text, relevant_sources

    def _create_context_prompt(self, question: str, sources: List[LegalSource]) -> str:
        """Create context prompt with sources"""
        if not sources:
            return f"USER QUESTION: {question}\n\nPlease provide a comprehensive legal analysis of this question."

        sources_text = ""
        for idx, source in enumerate(sources):
            sources_text += f"[{idx+1}] {source.title}\n"
            if source.citation:
                sources_text += f"Citation: {source.citation}\n"
            if source.jurisdiction:
                sources_text += f"Jurisdiction: {source.jurisdiction}\n"
            if source.year:
                sources_text += f"Year: {source.year}\n"
            sources_text += f"Content: {source.content_preview}\n\n"

        return f"""USER QUESTION: {question}

LEGAL SOURCES:
{sources_text}

INSTRUCTIONS:
1. Provide a detailed legal analysis addressing the user's question
2. Use inline citations in the format [Source X] when referencing specific sources
3. Clearly distinguish between different types of sources (statutes, cases, regulations)
4. Highlight Nigerian legal authority when available
5. Acknowledge any limitations in the available legal information
6. Always conclude with a recommendation to consult qualified legal counsel for specific legal advice

ANSWER:"""

    def get_conversation_history(self, conversation_id: str) -> List[ChatMessage]:
        """Get conversation history"""
        return self.memory.get_conversation(conversation_id)

    def clear_conversation(self, conversation_id: str) -> bool:
        """Clear conversation history"""
        return self.memory.clear_conversation(conversation_id)


# Singleton instance
_unified_chat_service = None

def get_unified_chat_service() -> UnifiedChatService:
    """Get singleton instance of unified chat service"""
    global _unified_chat_service
    if _unified_chat_service is None:
        _unified_chat_service = UnifiedChatService()
    return _unified_chat_service
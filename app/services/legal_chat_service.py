"""
Legal query chat service implementing conversational AI with source tracking.
Follows SOLID principles with clear separation of concerns.
"""

import asyncio
import logging
import uuid
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple

from haystack import Pipeline
from haystack.dataclasses import ChatMessage as HaystackChatMessage
from haystack.components.builders import ChatPromptBuilder
from haystack_integrations.components.generators.cohere import CohereChatGenerator

from app.models import (
    LegalSource, LegalChatResponse, ToolCallResult, ChatMessage,
    SourceType, SourceFactory, ConfidenceLevel, Question
)
from app.services.legal_research_tools import get_legal_research_manager
from app.pipelines.legal_decomposition_pipeline import process_question
from app.config.settings import COHERE_MODEL

logger = logging.getLogger("legal_chat_service")


class ConversationMemory:
    """Manages conversation state and history following single responsibility principle"""

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
            "context_sources": []
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

    def clear_conversation(self, conversation_id: str) -> bool:
        """Clear a conversation"""
        if conversation_id in self.conversations:
            del self.conversations[conversation_id]
            if conversation_id in self.conversation_metadata:
                del self.conversation_metadata[conversation_id]
            return True
        return False


class LegalQueryChatService:
    """
    Main service for legal query chat functionality.
    Integrates decomposition pipeline with conversational AI and external research.
    """

    def __init__(self):
        self.memory = ConversationMemory()
        self.research_manager = get_legal_research_manager()
        self.chat_pipeline = self._create_chat_pipeline()
        logger.info("LegalQueryChatService initialized")

    def _create_chat_pipeline(self) -> CohereChatGenerator:
        """Create Cohere chat generator directly"""
        from haystack.utils import Secret

        # Get API key from settings
        from app.config.settings import COHERE_API_KEY

        api_key = Secret.from_token(COHERE_API_KEY) if COHERE_API_KEY else None

        # Use a cost-effective but capable model
        # command-r7b-12-2024 is newer and more cost-effective than command-r-plus
        chat_generator = CohereChatGenerator(
            model="command-r7b-12-2024",  # Newer, more capable, and cost-effective
            api_key=api_key
        )

        return chat_generator

    async def start_chat(self, initial_question: str, enable_decomposition: bool = True) -> LegalChatResponse:
        """
        Start a new chat session, optionally with initial decomposition
        """
        start_time = datetime.now()
        conversation_id = self.memory.create_conversation()

        try:
            # Step 1: Add user message to conversation
            self.memory.add_message(conversation_id, "user", initial_question)

            # Step 2: Decide whether to use decomposition first
            if enable_decomposition and self._should_use_decomposition(initial_question):
                logger.info(f"Using decomposition for initial question: {initial_question[:50]}...")

                # Use existing decomposition pipeline
                decomp_result = await process_question(initial_question)

                # Ensure decomp_result is a dictionary
                if not isinstance(decomp_result, dict):
                    logger.error(f"ERROR: decomp_result is not a dict, it's a {type(decomp_result)}")
                    # Create fallback dict structure
                    decomp_result = {
                        "answer": "Error processing question",
                        "sub_questions": None,
                        "document_metadata": []
                    }

                # Convert decomposition result to sources
                sources = self._create_sources_from_decomposition(decomp_result)

                # Create context-aware response
                response = await self._generate_response_from_decomposition(
                    initial_question, decomp_result, sources, conversation_id
                )

            else:
                # Direct chat without decomposition
                logger.info(f"Using direct chat for question: {initial_question[:50]}...")
                response = await self._direct_chat_response(
                    initial_question, conversation_id
                )
                sources = []

            processing_time = (datetime.now() - start_time).total_seconds()

            # Add assistant response to conversation
            self.memory.add_message(
                conversation_id, "assistant", response.response,
                {"sources_used": len(sources), "processing_time": processing_time}
            )

            response.processing_time_seconds = processing_time
            logger.info(f"Chat session {conversation_id} completed in {processing_time:.2f}s")

            return response

        except Exception as e:
            logger.error(f"Chat session failed: {str(e)}")
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
        conversation_id: str,
        previous_decomposition_context: Optional[Dict] = None
    ) -> LegalChatResponse:
        """
        Continue an existing chat conversation
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

            # Collect sources from various origins
            all_sources = []
            tool_calls = []

            # Step 1: Use previous decomposition context if available
            if previous_decomposition_context:
                decomp_sources = self._create_sources_from_decomposition_context(
                    previous_decomposition_context
                )
                all_sources.extend(decomp_sources)

            # Step 2: Retrieve relevant legal documents
            doc_sources = await self._retrieve_document_sources(question)
            all_sources.extend(doc_sources)

            # Step 3: Conduct external research if needed
            if self._needs_external_research(question, all_sources):
                external_sources, external_tools = await self._conduct_external_research(question)
                all_sources.extend(external_sources)
                tool_calls.extend(external_tools)

            # Step 4: Generate response with sources
            response = await self._generate_sourced_response(
                question, all_sources, hay_messages, conversation_id
            )

            # Add tool calls to response
            response.tools_called = tool_calls
            response.external_research_used = len(tool_calls) > 0
            response.processing_time_seconds = (datetime.now() - start_time).total_seconds()

            # Add assistant response to conversation
            self.memory.add_message(
                conversation_id, "assistant", response.response,
                {
                    "sources_used": len(all_sources),
                    "external_research": response.external_research_used,
                    "processing_time": response.processing_time_seconds
                }
            )

            logger.info(f"Chat response generated for {conversation_id} in {response.processing_time_seconds:.2f}s")
            return response

        except Exception as e:
            logger.error(f"Continue chat failed: {str(e)}")
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

        # Keywords that suggest complex legal questions needing decomposition
        decomp_keywords = [
            "what constitutes", "elements of", "requirements for",
            "how does", "process for", "procedure to", "what are the",
            "legal framework", "comprehensive analysis", "detailed explanation"
        ]

        # Check if any decomp keywords are present
        for keyword in decomp_keywords:
            if keyword in question_lower:
                return True

        # Also check question length - longer questions might need decomposition
        if len(question) > 100:
            return True

        return False

    def _create_sources_from_decomposition(self, decomp_result: Dict) -> List[LegalSource]:
        """Create sources from decomposition pipeline results"""
        sources = []

        logger.debug(f"DEBUG: decomp_result type: {type(decomp_result)}")
        logger.debug(f"DEBUG: decomp_result keys: {decomp_result.keys() if isinstance(decomp_result, dict) else 'Not a dict'}")

        # Handle sub_questions - decomp_result is a dict from pipeline
        if not isinstance(decomp_result, dict):
            logger.error(f"ERROR: Expected dict but got {type(decomp_result)}")
            return sources

        sub_questions_data = decomp_result.get("sub_questions")
        logger.debug(f"DEBUG: sub_questions_data type: {type(sub_questions_data)}")

        if sub_questions_data:
            # Check if sub_questions is a Questions object (has 'questions' attribute)
            if hasattr(sub_questions_data, 'questions'):
                logger.debug("DEBUG: sub_questions is a Questions object")
                questions_list = sub_questions_data.questions
            elif isinstance(sub_questions_data, dict):
                logger.debug("DEBUG: sub_questions is a dict")
                # If it's a dict, get the questions list
                questions_list = sub_questions_data.get("questions", [])
            elif hasattr(sub_questions_data, 'content') and hasattr(sub_questions_data, 'role'):
                # It's a ChatMessage object - this shouldn't happen but handle it gracefully
                logger.warning(f"DEBUG: sub_questions is a ChatMessage object - this is unexpected")
                questions_list = []
            else:
                logger.debug(f"DEBUG: sub_questions is type: {type(sub_questions_data)}")
                # If it's a list directly, use it as is
                questions_list = sub_questions_data if isinstance(sub_questions_data, list) else []

            logger.debug(f"DEBUG: questions_list type: {type(questions_list)}, length: {len(questions_list) if hasattr(questions_list, '__len__') else 'unknown'}")

            for idx, question in enumerate(questions_list):
                logger.debug(f"DEBUG: question {idx} type: {type(question)}")
                # Handle both Question objects and dict formats
                if hasattr(question, 'question'):
                    source = SourceFactory.from_decomposition_question(question, idx)
                    sources.append(source)
                elif isinstance(question, dict) and "question" in question:
                    logger.debug("DEBUG: Converting dict question to Question object")
                    # Convert dict to Question object
                    from app.models import Question
                    q_obj = Question(
                        question=question.get("question", ""),
                        answer=question.get("answer")
                    )
                    source = SourceFactory.from_decomposition_question(q_obj, idx)
                    sources.append(source)

        # Add document sources from decomposition
        document_metadata = decomp_result.get("document_metadata", [])
        logger.debug(f"DEBUG: document_metadata count: {len(document_metadata)}")

        for doc_meta in document_metadata:
            source = SourceFactory.from_decomposition_result(doc_meta)
            sources.append(source)

        logger.debug(f"DEBUG: Created {len(sources)} sources")
        return sources

    def _create_sources_from_decomposition_context(self, context: Dict) -> List[LegalSource]:
        """Create sources from previous decomposition context"""
        sources = []

        # Convert context to sources
        if "sub_questions" in context:
            sub_questions = context["sub_questions"]

            # Handle different types of sub_questions
            questions_list = []
            if hasattr(sub_questions, 'questions'):
                # It's a Questions object
                questions_list = sub_questions.questions
            elif isinstance(sub_questions, dict):
                # It's a dictionary
                questions_list = sub_questions.get("questions", [])
            elif isinstance(sub_questions, list):
                # It's already a list
                questions_list = sub_questions

            for idx, question_data in enumerate(questions_list):
                if isinstance(question_data, dict):
                    from app.models import Question
                    question = Question(
                        question=question_data.get("question", ""),
                        answer=question_data.get("answer")
                    )
                    source = SourceFactory.from_decomposition_question(question, idx)
                    sources.append(source)
                elif hasattr(question_data, 'question'):
                    # It's already a Question object
                    source = SourceFactory.from_decomposition_question(question_data, idx)
                    sources.append(source)

        if "document_metadata" in context:
            for doc_meta in context["document_metadata"]:
                source = SourceFactory.from_decomposition_result(doc_meta)
                sources.append(source)

        return sources

    async def _retrieve_document_sources(self, question: str) -> List[LegalSource]:
        """Retrieve relevant legal documents as sources using hybrid retrieval"""
        try:
            logger.info(f"Retrieving documents for: {question[:50]}...")

            # Import required components
            from app.components.embedders import get_dense_embedder, get_sparse_embedder
            from app.components.retrievers import get_hybrid_retriever
            from app.document_store.store import get_document_store
            from app.models import Questions, Question
            from haystack_integrations.components.retrievers.qdrant import QdrantHybridRetriever
            from app.config.settings import DEFAULT_TOP_K, DEFAULT_SCORE_THRESHOLD

            # Create document store and retriever
            document_store = get_document_store()
            qdrant_retriever = QdrantHybridRetriever(
                document_store=document_store,
                top_k=DEFAULT_TOP_K,
                score_threshold=DEFAULT_SCORE_THRESHOLD
            )
            hybrid_retriever = get_hybrid_retriever(qdrant_retriever)

            # Create a single-question wrapper for the retriever
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

            # Convert retrieval results to LegalSource objects
            sources = []
            question_context_pairs = retrieval_result.get("question_context_pairs", [])

            for pair in question_context_pairs:
                if "documents" in pair and pair["documents"]:
                    for doc_idx, doc in enumerate(pair["documents"]):
                        # Extract metadata from document
                        metadata = doc.get("metadata", {})
                        content = doc.get("content", "")

                        # Create LegalSource using existing SourceFactory method
                        # Enhance metadata with content and score
                        enhanced_metadata = metadata.copy()
                        enhanced_metadata.update({
                            "content_preview": content[:200] if content else "",
                            "retrieval_score": 1.0 - (doc_idx * 0.1)  # Simulate relevance score
                        })

                        source = SourceFactory.from_decomposition_result(enhanced_metadata)
                        sources.append(source)

            logger.info(f"Retrieved {len(sources)} sources for: {question[:50]}...")
            return sources

        except Exception as e:
            logger.error(f"Document retrieval failed: {str(e)}", exc_info=True)
            return []

    def _needs_external_research(self, question: str, existing_sources: List[LegalSource]) -> bool:
        """Determine if external research is needed"""
        if not existing_sources:
            return True

        # If question asks for specific legal information not in existing sources
        question_lower = question.lower()
        research_keywords = [
            "latest", "recent", "current", "up-to-date", "precedent",
            "case law", "statute", "regulation", "specific section"
        ]

        return any(keyword in question_lower for keyword in research_keywords)

    async def _conduct_external_research(self, question: str) -> Tuple[List[LegalSource], List[ToolCallResult]]:
        """Conduct external legal research"""
        all_sources = []
        all_tool_calls = []

        # Get relevant tools for the query
        relevant_tools = self.research_manager.get_tools_for_query(question)

        # Execute tools in parallel
        tasks = []
        for tool in relevant_tools:
            # Prepare tool arguments based on query
            args = self._prepare_tool_arguments(tool.name, question)
            if args:
                task = tool.call(**args)
                tasks.append(task)

        if tasks:
            tool_results = await asyncio.gather(*tasks, return_exceptions=True)

            for result in tool_results:
                if isinstance(result, ToolCallResult):
                    all_tool_calls.append(result)
                    all_sources.extend(result.sources_generated)
                elif isinstance(result, Exception):
                    logger.error(f"Tool execution failed: {str(result)}")

        return all_sources, all_tool_calls

    def _prepare_tool_arguments(self, tool_name: str, question: str) -> Dict[str, Any]:
        """Prepare arguments for tool execution"""
        base_args = {"query": question} if tool_name == "search_nigerian_statutes" else {}

        if tool_name == "search_case_precedents":
            base_args = {"legal_issue": question}
        elif tool_name == "search_regulations":
            base_args = {"industry": question}  # This is simplified

        return base_args

    async def _generate_response_from_decomposition(
        self,
        question: str,
        decomp_result: Dict,
        sources: List[LegalSource],
        conversation_id: str
    ) -> LegalChatResponse:
        """Generate response using decomposition results"""

        # Use the final answer from decomposition
        final_answer = decomp_result.get("answer", "No answer generated")

        # Format response with source citations
        formatted_response = self._format_response_with_sources(final_answer, sources)

        return LegalChatResponse(
            response=formatted_response,
            sources=sources,
            conversation_id=conversation_id,
            timestamp=datetime.now(),
            previous_decomposition_used=True
        )

    async def _direct_chat_response(self, question: str, conversation_id: str) -> LegalChatResponse:
        """Generate direct chat response without decomposition"""

        # Create system message
        system_message = HaystackChatMessage.from_system(
            "You are an expert legal assistant specializing in Nigerian jurisprudence. "
            "Provide accurate, well-researched legal information with proper citations. "
            "Always specify when information is general guidance and recommend consulting with qualified legal counsel."
        )

        # Create user message
        user_message = HaystackChatMessage.from_user(question)

        # Generate response - chat_pipeline is now a CohereChatGenerator directly
        messages = [system_message, user_message]
        result = self.chat_pipeline.run(messages=messages)

        # Handle both dictionary and object response formats
        if isinstance(result, dict):
            response_text = result.get("replies", [{}])[0].get("text", "I'm unable to provide a response.") if result.get("replies") else "I'm unable to provide a response."
        else:
            response_text = result.replies[0].text if result.replies else "I'm unable to provide a response."

        return LegalChatResponse(
            response=response_text,
            sources=[],  # No sources for direct chat
            conversation_id=conversation_id,
            timestamp=datetime.now(),
            previous_decomposition_used=False
        )

    async def _generate_sourced_response(
        self,
        question: str,
        sources: List[LegalSource],
        conversation_history: List[HaystackChatMessage],
        conversation_id: str
    ) -> LegalChatResponse:
        """Generate response with comprehensive source citations"""

        # Sort sources by relevance and display priority
        sorted_sources = sorted(
            sources,
            key=lambda x: (x.display_priority, -x.relevance_score)
        )

        # Create a comprehensive prompt with sources
        sources_text = ""
        for idx, source in enumerate(sorted_sources[:5]):  # Limit to top 5 sources
            sources_text += f"[{idx+1}] {source.title}\n"
            if source.citation:
                sources_text += f"Citation: {source.citation}\n"
            if source.jurisdiction:
                sources_text += f"Jurisdiction: {source.jurisdiction}\n"
            if source.year:
                sources_text += f"Year: {source.year}\n"
            sources_text += f"Content: {source.content_preview}\n\n"

        prompt = f"""Based on the legal sources below, please provide a comprehensive answer to the user's question.

USER QUESTION: {question}

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

        # Add system message if not in conversation history
        if not any(msg.role == "system" for msg in conversation_history):
            system_message = HaystackChatMessage.from_system(
                "You are an expert legal assistant specializing in Nigerian jurisprudence. "
                "Use the provided sources to give accurate, well-cited legal information."
            )
            conversation_history = [system_message] + conversation_history

        # Add the prompt as user message
        prompt_message = HaystackChatMessage.from_user(prompt)
        messages = conversation_history + [prompt_message]

        # Generate response using the direct generator
        result = self.chat_pipeline.run(messages=messages)

        # Handle both dictionary and object response formats
        if isinstance(result, dict):
            response_text = result.get("replies", [{}])[0].get("text", "I'm unable to provide a response.") if result.get("replies") else "I'm unable to provide a response."
        else:
            response_text = result.replies[0].text if result.replies else "I'm unable to provide a response."

        return LegalChatResponse(
            response=response_text,
            sources=sorted_sources,
            conversation_id=conversation_id,
            timestamp=datetime.now()
        )

    def _create_source_context(self, sources: List[LegalSource]) -> str:
        """Create formatted context from sources"""
        context_parts = []

        for idx, source in enumerate(sources[:5]):  # Limit to top 5
            context_part = f"[{idx+1}] {source.title}"
            if source.citation:
                context_part += f" ({source.citation})"
            if source.jurisdiction:
                context_part += f" - {source.jurisdiction}"
            context_part += f"\n{source.content_preview}\n"
            context_parts.append(context_part)

        return "\n".join(context_parts)

    def _format_response_with_sources(self, response: str, sources: List[LegalSource]) -> str:
        """Format response with source citations"""
        if not sources:
            return response

        # Add source citations to the response
        formatted_response = response

        # Add source list at the end
        if len(sources) > 0:
            formatted_response += "\n\n**Sources:**\n"
            for idx, source in enumerate(sources[:3]):  # Top 3 sources
                source_ref = f"[{idx+1}] {source.title}"
                if source.citation:
                    source_ref += f" ({source.citation})"
                formatted_response += f"\n{source_ref}"

        return formatted_response

    def get_conversation_history(self, conversation_id: str) -> List[ChatMessage]:
        """Get conversation history"""
        return self.memory.get_conversation(conversation_id)

    def clear_conversation(self, conversation_id: str) -> bool:
        """Clear conversation history"""
        return self.memory.clear_conversation(conversation_id)


# Singleton instance
_legal_chat_service = None

def get_legal_chat_service() -> LegalQueryChatService:
    """Get singleton instance of legal chat service"""
    global _legal_chat_service
    if _legal_chat_service is None:
        _legal_chat_service = LegalQueryChatService()
    return _legal_chat_service
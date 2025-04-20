import logging
import threading
import time
import hashlib
from typing import Optional, Dict, Any
from pydantic import BaseModel
from haystack import AsyncPipeline
from haystack.utils import Secret
from haystack.components.builders import PromptBuilder
from haystack_integrations.components.retrievers.qdrant import QdrantHybridRetriever
from app.components.custom_generators import ExtendedCohereGenerator
from app.components.embedders import MultiQueryDenseEmbedder, MultiQuerySparseEmbedder
from app.components.retrievers import MultiQueryHybridRetriever
from app.components.decomposition_validator import DecompositionValidator
from app.document_store.store import get_document_store
from app.models import Questions, Question, DocumentMetadata
from app.prompts.decomposition import LEGAL_QUERY_DECOMPOSITION_PROMPT
from app.prompts.answering import LEGAL_MULTI_QUERY_TEMPLATE
from app.prompts.reasoning import LEGAL_REASONING_TEMPLATE
from app.config.settings import COHERE_MODEL, COHERE_API_KEY, DEFAULT_TOP_K, DEFAULT_SCORE_THRESHOLD
from app.utils.cache import get_cached_result, cache_result

logger = logging.getLogger("pipeline")


class LegalDecompositionPipelineService:
    """
    Singleton service for legal decomposition pipeline
    """
    _instance = None
    _lock = threading.RLock()
    _initialized = False

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        # Only initialize once
        with self._lock:
            if not self._initialized:
                self._initialize()
                LegalDecompositionPipelineService._initialized = True

    def _initialize(self):
        """Initialize the legal decomposition pipeline"""
        logger.info("Creating legal decomposition pipeline")

        # Get document store
        document_store = get_document_store()

        # Create pipeline (async for better performance)
        self.pipeline = AsyncPipeline()

        # Create base retriever
        self.qdrant_retriever = QdrantHybridRetriever(
            document_store=document_store,
            top_k=DEFAULT_TOP_K,
            score_threshold=DEFAULT_SCORE_THRESHOLD
        )

        # Create API key secret
        cohere_api_key_secret = Secret.from_token(COHERE_API_KEY) if COHERE_API_KEY else None

        # Create components
        self.pipeline.add_component(
            "prompt",
            PromptBuilder(template=LEGAL_QUERY_DECOMPOSITION_PROMPT)
        )

        self.pipeline.add_component(
            "decomposer",
            ExtendedCohereGenerator(
                model=COHERE_MODEL,
                model_type=Questions,
                api_key=cohere_api_key_secret
            )
        )

        # Add validator to handle empty decomposition results
        self.pipeline.add_component(
            "validator",
            DecompositionValidator()
        )

        # Embedders
        self.pipeline.add_component(
            "dense_embedder",
            MultiQueryDenseEmbedder()
        )

        self.pipeline.add_component(
            "sparse_embedder",
            MultiQuerySparseEmbedder()
        )

        # Retriever with built-in ranker
        self.pipeline.add_component(
            "hybrid_retriever",
            MultiQueryHybridRetriever(retriever=self.qdrant_retriever)
        )

        # Prompt builder for multi-query results
        self.pipeline.add_component(
            "multi_query_prompt",
            PromptBuilder(template=LEGAL_MULTI_QUERY_TEMPLATE)
        )

        # Query resolver (processes context and provides answers to sub-questions)
        self.pipeline.add_component(
            "query_resolver",
            ExtendedCohereGenerator(
                model=COHERE_MODEL,
                model_type=Questions,
                api_key=cohere_api_key_secret
            )
        )

        # Final reasoning components
        self.pipeline.add_component(
            "reasoning_prompt",
            PromptBuilder(template=LEGAL_REASONING_TEMPLATE)
        )

        self.pipeline.add_component(
            "reasoning_llm",
            ExtendedCohereGenerator(
                model=COHERE_MODEL,
                api_key=cohere_api_key_secret
            )
        )

        # Connect components - only connect outputs to inputs
        self.pipeline.connect("prompt", "decomposer")
        self.pipeline.connect("decomposer.structured_reply", "validator.questions")
        self.pipeline.connect("validator.valid_questions", "dense_embedder.questions")
        self.pipeline.connect("validator.valid_questions", "sparse_embedder.questions")
        self.pipeline.connect("validator.valid_questions", "hybrid_retriever.queries")
        self.pipeline.connect("dense_embedder.embeddings", "hybrid_retriever.dense_embeddings")
        self.pipeline.connect("sparse_embedder.sparse_embeddings", "hybrid_retriever.sparse_embeddings")
        self.pipeline.connect("hybrid_retriever.question_context_pairs", "multi_query_prompt.question_context_pairs")
        self.pipeline.connect("multi_query_prompt", "query_resolver")
        self.pipeline.connect("query_resolver.structured_reply", "reasoning_prompt.question_answer_pair")
        self.pipeline.connect("reasoning_prompt", "reasoning_llm")

        logger.info("Legal decomposition pipeline created successfully")

    def get_pipeline(self):
        """Get the pipeline instance"""
        return self.pipeline

    def _generate_cache_key(self, question: str) -> str:
        """
        Generate a cache key for a question

        Args:
            question: The original question

        Returns:
            A hash-based cache key
        """
        # Create a consistent hash of the question for cache key
        # Using md5 for speed (not for security)
        return hashlib.md5(question.lower().strip().encode()).hexdigest()

    async def run_pipeline(self, question: str, **kwargs):
        """
        Run the pipeline with proper error handling and metadata extraction

        Args:
            question: The user's legal question
            **kwargs: Additional arguments for pipeline components

        Returns:
            Dictionary with answer, sub-questions, and document metadata
        """
        try:
            start_time = time.time()
            logger.info(f"Processing question: '{question}'")

            # Generate cache key
            cache_key = self._generate_cache_key(question)

            # Check cache before processing
            cached_result = await get_cached_result(cache_key)
            if cached_result:
                logger.info(f"Using cached result for question: '{question[:50]}...'")
                # Add processing time information
                cached_result["processing_time"] = 0.0  # Negligible processing time for cache hit
                cached_result["cache_hit"] = True
                return cached_result

            # Cache miss - continue with normal processing
            logger.info(f"Cache miss, running pipeline for question: '{question[:50]}...'")

            # Set up inputs for the pipeline
            inputs = {
                "prompt": {"question": question},
                "validator": {"original_question": question},
                "multi_query_prompt": {"question": question},
                "reasoning_prompt": {"question": question},
                **kwargs
            }

            # Run the pipeline asynchronously and include outputs from key components
            results = await self.pipeline.run_async(
                inputs,
                include_outputs_from=["decomposer", "validator", "query_resolver",
                                      "reasoning_llm", "hybrid_retriever"]
            )

            # 1. Extract final answer
            final_answer = "No answer generated"
            if "reasoning_llm" in results and "replies" in results["reasoning_llm"]:
                replies = results["reasoning_llm"]["replies"]
                if replies and len(replies) > 0:
                    final_answer = replies[0]

            # 2. Extract sub-questions (prefer resolved questions if available)
            sub_questions = None
            if "query_resolver" in results and "structured_reply" in results["query_resolver"]:
                sub_questions = results["query_resolver"]["structured_reply"]
            elif "validator" in results and "valid_questions" in results["validator"]:
                sub_questions = results["validator"]["valid_questions"]

            # 3. Extract document metadata
            document_metadata = []
            if "hybrid_retriever" in results and "question_context_pairs" in results["hybrid_retriever"]:
                for pair in results["hybrid_retriever"]["question_context_pairs"]:
                    question_text = pair.get("question", "")

                    # Process documents for this question
                    if "documents" in pair and pair["documents"]:
                        # Get top documents (each now has content and metadata)
                        formatted_docs = pair["documents"]

                        # Process each document (limit to top 3 per question for display purposes)
                        for doc_idx, formatted_doc in enumerate(formatted_docs[:3]):
                            # Extract metadata from the formatted document
                            doc_metadata = formatted_doc.get("metadata", {})

                            # Create metadata entry with required fields
                            metadata_entry = {
                                "id": f"doc-{len(document_metadata) + 1}",
                                "score": 1.0 - (doc_idx * 0.1),  # Simulate relevance score
                                "document_id": doc_metadata.get("document_id", f"unknown-{len(document_metadata) + 1}")
                            }

                            # Add title field (case_title, article_title, or legislation_title)
                            for title_field in ["case_title", "article_title", "legislation_title"]:
                                if title_field in doc_metadata:
                                    metadata_entry[title_field] = doc_metadata[title_field]
                                    break

                            # Ensure there's at least a default title if none found
                            if not any(key in metadata_entry for key in
                                       ["case_title", "article_title", "legislation_title"]):
                                metadata_entry[
                                    "case_title"] = f"Result for: {question_text[:30]}..." if question_text else "Unknown document"

                            document_metadata.append(metadata_entry)

            # Calculate processing time
            processing_time = time.time() - start_time
            logger.info(f"Pipeline processing completed in {processing_time:.2f} seconds")

            # Create the final result
            pipeline_result = {
                "answer": final_answer,
                "sub_questions": sub_questions,
                "document_metadata": document_metadata,
                "processing_time": processing_time,
                "cache_hit": False
            }

            # Store result in cache for future use
            await cache_result(cache_key, pipeline_result)

            return pipeline_result

        except Exception as e:
            logger.error(f"Error running pipeline: {str(e)}", exc_info=True)
            return {
                "answer": f"An error occurred while processing your question: {str(e)}",
                "error": str(e),
                "document_metadata": [],
                "cache_hit": False
            }


# Factory function to get singleton pipeline
def get_decomposition_pipeline():
    """Get singleton pipeline instance"""
    pipeline_service = LegalDecompositionPipelineService()
    return pipeline_service.get_pipeline()


# Function to run the pipeline with a question
async def process_question(question: str):
    """
    Process a question through the decomposition pipeline

    Args:
        question: The user's question

    Returns:
        The final answer and any intermediate outputs
    """
    pipeline_service = LegalDecompositionPipelineService()
    return await pipeline_service.run_pipeline(question)
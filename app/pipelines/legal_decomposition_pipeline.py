import logging
import time
import hashlib
import asyncio
from typing import Optional, Dict, Any, Set
from functools import lru_cache
from haystack import AsyncPipeline
from haystack.utils import Secret
from haystack.components.builders import PromptBuilder
from haystack_integrations.components.retrievers.qdrant import QdrantHybridRetriever
from app.components.custom_generators import ExtendedCohereGenerator
from app.components.embedders import get_dense_embedder, get_sparse_embedder
from app.components.retrievers import get_hybrid_retriever
from app.components.decomposition_validator import DecompositionValidator
from app.document_store.store import get_document_store
from app.models import Questions, Question, DocumentMetadata
from app.prompts.decomposition import LEGAL_QUERY_DECOMPOSITION_PROMPT
from app.prompts.answering import LEGAL_MULTI_QUERY_TEMPLATE
from app.prompts.reasoning import LEGAL_REASONING_TEMPLATE
from app.config.settings import COHERE_MODEL, COHERE_API_KEY, DEFAULT_TOP_K, DEFAULT_SCORE_THRESHOLD
from app.utils.cache import get_cached_result, cache_result
from app.core.singleton import SingletonMeta

logger = logging.getLogger("pipeline")


class PipelineBuilder:
    """
    Builder for creating pipelines with a fluent API.
    """

    def __init__(self):
        self.pipeline = AsyncPipeline()

    def add_prompt(self, name: str, template: str, required_variables: list = None) -> 'PipelineBuilder':
        """Add a prompt builder component"""
        self.pipeline.add_component(
            name,
            PromptBuilder(template=template, required_variables=required_variables or [])
        )
        return self

    def add_generator(self, name: str, model_type=None) -> 'PipelineBuilder':
        """Add a generator component"""
        # Create API key secret
        api_key = Secret.from_token(COHERE_API_KEY) if COHERE_API_KEY else None

        self.pipeline.add_component(
            name,
            ExtendedCohereGenerator(
                model=COHERE_MODEL,
                model_type=model_type,
                api_key=api_key
            )
        )
        return self

    def add_embedders(self) -> 'PipelineBuilder':
        """Add embedder components"""
        self.pipeline.add_component("dense_embedder", get_dense_embedder())
        self.pipeline.add_component("sparse_embedder", get_sparse_embedder())
        return self

    def add_validator(self) -> 'PipelineBuilder':
        """Add decomposition validator"""
        self.pipeline.add_component("validator", DecompositionValidator())
        return self

    def add_retriever(self) -> 'PipelineBuilder':
        """Add hybrid retriever"""
        document_store = get_document_store()
        qdrant_retriever = QdrantHybridRetriever(
            document_store=document_store,
            top_k=DEFAULT_TOP_K,
            score_threshold=DEFAULT_SCORE_THRESHOLD
        )

        self.pipeline.add_component(
            "hybrid_retriever",
            get_hybrid_retriever(qdrant_retriever)
        )
        return self

    def connect_components(self) -> 'PipelineBuilder':
        """Connect all pipeline components"""
        # Decomposition path
        self.pipeline.connect("prompt", "decomposer")
        self.pipeline.connect("decomposer.structured_reply", "validator.questions")

        # Embedding and retrieval path
        self.pipeline.connect("validator.valid_questions", "dense_embedder.questions")
        self.pipeline.connect("validator.valid_questions", "sparse_embedder.questions")
        self.pipeline.connect("validator.valid_questions", "hybrid_retriever.queries")
        self.pipeline.connect("dense_embedder.embeddings", "hybrid_retriever.dense_embeddings")
        self.pipeline.connect("sparse_embedder.sparse_embeddings", "hybrid_retriever.sparse_embeddings")

        # Answer generation path
        self.pipeline.connect("hybrid_retriever.question_context_pairs", "multi_query_prompt.question_context_pairs")
        self.pipeline.connect("multi_query_prompt", "query_resolver")

        # Final reasoning path
        self.pipeline.connect("query_resolver.structured_reply", "reasoning_prompt.question_answer_pair")
        self.pipeline.connect("reasoning_prompt", "reasoning_llm")

        return self

    def build(self) -> AsyncPipeline:
        """Get the constructed pipeline"""
        return self.pipeline


class LegalDecompositionPipelineService(metaclass=SingletonMeta):
    """
    Service for managing and executing the legal decomposition pipeline.
    """

    def __init__(self):
        self._initialize_pipeline()

    def _initialize_pipeline(self):
        """Initialize the legal decomposition pipeline"""
        logger.info("Creating legal decomposition pipeline")

        # Create pipeline using builder
        builder = PipelineBuilder()
        self.pipeline = (builder
                         .add_prompt("prompt", LEGAL_QUERY_DECOMPOSITION_PROMPT, required_variables=["question"])
                         .add_generator("decomposer", Questions)
                         .add_validator()
                         .add_embedders()
                         .add_retriever()
                         .add_prompt("multi_query_prompt", LEGAL_MULTI_QUERY_TEMPLATE, required_variables=["question", "question_context_pairs"])
                         .add_generator("query_resolver", Questions)
                         .add_prompt("reasoning_prompt", LEGAL_REASONING_TEMPLATE, required_variables=["question", "question_answer_pair"])
                         .add_generator("reasoning_llm")
                         .connect_components()
                         .build())

        logger.info("Legal decomposition pipeline created successfully")

    def get_pipeline(self):
        """Get the pipeline instance"""
        return self.pipeline

    def _generate_cache_key(self, question: str) -> str:
        """Generate a cache key for a question"""
        return hashlib.md5(question.lower().strip().encode()).hexdigest()

    #@cached(key_prefix="legal_decomposition")
    async def run_pipeline(self, question: str, **kwargs) -> Dict[str, Any]:
        """
        Run the pipeline with proper error handling and metadata extraction
        """
        logger.info(f"Processing question: '{question}'")
        start_time = time.time()

        # Generate cache key
        cache_key = self._generate_cache_key(question)

        # Check cache before processing
        cached_result = await get_cached_result(cache_key)
        if cached_result:
            logger.info(f"Using cached result for question: '{question[:50]}...'")
            # Add processing time information
            cached_result["processing_time"] = 0.0
            return cached_result

        logger.info(f"Cache miss - processing question through pipeline: '{question[:50]}...'")

        try:
            # Set up inputs for the pipeline
            inputs = {
                "prompt": {"question": question},
                "validator": {"original_question": question},
                "multi_query_prompt": {"question": question},
                "reasoning_prompt": {"question": question},
                **kwargs
            }

            # Key components to include in results
            output_components = [
                "decomposer", "validator", "query_resolver",
                "reasoning_llm", "hybrid_retriever"
            ]

            # Run the pipeline asynchronously
            results = await self.pipeline.run_async(
                inputs,
                include_outputs_from=output_components
            )

            # Process results
            pipeline_result = self._process_pipeline_results(results, question)

            # Add processing time
            processing_time = time.time() - start_time
            logger.info(f"Pipeline processing completed in {processing_time:.2f} seconds")
            pipeline_result["processing_time"] = processing_time
            pipeline_result["cache_hit"] = False

            # CRITICAL: Store result in cache for future use
            cache_success = await cache_result(cache_key, pipeline_result)
            if cache_success:
                logger.info(f"Successfully cached result for: {question[:50]}...")
            else:
                logger.warning(f"Failed to cache result for: {question[:50]}...")

            return pipeline_result

        except Exception as e:
            logger.error(f"Error running pipeline: {str(e)}", exc_info=True)
            error_result = {
                "answer": f"An error occurred while processing your question: {str(e)}",
                "error": str(e),
                "document_metadata": [],
                "processing_time": time.time() - start_time,
                "cache_hit": False
            }
            return error_result

    def _process_pipeline_results(self, results: Dict[str, Any], question: str) -> Dict[str, Any]:
        """Process pipeline results into a standardized format"""
        # 1. Extract final answer
        final_answer = "No answer generated"
        if "reasoning_llm" in results and "replies" in results["reasoning_llm"]:
            replies = results["reasoning_llm"]["replies"]
            if replies and len(replies) > 0:
                final_answer = replies[0]

        # 2. Extract sub-questions with proper type handling
        sub_questions = None
        if "query_resolver" in results and "structured_reply" in results["query_resolver"]:
            sub_questions_raw = results["query_resolver"]["structured_reply"]
            # Handle different types of structured_reply
            if hasattr(sub_questions_raw, 'questions'):
                # It's a Questions object (expected case)
                sub_questions = sub_questions_raw
            elif isinstance(sub_questions_raw, dict) and "questions" in sub_questions_raw:
                # It's a dictionary representation
                from app.models import Questions
                sub_questions = Questions.model_validate(sub_questions_raw)
            elif isinstance(sub_questions_raw, list):
                # It's a list of question dictionaries
                from app.models import Questions, Question
                questions_list = []
                for q_data in sub_questions_raw:
                    if isinstance(q_data, dict):
                        questions_list.append(Question.model_validate(q_data))
                    elif hasattr(q_data, 'question'):
                        questions_list.append(q_data)
                sub_questions = Questions(questions=questions_list)
            elif hasattr(sub_questions_raw, 'content') and hasattr(sub_questions_raw, 'role'):
                # It's a ChatMessage object - this shouldn't happen but handle it gracefully
                logger.warning(f"structured_reply is a ChatMessage object - this is unexpected")
                from app.models import Questions
                sub_questions = Questions(questions=[])
            else:
                # Fallback - create empty Questions object
                logger.warning(f"Unexpected structured_reply type: {type(sub_questions_raw)}")
                from app.models import Questions
                sub_questions = Questions(questions=[])
        elif "validator" in results and "valid_questions" in results["validator"]:
            sub_questions = results["validator"]["valid_questions"]

        # 3. Extract document metadata
        document_metadata = self._extract_document_metadata(results)

        return {
            "answer": final_answer,
            "sub_questions": sub_questions,
            "document_metadata": document_metadata,
            "cache_hit": False
        }

    def _extract_document_metadata(self, results: Dict[str, Any]) -> list:
        """Extract document metadata from hybrid retriever results"""
        document_metadata = []

        if ("hybrid_retriever" in results and
                "question_context_pairs" in results["hybrid_retriever"]):

            for pair in results["hybrid_retriever"]["question_context_pairs"]:
                question_text = pair.get("question", "")

                # Process documents for this question
                if "documents" in pair and pair["documents"]:
                    formatted_docs = pair["documents"]

                    # Process each document (limit to top 3 per question)
                    for doc_idx, formatted_doc in enumerate(formatted_docs[:3]):
                        doc_metadata = formatted_doc.get("metadata", {})

                        # Create metadata entry with required fields
                        metadata_entry = {
                            "id": f"doc-{len(document_metadata) + 1}",
                            "score": 1.0 - (doc_idx * 0.1),  # Simulate relevance score
                            "document_id": doc_metadata.get("document_id", f"unknown-{len(document_metadata) + 1}")
                        }

                        # Add title field
                        for title_field in ["case_title", "article_title", "legislation_title"]:
                            if title_field in doc_metadata:
                                metadata_entry[title_field] = doc_metadata[title_field]
                                break

                        # Ensure there's a default title if none found
                        if not any(key in metadata_entry for key in
                                   ["case_title", "article_title", "legislation_title"]):
                            metadata_entry["case_title"] = (
                                f"Result for: {question_text[:30]}..."
                                if question_text else "Unknown document"
                            )

                        document_metadata.append(metadata_entry)

        return document_metadata


# Factory function with caching
@lru_cache(maxsize=1)
def get_decomposition_pipeline() -> AsyncPipeline:
    """Get singleton pipeline instance"""
    pipeline_service = LegalDecompositionPipelineService()
    return pipeline_service.get_pipeline()


# Process a question through the pipeline
async def process_question(question: str) -> Dict[str, Any]:
    """Process a question through the decomposition pipeline"""
    pipeline_service = LegalDecompositionPipelineService()
    return await pipeline_service.run_pipeline(question)
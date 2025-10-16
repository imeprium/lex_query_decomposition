import logging
import threading
import asyncio
from typing import List, Dict, Any, Optional, Type
from haystack import component
from haystack.dataclasses import SparseEmbedding
from haystack_integrations.components.embedders.fastembed import FastembedTextEmbedder, FastembedSparseTextEmbedder
from pydantic import BaseModel
from app.config.settings import DENSE_EMBEDDING_MODEL, SPARSE_EMBEDDING_MODEL
from app.core.singleton import SingletonMeta
from app.core.async_component import AsyncComponent

logger = logging.getLogger("components")


class BaseEmbedderService(metaclass=SingletonMeta):
    """
    Base class for embedder services with shared initialization and warm-up logic.
    """

    def __init__(self, model: str, embedder_class: Type, **kwargs):
        self.model = model
        self.embedder_class = embedder_class
        self.kwargs = kwargs
        self._initialize()

    def _initialize(self):
        """Initialize the embedder with the configured model"""
        logger.info(f"Initializing embedder with model: {self.model}")
        self.embedder = self.embedder_class(
            model=self.model,
            **self.kwargs
        )
        logger.info("Warming up embedder...")
        self.embedder.warm_up()
        logger.info("Embedder initialized and warmed up")

    def get_embedder(self):
        """Get the embedder instance"""
        return self.embedder


class DenseEmbedderService(BaseEmbedderService):
    """Singleton service for dense text embedder"""

    def __init__(self):
        super().__init__(
            model=DENSE_EMBEDDING_MODEL,
            embedder_class=FastembedTextEmbedder,
            prefix="Represent this sentence for searching relevant legal passages:",
            threads=4
        )


class SparseEmbedderService(BaseEmbedderService):
    """Singleton service for sparse text embedder"""

    def __init__(self):
        super().__init__(
            model=SPARSE_EMBEDDING_MODEL,
            embedder_class=FastembedSparseTextEmbedder,
            threads=4
        )


# Fixed inheritance for base embedder class
class MultiQueryEmbedder:
    """
    Base class for multi-query embedders that handles common functionality.
    """

    def __init__(self, service):
        # This is no longer inheriting from AsyncComponent
        # so we don't need to call super().__init__
        self.service = service
        self.embedder = self.service.get_embedder()
        logger.info(f"{self.__class__.__name__} initialized")

    def _process_questions(self, questions: BaseModel):
        """Extract question texts from a BaseModel questions object"""
        if not hasattr(questions, 'questions') or not questions.questions:
            logger.warning("Input has no questions attribute or empty questions list")
            return []
        return [q.question for q in questions.questions]


@component
class MultiQueryDenseEmbedder(MultiQueryEmbedder):
    """
    Component for embedding multiple questions using FastEmbed's dense embedding model.
    Uses the singleton embedder service to avoid reloading models.
    """

    def __init__(self):
        # Call parent class init properly
        MultiQueryEmbedder.__init__(self, DenseEmbedderService())

    @component.output_types(embeddings=List[List[float]])
    def run(self, questions: BaseModel):
        """
        Generate dense embeddings for each question in the Questions object.
        """
        question_texts = self._process_questions(questions)
        embeddings = []

        try:
            for idx, question_text in enumerate(question_texts):
                logger.debug(f"Embedding question {idx + 1}: {question_text[:50]}...")
                embedding_result = self.embedder.run(question_text)
                embeddings.append(embedding_result["embedding"])

            logger.debug(f"Successfully generated {len(embeddings)} dense embeddings")
            return {"embeddings": embeddings}
        except Exception as e:
            logger.error(f"Error generating dense embeddings: {str(e)}", exc_info=True)
            return {"embeddings": []}

    @component.output_types(embeddings=List[List[float]])
    async def run_async(self, questions: BaseModel):
        """
        Asynchronously generate dense embeddings for multiple questions in parallel.
        """
        question_texts = self._process_questions(questions)
        embeddings = []

        try:
            # Create a list of tasks for concurrent execution if embedder supports async
            if hasattr(self.embedder, 'run_async'):
                tasks = []
                for question_text in question_texts:
                    tasks.append(self.embedder.run_async(question_text))

                if tasks:
                    results = await asyncio.gather(*tasks)
                    embeddings = [result["embedding"] for result in results]
            else:
                # Process sequentially in thread pool if no async support
                for question_text in question_texts:
                    # Use asyncio.to_thread directly since we're not using AsyncComponent
                    result = await asyncio.to_thread(self.embedder.run, question_text)
                    embeddings.append(result["embedding"])

            logger.debug(f"Successfully generated {len(embeddings)} dense embeddings asynchronously")
            return {"embeddings": embeddings}
        except Exception as e:
            logger.error(f"Error in async embeddings: {str(e)}", exc_info=True)
            return {"embeddings": []}


@component
class MultiQuerySparseEmbedder(MultiQueryEmbedder):
    """
    Component for embedding multiple questions using FastEmbed's sparse embedding model.
    Uses the singleton embedder service to avoid reloading models.
    """

    def __init__(self):
        # Call parent class init properly
        MultiQueryEmbedder.__init__(self, SparseEmbedderService())

    @component.output_types(sparse_embeddings=List[SparseEmbedding])
    def run(self, questions: BaseModel):
        """Generate sparse embeddings for each question"""
        question_texts = self._process_questions(questions)
        sparse_embeddings = []

        try:
            for idx, question_text in enumerate(question_texts):
                logger.debug(f"Sparse embedding question {idx + 1}: {question_text[:50]}...")
                embedding_result = self.embedder.run(question_text)
                sparse_embeddings.append(embedding_result["sparse_embedding"])

            logger.debug(f"Successfully generated {len(sparse_embeddings)} sparse embeddings")
            return {"sparse_embeddings": sparse_embeddings}
        except Exception as e:
            logger.error(f"Error generating sparse embeddings: {str(e)}", exc_info=True)
            return {"sparse_embeddings": []}

    @component.output_types(sparse_embeddings=List[SparseEmbedding])
    async def run_async(self, questions: BaseModel):
        """Asynchronously generate sparse embeddings for multiple questions"""
        question_texts = self._process_questions(questions)
        sparse_embeddings = []

        try:
            # Use async methods if available
            if hasattr(self.embedder, 'run_async'):
                tasks = []
                for question_text in question_texts:
                    tasks.append(self.embedder.run_async(question_text))

                if tasks:
                    results = await asyncio.gather(*tasks)
                    sparse_embeddings = [result["sparse_embedding"] for result in results]
            else:
                # Process sequentially in thread pool
                for question_text in question_texts:
                    # Use asyncio.to_thread directly since we're not using AsyncComponent
                    result = await asyncio.to_thread(self.embedder.run, question_text)
                    sparse_embeddings.append(result["sparse_embedding"])

            logger.debug(f"Successfully generated {len(sparse_embeddings)} sparse embeddings asynchronously")
            return {"sparse_embeddings": sparse_embeddings}
        except Exception as e:
            logger.error(f"Error in async sparse embeddings: {str(e)}", exc_info=True)
            return {"sparse_embeddings": []}


# Factory functions with cached results
from functools import lru_cache


@lru_cache(maxsize=1)
def get_dense_embedder():
    """Get a singleton MultiQueryDenseEmbedder instance"""
    return MultiQueryDenseEmbedder()


@lru_cache(maxsize=1)
def get_sparse_embedder():
    """Get a singleton MultiQuerySparseEmbedder instance"""
    return MultiQuerySparseEmbedder()
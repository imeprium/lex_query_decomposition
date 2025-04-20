import logging
import threading
import asyncio
from typing import List, Dict, Any, Optional
from haystack import component
from haystack.dataclasses import SparseEmbedding
from haystack_integrations.components.embedders.fastembed import FastembedTextEmbedder, FastembedSparseTextEmbedder
from pydantic import BaseModel
from app.config.settings import DENSE_EMBEDDING_MODEL, SPARSE_EMBEDDING_MODEL

logger = logging.getLogger("components")


# Singleton pattern for model instances to ensure they're loaded only once
class DenseEmbedderService:
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
                DenseEmbedderService._initialized = True

    def _initialize(self):
        """Initialize the dense embedder"""
        logger.info(f"Initializing dense embedder with model: {DENSE_EMBEDDING_MODEL}")
        self.embedder = FastembedTextEmbedder(
            model=DENSE_EMBEDDING_MODEL,
            prefix="Represent this sentence for searching relevant legal passages:",
            threads=4
        )
        logger.info("Warming up dense embedder...")
        self.embedder.warm_up()
        logger.info("Dense embedder initialized and warmed up")

    def get_embedder(self):
        return self.embedder


class SparseEmbedderService:
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
                SparseEmbedderService._initialized = True

    def _initialize(self):
        """Initialize the sparse embedder"""
        logger.info(f"Initializing sparse embedder with model: {SPARSE_EMBEDDING_MODEL}")
        self.embedder = FastembedSparseTextEmbedder(
            model=SPARSE_EMBEDDING_MODEL,
            threads=4
        )
        logger.info("Warming up sparse embedder...")
        self.embedder.warm_up()
        logger.info("Sparse embedder initialized and warmed up")

    def get_embedder(self):
        return self.embedder


# Actual Haystack components that use the singleton services
@component
class MultiQueryDenseEmbedder:
    """
    Component for embedding multiple questions using FastEmbed's dense embedding model.
    Uses the singleton embedder service to avoid reloading models.
    """

    def __init__(self):
        # Get embedder from singleton service
        self.service = DenseEmbedderService()
        self.embedder = self.service.get_embedder()
        logger.info("MultiQueryDenseEmbedder initialized")

    @component.output_types(embeddings=List[List[float]])
    def run(self, questions: BaseModel):
        """
        Generate dense embeddings for each question in the Questions object.

        Args:
            questions: A Pydantic model with a 'questions' attribute

        Returns:
            Dictionary with 'embeddings' key containing a list of dense embeddings.
        """
        logger.debug(f"Generating dense embeddings for {len(questions.questions)} questions")
        embeddings = []

        try:
            for idx, question in enumerate(questions.questions):
                logger.debug(f"Embedding question {idx + 1}: {question.question[:50]}...")
                embedding_result = self.embedder.run(question.question)
                embeddings.append(embedding_result["embedding"])

            logger.debug(f"Successfully generated {len(embeddings)} dense embeddings")
            return {"embeddings": embeddings}

        except Exception as e:
            logger.error(f"Error generating dense embeddings: {str(e)}", exc_info=True)
            # Return empty embeddings if we fail
            return {"embeddings": []}

    @component.output_types(embeddings=List[List[float]])
    async def run_async(self, questions: BaseModel):
        """
        Asynchronously generate dense embeddings for multiple questions in parallel.

        Args:
            questions: A Pydantic model with a 'questions' attribute

        Returns:
            Dictionary with 'embeddings' key containing a list of dense embeddings.
        """
        logger.debug(f"Asynchronously generating dense embeddings for {len(questions.questions)} questions")
        embeddings = []

        try:
            # Check if the underlying embedder has async capabilities
            if hasattr(self.embedder, 'run_async'):
                # Create a list of tasks for concurrent execution
                tasks = []
                for idx, question in enumerate(questions.questions):
                    logger.debug(f"Creating embedding task for question {idx + 1}: {question.question[:50]}...")
                    task = self.embedder.run_async(question.question)
                    tasks.append(task)

                # Execute all embedding tasks concurrently
                if tasks:
                    results = await asyncio.gather(*tasks)
                    embeddings = [result["embedding"] for result in results]
            else:
                # If no async support in underlying embedder, process sequentially in thread pool
                for idx, question in enumerate(questions.questions):
                    logger.debug(f"Sequential embedding of question {idx + 1}: {question.question[:50]}...")
                    # Use asyncio.to_thread to avoid blocking the event loop
                    embedding_result = await asyncio.to_thread(self.embedder.run, question.question)
                    embeddings.append(embedding_result["embedding"])

            logger.debug(f"Successfully generated {len(embeddings)} dense embeddings asynchronously")
            return {"embeddings": embeddings}

        except Exception as e:
            logger.error(f"Error generating dense embeddings asynchronously: {str(e)}", exc_info=True)
            # Return empty embeddings if we fail
            return {"embeddings": []}


@component
class MultiQuerySparseEmbedder:
    """
    Component for embedding multiple questions using FastEmbed's sparse embedding model.
    Uses the singleton embedder service to avoid reloading models.
    """

    def __init__(self):
        # Get embedder from singleton service
        self.service = SparseEmbedderService()
        self.embedder = self.service.get_embedder()
        logger.info("MultiQuerySparseEmbedder initialized")

    @component.output_types(sparse_embeddings=List[SparseEmbedding])
    def run(self, questions: BaseModel):
        """
        Generate sparse embeddings for each question in the Questions object.

        Args:
            questions: A Pydantic model with a 'questions' attribute

        Returns:
            Dictionary with 'sparse_embeddings' key containing a list of sparse embeddings.
        """
        logger.debug(f"Generating sparse embeddings for {len(questions.questions)} questions")
        sparse_embeddings = []

        try:
            for idx, question in enumerate(questions.questions):
                logger.debug(f"Sparse embedding question {idx + 1}: {question.question[:50]}...")
                embedding_result = self.embedder.run(question.question)
                sparse_embeddings.append(embedding_result["sparse_embedding"])

            logger.debug(f"Successfully generated {len(sparse_embeddings)} sparse embeddings")
            return {"sparse_embeddings": sparse_embeddings}

        except Exception as e:
            logger.error(f"Error generating sparse embeddings: {str(e)}", exc_info=True)
            # Return empty embeddings if we fail
            return {"sparse_embeddings": []}

    @component.output_types(sparse_embeddings=List[SparseEmbedding])
    async def run_async(self, questions: BaseModel):
        """
        Asynchronously generate sparse embeddings for multiple questions in parallel.

        Args:
            questions: A Pydantic model with a 'questions' attribute

        Returns:
            Dictionary with 'sparse_embeddings' key containing a list of sparse embeddings.
        """
        logger.debug(f"Asynchronously generating sparse embeddings for {len(questions.questions)} questions")
        sparse_embeddings = []

        try:
            # Check if the underlying embedder has async capabilities
            if hasattr(self.embedder, 'run_async'):
                # Create a list of tasks for concurrent execution
                tasks = []
                for idx, question in enumerate(questions.questions):
                    logger.debug(f"Creating sparse embedding task for question {idx + 1}: {question.question[:50]}...")
                    task = self.embedder.run_async(question.question)
                    tasks.append(task)

                # Execute all embedding tasks concurrently
                if tasks:
                    results = await asyncio.gather(*tasks)
                    sparse_embeddings = [result["sparse_embedding"] for result in results]
            else:
                # If no async support in underlying embedder, process sequentially in thread pool
                for idx, question in enumerate(questions.questions):
                    logger.debug(f"Sequential sparse embedding of question {idx + 1}: {question.question[:50]}...")
                    # Use asyncio.to_thread to avoid blocking the event loop
                    embedding_result = await asyncio.to_thread(self.embedder.run, question.question)
                    sparse_embeddings.append(embedding_result["sparse_embedding"])

            logger.debug(f"Successfully generated {len(sparse_embeddings)} sparse embeddings asynchronously")
            return {"sparse_embeddings": sparse_embeddings}

        except Exception as e:
            logger.error(f"Error generating sparse embeddings asynchronously: {str(e)}", exc_info=True)
            # Return empty embeddings if we fail
            return {"sparse_embeddings": []}


# Factory functions to get components (now returning component instances, not services)
def get_dense_embedder():
    return MultiQueryDenseEmbedder()


def get_sparse_embedder():
    return MultiQuerySparseEmbedder()
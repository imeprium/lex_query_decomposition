import logging
import threading
import asyncio
from typing import List, Dict, Any, Optional, Set
from haystack import component, Document
from haystack.dataclasses import SparseEmbedding
from haystack_integrations.components.retrievers.qdrant import QdrantHybridRetriever
from haystack_integrations.components.rankers.fastembed import FastembedRanker
from haystack.components.joiners import DocumentJoiner
from pydantic import BaseModel
from app.config.settings import DEFAULT_TOP_K, DEFAULT_SCORE_THRESHOLD, RANKER_MODEL

logger = logging.getLogger("components")


# Singleton ranker service to ensure model is loaded only once
class RankerService:
    """
    Singleton service for FastEmbed ranker
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
                RankerService._initialized = True

    def _initialize(self):
        """Initialize the ranker"""
        logger.info(f"Initializing FastEmbed ranker with model: {RANKER_MODEL}")
        self.ranker = FastembedRanker(
            model_name=RANKER_MODEL,
            top_k=DEFAULT_TOP_K,
            meta_fields_to_embed=["case_title", "court", "year"],
            meta_data_separator=" | "
        )

        logger.info("Warming up ranker...")
        self.ranker.warm_up()
        logger.info("Ranker initialized and warmed up")

    def get_ranker(self):
        """Get the ranker instance"""
        return self.ranker


# Actual Haystack component that uses the singleton service
@component
class MultiQueryHybridRetriever:
    """
    Component that performs hybrid retrieval for multiple queries.
    Uses QdrantHybridRetriever for each query's dense and sparse embeddings.
    """

    def __init__(self, retriever, ranker=None):
        """
        Initialize the multi-query hybrid retriever

        Args:
            retriever: A QdrantHybridRetriever instance
            ranker: Optional FastembedRanker instance
        """
        self.retriever = retriever

        # Get ranker from service if not provided
        if ranker is None:
            ranker_service = RankerService()
            self.ranker = ranker_service.get_ranker()
        else:
            self.ranker = ranker

        # Initialize joiner for combining multiple document sets
        self.joiner = DocumentJoiner(join_mode="reciprocal_rank_fusion")

        logger.info("Multi-query hybrid retriever initialized")

    def _extract_document_metadata(self, doc: Document) -> Dict[str, Any]:
        """
        Extract relevant metadata from a document

        Args:
            doc: The document object from which to extract metadata

        Returns:
            Dictionary containing document_id and one of case_title, article_title, or legislation_title
        """
        metadata = {}

        # Extract document ID (prioritize document_id from meta if available)
        if hasattr(doc, "meta") and doc.meta and "document_id" in doc.meta:
            metadata["document_id"] = doc.meta["document_id"]
        else:
            metadata["document_id"] = getattr(doc, "id", None)

        # Extract one title field (case_title, article_title, or legislation_title)
        if hasattr(doc, "meta") and doc.meta:
            for title_field in ["case_title", "article_title", "legislation_title"]:
                if title_field in doc.meta and doc.meta[title_field]:
                    metadata[title_field] = doc.meta[title_field]
                    break  # Only take the first available title field

        return metadata

    def _format_documents_for_context(self, documents: List[Document]) -> List[Dict[str, Any]]:
        """
        Format retrieved documents with both content and metadata, ensuring unique sources

        Args:
            documents: List of retrieved documents

        Returns:
            List of dictionaries containing document content and metadata with unique sources
        """
        formatted_docs = []
        seen_sources = set()  # Track unique sources

        # Sort documents by score if available for better selection
        sorted_docs = sorted(documents, key=lambda x: getattr(x, 'score', 0) or 0, reverse=True)

        for doc in sorted_docs:
            # Extract metadata
            metadata = self._extract_document_metadata(doc)

            # Generate a source key based on title field and document ID
            source_key = metadata.get("document_id", "")

            # Add title field to source key if available
            for title_field in ["case_title", "article_title", "legislation_title"]:
                if title_field in metadata:
                    source_key = f"{title_field}:{metadata[title_field]}"
                    break

            # Skip if we've already seen this source
            if source_key in seen_sources:
                continue

            # Add to seen sources
            seen_sources.add(source_key)

            # Create document with content and metadata
            formatted_doc = {
                "content": doc.content,
                "metadata": metadata
            }

            formatted_docs.append(formatted_doc)

        return formatted_docs

    @component.output_types(question_context_pairs=List[Dict])
    def run(self, queries: BaseModel, dense_embeddings: List[List[float]],
            sparse_embeddings: List[SparseEmbedding], top_k: Optional[int] = None):
        """
        Retrieve documents for multiple queries using hybrid search.

        Args:
            queries: A Pydantic model with a 'questions' attribute
            dense_embeddings: List of dense embeddings, one per query
            sparse_embeddings: List of sparse embeddings, one per query
            top_k: Number of documents to retrieve per query

        Returns:
            Dictionary with 'question_context_pairs' containing a list of question-document pairs
        """
        if top_k is None:
            top_k = DEFAULT_TOP_K

        logger.info(f"Retrieving documents for {len(queries.questions)} queries with top_k={top_k}")
        question_context_pairs = []

        try:
            for idx, (query, dense_emb, sparse_emb) in enumerate(
                    zip(queries.questions, dense_embeddings, sparse_embeddings)
            ):
                # Get the question text for use in reranker
                query_text = query.question if hasattr(query, "question") else str(query)

                # Retrieve documents using hybrid search
                try:
                    retrieval_result = self.retriever.run(
                        query_embedding=dense_emb,
                        query_sparse_embedding=sparse_emb,
                        top_k=top_k
                    )
                    doc_count = len(retrieval_result["documents"]) if "documents" in retrieval_result else 0
                    logger.info(f"Retrieved {doc_count} documents for query #{idx + 1}")
                except Exception as e:
                    logger.error(f"Error during document retrieval: {str(e)}")
                    retrieval_result = {"documents": []}

                if not retrieval_result["documents"]:
                    logger.warning(f"No documents retrieved for query #{idx + 1}")
                    docs = []
                else:
                    # Rerank the documents
                    try:
                        rerank_result = self.ranker.run(
                            query=query_text,
                            documents=retrieval_result["documents"],
                            top_k=top_k
                        )
                        docs = rerank_result["documents"]
                    except Exception as e:
                        logger.error(f"Error in document reranking: {str(e)}")
                        docs = retrieval_result["documents"]

                # Format documents with content and metadata
                formatted_docs = self._format_documents_for_context(docs)

                # Create the question-context pair
                question_context_pairs.append({
                    "question": query_text,
                    "documents": formatted_docs
                })

            logger.info(f"Completed retrieval for {len(question_context_pairs)} queries")
            return {"question_context_pairs": question_context_pairs}

        except Exception as e:
            logger.error(f"Error in document retrieval: {str(e)}", exc_info=True)
            return {"question_context_pairs": []}

    @component.output_types(question_context_pairs=List[Dict])
    async def run_async(self, queries: BaseModel, dense_embeddings: List[List[float]],
                        sparse_embeddings: List[SparseEmbedding], top_k: Optional[int] = None):
        """
        Asynchronously retrieve documents for multiple queries using hybrid search.

        Args:
            queries: A Pydantic model with a 'questions' attribute
            dense_embeddings: List of dense embeddings, one per query
            sparse_embeddings: List of sparse embeddings, one per query
            top_k: Number of documents to retrieve per query

        Returns:
            Dictionary with 'question_context_pairs' containing a list of question-document pairs
        """
        if top_k is None:
            top_k = DEFAULT_TOP_K

        logger.info(f"Asynchronously retrieving documents for {len(queries.questions)} queries with top_k={top_k}")
        question_context_pairs = []

        try:
            # Process all queries concurrently
            async def process_query(idx, query, dense_emb, sparse_emb):
                # Get the question text for use in reranker
                query_text = query.question if hasattr(query, "question") else str(query)

                # Retrieve documents using hybrid search (asynchronously if possible)
                try:
                    # Check if retriever has async support
                    if hasattr(self.retriever, 'run_async'):
                        retrieval_result = await self.retriever.run_async(
                            query_embedding=dense_emb,
                            query_sparse_embedding=sparse_emb,
                            top_k=top_k
                        )
                    else:
                        # Fallback to sync execution in thread pool
                        retrieval_result = await asyncio.to_thread(
                            self.retriever.run,
                            query_embedding=dense_emb,
                            query_sparse_embedding=sparse_emb,
                            top_k=top_k
                        )

                    doc_count = len(retrieval_result["documents"]) if "documents" in retrieval_result else 0
                    logger.info(f"Retrieved {doc_count} documents for query #{idx + 1}")
                except Exception as e:
                    logger.error(f"Error during async document retrieval: {str(e)}")
                    retrieval_result = {"documents": []}

                if not retrieval_result["documents"]:
                    logger.warning(f"No documents retrieved for query #{idx + 1}")
                    docs = []
                else:
                    # Rerank the documents (asynchronously if possible)
                    try:
                        # Check if ranker has async support
                        if hasattr(self.ranker, 'run_async'):
                            rerank_result = await self.ranker.run_async(
                                query=query_text,
                                documents=retrieval_result["documents"],
                                top_k=top_k
                            )
                        else:
                            # Fallback to sync execution in thread pool
                            rerank_result = await asyncio.to_thread(
                                self.ranker.run,
                                query=query_text,
                                documents=retrieval_result["documents"],
                                top_k=top_k
                            )

                        docs = rerank_result["documents"]
                    except Exception as e:
                        logger.error(f"Error in async document reranking: {str(e)}")
                        docs = retrieval_result["documents"]

                # Format documents with content and metadata
                formatted_docs = self._format_documents_for_context(docs)

                # Return the question-context pair
                return {
                    "question": query_text,
                    "documents": formatted_docs
                }

            # Create tasks for all queries
            tasks = []
            for idx, (query, dense_emb, sparse_emb) in enumerate(
                    zip(queries.questions, dense_embeddings, sparse_embeddings)
            ):
                task = process_query(idx, query, dense_emb, sparse_emb)
                tasks.append(task)

            # Execute all tasks concurrently
            if tasks:
                results = await asyncio.gather(*tasks)
                question_context_pairs = results

            logger.info(f"Completed async retrieval for {len(question_context_pairs)} queries")
            return {"question_context_pairs": question_context_pairs}

        except Exception as e:
            logger.error(f"Error in async document retrieval: {str(e)}", exc_info=True)
            return {"question_context_pairs": []}


# Factory functions - preserved from original implementation
def get_ranker():
    """Get singleton ranker instance"""
    ranker_service = RankerService()
    return ranker_service.get_ranker()


def get_hybrid_retriever(retriever):
    """Get a new multi-query hybrid retriever instance"""
    ranker = get_ranker()
    return MultiQueryHybridRetriever(retriever, ranker)
import logging
import asyncio
from typing import List, Dict, Any, Optional, Set
from haystack import component, Document
from haystack.dataclasses import SparseEmbedding
from haystack_integrations.components.retrievers.qdrant import QdrantHybridRetriever
from haystack_integrations.components.rankers.fastembed import FastembedRanker
from haystack.components.joiners import DocumentJoiner
from pydantic import BaseModel
from functools import lru_cache
from app.config.settings import DEFAULT_TOP_K, DEFAULT_SCORE_THRESHOLD, RANKER_MODEL
from app.core.singleton import SingletonMeta
from app.core.async_component import AsyncComponent

logger = logging.getLogger("components")


class RankerService(metaclass=SingletonMeta):
    """Singleton service for FastEmbed ranker"""

    def __init__(self):
        self._initialize()

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


class DocumentFormatter:
    """Handles document formatting for retrieval results"""

    def format_documents(self, documents: List[Document]) -> List[Dict[str, Any]]:
        """
        Format retrieved documents with content and metadata, ensuring unique sources
        """
        formatted_docs = []
        seen_sources = set()

        # Sort documents by score if available
        sorted_docs = sorted(
            documents,
            key=lambda x: getattr(x, 'score', 0) or 0,
            reverse=True
        )

        for doc in sorted_docs:
            # Extract metadata
            metadata = self._extract_document_metadata(doc)
            source_key = self._generate_source_key(metadata)

            # Skip duplicates
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

    def _extract_document_metadata(self, doc: Document) -> Dict[str, Any]:
        """Extract relevant metadata from a document"""
        metadata = {}

        # Extract document ID
        if hasattr(doc, "meta") and doc.meta and "document_id" in doc.meta:
            metadata["document_id"] = doc.meta["document_id"]
        else:
            metadata["document_id"] = getattr(doc, "id", None)

        # Extract title field
        if hasattr(doc, "meta") and doc.meta:
            for title_field in ["case_title", "article_title", "legislation_title"]:
                if title_field in doc.meta and doc.meta[title_field]:
                    metadata[title_field] = doc.meta[title_field]
                    break  # Only take the first available title field

        return metadata

    def _generate_source_key(self, metadata: Dict[str, Any]) -> str:
        """Generate a unique key for deduplication"""
        # Use document ID as base
        source_key = metadata.get("document_id", "unknown")

        # Add title if available
        for title_field in ["case_title", "article_title", "legislation_title"]:
            if title_field in metadata:
                source_key = f"{title_field}:{metadata[title_field]}"
                break

        return source_key


@component
class MultiQueryHybridRetriever(AsyncComponent):
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

        # Document formatter for consistent formatting
        self.formatter = DocumentFormatter()

        logger.info("Multi-query hybrid retriever initialized")

    @component.output_types(question_context_pairs=List[Dict])
    def run(self, queries: BaseModel, dense_embeddings: List[List[float]],
            sparse_embeddings: List[SparseEmbedding], top_k: Optional[int] = None):
        """
        Retrieve documents for multiple queries using hybrid search.
        """
        top_k = top_k or DEFAULT_TOP_K
        logger.info(f"Retrieving documents for {len(queries.questions)} queries with top_k={top_k}")

        question_context_pairs = []

        # Process each query
        for idx, (query, dense_emb, sparse_emb) in enumerate(
                zip(queries.questions, dense_embeddings, sparse_embeddings)
        ):
            try:
                # Get the query text
                query_text = query.question if hasattr(query, "question") else str(query)

                # Retrieve documents
                docs = self._retrieve_and_rank(
                    query_text=query_text,
                    dense_emb=dense_emb,
                    sparse_emb=sparse_emb,
                    top_k=top_k
                )

                # Format documents
                formatted_docs = self.formatter.format_documents(docs)

                # Create question-context pair
                question_context_pairs.append({
                    "question": query_text,
                    "documents": formatted_docs
                })
            except Exception as e:
                logger.error(f"Error processing query #{idx + 1}: {str(e)}", exc_info=True)
                # Add empty result for this query to maintain order
                question_context_pairs.append({
                    "question": query.question if hasattr(query, "question") else str(query),
                    "documents": []
                })

        logger.info(f"Completed retrieval for {len(question_context_pairs)} queries")
        return {"question_context_pairs": question_context_pairs}

    @component.output_types(question_context_pairs=List[Dict])
    async def run_async(self, queries: BaseModel, dense_embeddings: List[List[float]],
                        sparse_embeddings: List[SparseEmbedding], top_k: Optional[int] = None):
        """
        Asynchronously retrieve documents for multiple queries using hybrid search.
        """
        top_k = top_k or DEFAULT_TOP_K
        logger.info(f"Asynchronously retrieving documents for {len(queries.questions)} queries")

        async def process_query(idx, query, dense_emb, sparse_emb):
            """Process a single query asynchronously"""
            try:
                # Get query text
                query_text = query.question if hasattr(query, "question") else str(query)

                # Retrieve and rank documents
                docs = await self._retrieve_and_rank_async(
                    query_text=query_text,
                    dense_emb=dense_emb,
                    sparse_emb=sparse_emb,
                    top_k=top_k
                )

                # Format documents
                formatted_docs = self.formatter.format_documents(docs)

                return {
                    "question": query_text,
                    "documents": formatted_docs
                }
            except Exception as e:
                logger.error(f"Error in async processing of query #{idx + 1}: {str(e)}")
                return {
                    "question": query.question if hasattr(query, "question") else str(query),
                    "documents": []
                }

        # Create tasks for all queries
        tasks = []
        for idx, (query, dense_emb, sparse_emb) in enumerate(
                zip(queries.questions, dense_embeddings, sparse_embeddings)
        ):
            tasks.append(process_query(idx, query, dense_emb, sparse_emb))

        # Execute all tasks concurrently
        question_context_pairs = await asyncio.gather(*tasks) if tasks else []

        logger.info(f"Completed async retrieval for {len(question_context_pairs)} queries")
        return {"question_context_pairs": question_context_pairs}

    def _retrieve_and_rank(self, query_text, dense_emb, sparse_emb, top_k):
        """Retrieve and rerank documents"""
        # Retrieve documents
        retrieval_result = self._retrieve_documents(
            dense_emb=dense_emb,
            sparse_emb=sparse_emb,
            top_k=top_k
        )

        if not retrieval_result["documents"]:
            return []

        # Rerank documents
        try:
            rerank_result = self.ranker.run(
                query=query_text,
                documents=retrieval_result["documents"],
                top_k=top_k
            )
            return rerank_result["documents"]
        except Exception as e:
            logger.error(f"Error reranking: {str(e)}")
            return retrieval_result["documents"]

    async def _retrieve_and_rank_async(self, query_text, dense_emb, sparse_emb, top_k):
        """Retrieve and rerank documents asynchronously"""
        # Retrieve documents
        retrieval_result = await self._retrieve_documents_async(
            dense_emb=dense_emb,
            sparse_emb=sparse_emb,
            top_k=top_k
        )

        if not retrieval_result["documents"]:
            return []

        # Rerank documents
        try:
            if hasattr(self.ranker, 'run_async'):
                rerank_result = await self.ranker.run_async(
                    query=query_text,
                    documents=retrieval_result["documents"],
                    top_k=top_k
                )
            else:
                rerank_result = await self.to_thread(
                    self.ranker.run,
                    query=query_text,
                    documents=retrieval_result["documents"],
                    top_k=top_k
                )

            return rerank_result["documents"]
        except Exception as e:
            logger.error(f"Error reranking asynchronously: {str(e)}")
            return retrieval_result["documents"]

    def _retrieve_documents(self, dense_emb, sparse_emb, top_k):
        """Retrieve documents using hybrid search"""
        try:
            return self.retriever.run(
                query_embedding=dense_emb,
                query_sparse_embedding=sparse_emb,
                top_k=top_k
            )
        except Exception as e:
            logger.error(f"Error during document retrieval: {str(e)}")
            return {"documents": []}

    async def _retrieve_documents_async(self, dense_emb, sparse_emb, top_k):
        """Retrieve documents using hybrid search asynchronously"""
        try:
            if hasattr(self.retriever, 'run_async'):
                return await self.retriever.run_async(
                    query_embedding=dense_emb,
                    query_sparse_embedding=sparse_emb,
                    top_k=top_k
                )
            else:
                return await self.to_thread(
                    self.retriever.run,
                    query_embedding=dense_emb,
                    query_sparse_embedding=sparse_emb,
                    top_k=top_k
                )
        except Exception as e:
            logger.error(f"Error during async document retrieval: {str(e)}")
            return {"documents": []}


# Factory functions
@lru_cache(maxsize=1)
def get_ranker():
    """Get singleton ranker instance"""
    ranker_service = RankerService()
    return ranker_service.get_ranker()


def get_hybrid_retriever(retriever):
    """Get a new multi-query hybrid retriever instance"""
    ranker = get_ranker()
    return MultiQueryHybridRetriever(retriever, ranker)
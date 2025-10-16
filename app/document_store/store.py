import logging
import asyncio
from functools import lru_cache
from haystack_integrations.document_stores.qdrant import QdrantDocumentStore
from haystack.utils import Secret
from app.config.settings import QDRANT_URL, QDRANT_API_KEY, QDRANT_COLLECTION_NAME
from app.core.singleton import SingletonMeta

logger = logging.getLogger("document_store")


class DocumentStoreService(metaclass=SingletonMeta):
    """Singleton service for Qdrant document store"""

    def __init__(self):
        self._initialize()

    def _initialize(self):
        """Initialize the Qdrant document store"""
        logger.info(f"Initializing Qdrant document store with collection: {QDRANT_COLLECTION_NAME}")

        try:
            # Convert the API key to a Secret object
            api_key_secret = Secret.from_token(QDRANT_API_KEY) if QDRANT_API_KEY else None

            self.document_store = QdrantDocumentStore(
                url=QDRANT_URL,
                api_key=api_key_secret,
                index=QDRANT_COLLECTION_NAME,
                embedding_dim=384,  # Common dimension for BAAI/bge-small-en-v1.5
                use_sparse_embeddings=True,  # Enable hybrid search
                similarity="cosine",
                prefer_grpc=True,
            )

            logger.info(f"Successfully connected to Qdrant. Collection: {QDRANT_COLLECTION_NAME}")
            logger.info(f"Document count: {self.document_store.count_documents()}")

        except Exception as e:
            logger.error(f"Failed to initialize Qdrant document store: {str(e)}")
            raise

    def get_document_store(self):
        """Get the Qdrant document store instance"""
        return self.document_store

    async def count_documents_async(self):
        """Asynchronously count documents in the store"""
        try:
            # Use native async method if available
            if hasattr(self.document_store, 'count_documents_async'):
                return await self.document_store.count_documents_async()
            else:
                # Fall back to sync method in thread pool
                return await asyncio.to_thread(self.document_store.count_documents)
        except Exception as e:
            logger.error(f"Error counting documents asynchronously: {str(e)}")
            return 0

    async def query_async(self, method_name, *args, **kwargs):
        """
        Generic async query method that handles both native async and sync methods.

        Args:
            method_name: Name of the method to call
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Query results
        """
        try:
            # Check for native async method
            async_method_name = f"{method_name}_async"

            if hasattr(self.document_store, async_method_name):
                # Use native async method
                async_method = getattr(self.document_store, async_method_name)
                return await async_method(*args, **kwargs)
            else:
                # Fall back to sync method in thread pool
                sync_method = getattr(self.document_store, method_name)
                return await asyncio.to_thread(sync_method, *args, **kwargs)
        except Exception as e:
            logger.error(f"Error in async document store query: {str(e)}")
            raise


# Singleton factory functions
@lru_cache(maxsize=1)
def get_document_store():
    """Get the document store instance"""
    document_store_service = DocumentStoreService()
    return document_store_service.get_document_store()


async def count_documents_async():
    """Count documents asynchronously"""
    document_store_service = DocumentStoreService()
    return await document_store_service.count_documents_async()


async def query_documents_async(method_name, *args, **kwargs):
    """Generic async query wrapper"""
    document_store_service = DocumentStoreService()
    return await document_store_service.query_async(method_name, *args, **kwargs)
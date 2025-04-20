import logging
import threading
import asyncio
from haystack_integrations.document_stores.qdrant import QdrantDocumentStore
from haystack.utils import Secret
from app.config.settings import QDRANT_URL, QDRANT_API_KEY, QDRANT_COLLECTION_NAME

logger = logging.getLogger("document_store")


class DocumentStoreService:
    """Singleton service for Qdrant document store"""
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
                DocumentStoreService._initialized = True

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

    async def async_count_documents(self):
        """
        Asynchronously count documents in the document store
        """
        try:
            # If document_store has async count method, use it
            if hasattr(self.document_store, 'count_documents_async'):
                return await self.document_store.count_documents_async()
            else:
                # Otherwise, use a thread pool
                return await asyncio.to_thread(self.document_store.count_documents)
        except Exception as e:
            logger.error(f"Error counting documents asynchronously: {str(e)}")
            return 0

    async def async_query(self, *args, **kwargs):
        """
        Asynchronously query the document store

        This is a generic method that can be used for any query operation.
        It will use native async methods if available, otherwise it will run
        the synchronous methods in a thread pool.
        """
        try:
            # Check if document_store has a matching async method based on first parameter
            method_name = kwargs.pop('method', 'query')
            async_method_name = f"{method_name}_async"

            if hasattr(self.document_store, async_method_name):
                # Use native async method if available
                async_method = getattr(self.document_store, async_method_name)
                return await async_method(*args, **kwargs)
            else:
                # Fallback to sync method in thread pool
                sync_method = getattr(self.document_store, method_name)
                return await asyncio.to_thread(sync_method, *args, **kwargs)
        except Exception as e:
            logger.error(f"Error in async document store query: {str(e)}")
            raise


# Singleton instance
document_store_service = DocumentStoreService()


def get_document_store():
    """Get the singleton document store instance"""
    return document_store_service.get_document_store()


async def async_get_document_count():
    """Get the document count asynchronously"""
    return await document_store_service.async_count_documents()


async def async_query_documents(*args, **kwargs):
    """Query documents asynchronously"""
    return await document_store_service.async_query(*args, **kwargs)
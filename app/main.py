"""
Main application entry point for the Legal Query Decomposition API.
Sets up FastAPI application with dependencies and routes.
"""
import time
import logging
import uvicorn
from contextlib import asynccontextmanager
from pathlib import Path
from functools import lru_cache

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.config.logging import setup_logging
from app.endpoints.ask import router
from app.endpoints.chat import router as chat_router
from app.endpoints.chat_fixed import router as chat_fixed_router
from app.document_store.store import get_document_store, count_documents_async
from app.components.embedders import get_dense_embedder, get_sparse_embedder
from app.components.retrievers import get_ranker
from app.pipelines.legal_decomposition_pipeline import get_decomposition_pipeline
from app.config.settings import (
    APP_HOST, APP_PORT,
    CORS_ALLOWED_ORIGINS, CORS_ALLOW_CREDENTIALS,
    CORS_ALLOWED_METHODS, CORS_ALLOWED_HEADERS
)
from app.utils.cache import get_redis_client, DummyRedisClient
from app.auth import (
    auth_settings,
    create_authentication_middleware,
    create_rate_limit_middleware,
    create_request_logging_middleware
)

# Setup logging
logger = setup_logging()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Asynchronous lifespan context manager to handle startup and shutdown events.
    Initializes all resources and performs cleanup on shutdown.
    """
    # Startup: Initialize all resources
    start_time = time.time()
    logger.info("Starting Legal Query Decomposition API")

    try:
        # Ensure directories exist
        _ensure_directories()

        # Initialize and warm up core components
        await _initialize_components()

        # Log startup information
        _log_startup_info()

        init_time = time.time() - start_time
        logger.info(f"Startup completed in {init_time:.2f} seconds")

    except Exception as e:
        logger.error(f"Startup error: {str(e)}", exc_info=True)
        # Continue startup even with errors to allow partial functionality

    # Yield control back to FastAPI
    yield

    # Shutdown: Clean up resources
    logger.info("Shutting down Legal Query Decomposition API")
    # Any cleanup tasks would go here


def _ensure_directories():
    """Ensure required directories exist"""
    directories = ["static", "static/signatures", "logs"]
    for directory in directories:
        Path(directory).mkdir(exist_ok=True, parents=True)
        logger.debug(f"Ensured directory exists: {directory}")


async def _initialize_components():
    """Initialize and warm up core components"""
    # Initialize document store
    logger.info("Initializing document store...")
    document_store = get_document_store()
    doc_count = await count_documents_async()
    logger.info(f"Document store initialized with {doc_count} documents")

    # Initialize cache service
    logger.info("Initializing Redis cache client...")
    redis_client = await get_redis_client()
    if isinstance(redis_client, DummyRedisClient):
        logger.warning("Redis cache is not available, using dummy client")
    else:
        logger.info("Redis cache client initialized successfully")

    # Initialize authentication if enabled
    if auth_settings.auth_enabled:
        logger.info("Initializing authentication system...")
        logger.info(f"Authentication enabled for JWKS URL: {auth_settings.jwks_url}")
        logger.info(f"Mock authentication mode: {auth_settings.dev_mode_mock_auth}")
    else:
        logger.info("Authentication is disabled")

    # Warm up embedders
    logger.info("Initializing embedders...")
    dense_embedder = get_dense_embedder()
    sparse_embedder = get_sparse_embedder()

    # Initialize ranker
    logger.info("Initializing ranker...")
    ranker = get_ranker()

    # Initialize pipeline (this preloads all components)
    logger.info("Initializing pipeline...")
    pipeline = get_decomposition_pipeline()

    # Initialize chat service components
    logger.info("Initializing chat service components...")
    from app.services.legal_chat_service import get_legal_chat_service
    from app.services.enhanced_pipeline_service import get_enhanced_pipeline_service
    from app.services.unified_chat_service import get_unified_chat_service

    # Warm up chat services
    chat_service = get_legal_chat_service()
    enhanced_service = get_enhanced_pipeline_service()
    unified_service = get_unified_chat_service()
    logger.info("Chat service components initialized (including fixed unified service)")


def _log_startup_info():
    """Log important startup information"""
    logger.info(f"CORS allowed origins: {CORS_ALLOWED_ORIGINS}")
    logger.info(f"Server will listen on {APP_HOST}:{APP_PORT}")


# Create FastAPI app with lifespan handler
app = FastAPI(
    title="Legal Query Decomposition API",
    description="API for decomposing and answering complex legal queries",
    version="1.0.0",
    lifespan=lifespan
)

# Configure CORS with settings from environment variables
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ALLOWED_ORIGINS,
    allow_credentials=CORS_ALLOW_CREDENTIALS,
    allow_methods=CORS_ALLOWED_METHODS,
    allow_headers=CORS_ALLOWED_HEADERS,
)

# Add authentication middleware if enabled
if auth_settings.auth_enabled:
    # Add request logging middleware
    app.add_middleware(
        create_request_logging_middleware,
        log_level="INFO"
    )

    # Add rate limiting middleware
    app.add_middleware(
        create_rate_limit_middleware,
        redis_client=None  # Will be initialized during startup
    )

    # Add authentication middleware
    app.add_middleware(
        create_authentication_middleware,
        exclude_paths=[
            "/",
            "/health",
            "/docs",
            "/redoc",
            "/openapi.json",
            "/static"
        ]
    )

    logger.info("Authentication middleware enabled")

# Mount static files directory for logo, signatures, etc.
app.mount("/static", StaticFiles(directory="static"), name="static")

# Include API router with endpoints
app.include_router(router)
app.include_router(chat_router)
app.include_router(chat_fixed_router)


@app.get("/")
async def root():
    """Root endpoint with simple health check"""
    return {
        "status": "online",
        "message": "Legal Query Decomposition API is running",
        "version": "1.0.0"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring"""
    try:
        # Check document store connectivity
        doc_count = await count_documents_async()

        # Check cache service
        redis_client = await get_redis_client()
        cache_status = "connected"
        if isinstance(redis_client, DummyRedisClient):
            cache_status = "disabled"

        return {
            "status": "healthy",
            "document_store": {
                "status": "connected",
                "document_count": doc_count
            },
            "cache": {
                "status": cache_status,
                "type": redis_client.__class__.__name__
            }
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return {
            "status": "unhealthy",
            "error": str(e)
        }


def main():
    """Entry point for running the application"""
    uvicorn.run(
        "app.main:app",
        host=APP_HOST,
        port=APP_PORT,
        log_level="info"
    )


if __name__ == "__main__":
    main()
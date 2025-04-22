from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import logging
import time
import uvicorn
from pathlib import Path

from app.config.logging import setup_logging
from app.endpoints.ask import router  # Correct import path based on file structure
from app.document_store.store import get_document_store
from app.config.settings import (
    APP_HOST, APP_PORT,
    CORS_ALLOWED_ORIGINS, CORS_ALLOW_CREDENTIALS,
    CORS_ALLOWED_METHODS, CORS_ALLOWED_HEADERS
)
from app.pipelines.legal_decomposition_pipeline import get_decomposition_pipeline
from app.components.embedders import get_dense_embedder, get_sparse_embedder
from app.components.retrievers import get_ranker

# Setup logging
logger = setup_logging()

# Ensure directories exist
static_dir = Path("static")
static_dir.mkdir(exist_ok=True)
signatures_dir = Path("static/signatures")
signatures_dir.mkdir(exist_ok=True, parents=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Asynchronous lifespan context manager to handle startup and shutdown events.
    This replaces the deprecated @app.on_event decorators.
    """
    # Startup: Initialize all resources
    start_time = time.time()
    logger.info("Starting Legal Query Decomposition API")

    try:
        # Initialize document store
        logger.info("Initializing document store...")
        document_store = get_document_store()
        logger.info(f"Document store initialized with {document_store.count_documents()} documents")

        # Warm up models
        logger.info("Warming up models...")
        dense_embedder = get_dense_embedder()
        sparse_embedder = get_sparse_embedder()
        ranker = get_ranker()

        # Initialize pipeline (this preloads all components)
        logger.info("Initializing pipeline...")
        pipeline = get_decomposition_pipeline()

        # Log CORS settings
        logger.info(f"CORS allowed origins: {CORS_ALLOWED_ORIGINS}")

        init_time = time.time() - start_time
        logger.info(f"Startup completed in {init_time:.2f} seconds")

    except Exception as e:
        logger.error(f"Startup error: {str(e)}", exc_info=True)
        # Don't crash the application, but log the error

    # Yield control back to FastAPI
    yield

    # Shutdown: Clean up resources if needed
    logger.info("Shutting down Legal Query Decomposition API")
    # Any cleanup tasks would go here


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

# Mount static files directory for logo, signatures, etc.
app.mount("/static", StaticFiles(directory="static"), name="static")

# Include API router with both endpoints
app.include_router(router)


@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "Legal Query Decomposition API is running"}


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
from dotenv import load_dotenv
import os
import logging

logger = logging.getLogger("settings")

# Load environment variables
load_dotenv()

# API keys
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
if not COHERE_API_KEY:
    logger.warning("COHERE_API_KEY not found in environment variables")

# Qdrant settings
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME", "LegalDocs")

if not QDRANT_URL or not QDRANT_API_KEY:
    logger.warning("Qdrant configuration missing in environment variables")


# Redis cache settings
UPSTASH_REDIS_REST_URL = os.getenv("UPSTASH_REDIS_REST_URL", "https://new-flamingo-55708.upstash.io")
UPSTASH_REDIS_REST_TOKEN = os.getenv("UPSTASH_REDIS_REST_TOKEN", "")
REDIS_CACHE_TTL = int(os.getenv("REDIS_CACHE_TTL", 3600))  # 1 hour default
REDIS_CACHE_ENABLED = os.getenv("REDIS_CACHE_ENABLED", "True").lower() == "true"

# App settings
APP_PORT = int(os.getenv("APP_PORT", 9005))
APP_HOST = os.getenv("APP_HOST", "0.0.0.0")

# CORS settings
# Parse comma-separated string of allowed origins
CORS_ALLOWED_ORIGINS = os.getenv("CORS_ALLOWED_ORIGINS", "http://localhost:3000").split(",")
CORS_ALLOW_CREDENTIALS = os.getenv("CORS_ALLOW_CREDENTIALS", "True").lower() == "true"

# Parse comma-separated string of allowed methods
CORS_ALLOWED_METHODS = os.getenv("CORS_ALLOWED_METHODS", "GET,POST,OPTIONS").split(",")

# Parse comma-separated string of allowed headers
CORS_ALLOWED_HEADERS = os.getenv("CORS_ALLOWED_HEADERS", "Content-Type,Authorization").split(",")

# Model settings
COHERE_MODEL = os.getenv("COHERE_MODEL", "command-r-08-2024")
DENSE_EMBEDDING_MODEL = os.getenv("DENSE_EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
SPARSE_EMBEDDING_MODEL = os.getenv("SPARSE_EMBEDDING_MODEL", "Qdrant/bm42-all-minilm-l6-v2-attentions")
RANKER_MODEL = os.getenv("RANKER_MODEL", "Xenova/ms-marco-MiniLM-L-6-v2")

# Retrieval settings
DEFAULT_TOP_K = int(os.getenv("DEFAULT_TOP_K", "5"))
DEFAULT_SCORE_THRESHOLD = float(os.getenv("DEFAULT_SCORE_THRESHOLD", "0.4"))
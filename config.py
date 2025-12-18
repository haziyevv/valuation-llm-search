import os
from dotenv import load_dotenv
load_dotenv()



AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-08-01-preview")
AZURE_OPENAI_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o")

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
WEBSEARCH_PLUS_API_KEY = os.getenv("WEBSEARCH_PLUS_API_KEY")
SERPER_API_KEY = os.getenv("SERPER_API_KEY")

# Redis configuration
REDIS_HOST = os.getenv("REDIS_HOST", "0.0.0.0")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", "password")
REDIS_URL = os.getenv("REDIS_URL", f"redis url")

# Cache configuration
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.8"))
CACHE_EXPIRY_DAYS = int(os.getenv("CACHE_EXPIRY_DAYS", "90"))

EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM = 1536

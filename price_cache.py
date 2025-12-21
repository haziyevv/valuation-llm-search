"""
Price Cache - Semantic Vector Database with Caching Service
============================================================
Uses Redis Stack to store and retrieve price predictions based on
semantic similarity of product descriptions.

Features:
- Stores predictions with embeddings of description + context
- Finds similar past predictions to avoid redundant LLM calls
- Configurable similarity threshold and cache expiry
- CachedPriceService wraps the agent with automatic cache lookup/storage
"""

import os
import json
import uuid
import logging
from datetime import datetime, timedelta
from typing import Optional

import redis
from redis.commands.search.field import VectorField, TextField, NumericField
from redis.commands.search.index_definition import IndexDefinition, IndexType
from redis.commands.search.query import Query
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# Configure logging
logger = logging.getLogger(__name__)

# Configuration
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.85"))
CACHE_EXPIRY_DAYS = int(os.getenv("CACHE_EXPIRY_DAYS", "90"))
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM = 1536


class PriceCache:
    """
    Semantic cache for price predictions using Redis Stack.
    
    Stores predictions indexed by product description + trade context.
    Retrieves similar past predictions to avoid redundant LLM calls.
    """
    
    INDEX_NAME = "price_predictions_idx"
    KEY_PREFIX = "price:"
    
    def __init__(self):
        """Initialize the cache with Redis."""
        self.client = redis.from_url(REDIS_URL, decode_responses=False)
        self.openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self._ensure_index()
    
    def _ensure_index(self):
        """Create vector search index if it doesn't exist."""
        try:
            self.client.ft(self.INDEX_NAME).info()
        except redis.ResponseError:
            # Index doesn't exist, create it
            schema = (
                TextField("search_text"),
                TextField("description"),
                TextField("country_of_origin"),
                TextField("unit_of_measure"),
                NumericField("unit_price"),
                TextField("currency"),
                TextField("timestamp"),
                TextField("prediction_json"),
                VectorField(
                    "embedding",
                    "FLAT",
                    {
                        "TYPE": "FLOAT32",
                        "DIM": EMBEDDING_DIM,
                        "DISTANCE_METRIC": "COSINE"
                    }
                )
            )
            
            definition = IndexDefinition(
                prefix=[self.KEY_PREFIX],
                index_type=IndexType.HASH
            )
            
            self.client.ft(self.INDEX_NAME).create_index(
                schema,
                definition=definition
            )
    
    def _get_embedding(self, text: str) -> list[float]:
        """Get embedding from OpenAI."""
        response = self.openai.embeddings.create(
            model=EMBEDDING_MODEL,
            input=text
        )
        return response.data[0].embedding
    
    def _create_search_text(
        self,
        description: str,
        country_of_origin: str,
        unit_of_measure: str
    ) -> str:
        """Create searchable text combining description and trade context."""
        return f"{description} | Origin: {country_of_origin} | Unit: {unit_of_measure}"
    
    def _is_expired(self, timestamp_str: str) -> bool:
        """Check if a cache entry has expired."""
        try:
            cached_date = datetime.fromisoformat(timestamp_str)
            expiry_date = cached_date + timedelta(days=CACHE_EXPIRY_DAYS)
            return datetime.now() > expiry_date
        except (ValueError, TypeError):
            return True
    
    def search(
        self,
        description: str,
        country_of_origin: str,
        unit_of_measure: str,
        n_results: int = 1
    ) -> Optional[dict]:
        """
        Search for a similar cached prediction.
        
        Returns:
            Cached prediction dict if found and similar enough, None otherwise
        """
        search_text = self._create_search_text(
            description, country_of_origin, unit_of_measure
        )
        
        embedding = self._get_embedding(search_text)
        embedding_bytes = np.array(embedding, dtype=np.float32).tobytes()
        
        # KNN vector search query
        query = (
            Query(f"*=>[KNN {n_results} @embedding $vector AS score]")
            .sort_by("score")
            .return_fields("search_text", "description", "timestamp", "prediction_json", "score")
            .dialect(2)
        )
        
        try:
            results = self.client.ft(self.INDEX_NAME).search(
                query,
                query_params={"vector": embedding_bytes}
            )
        except redis.ResponseError:
            return None
        
        if not results.docs:
            return None
        
        top_result = results.docs[0]
        # Redis returns distance for COSINE, convert to similarity (1 - distance)
        distance = float(top_result.score)
        similarity = 1 - distance
        
        if similarity < SIMILARITY_THRESHOLD:
            return None
        
        timestamp = top_result.timestamp.decode() if isinstance(top_result.timestamp, bytes) else top_result.timestamp
        
        # Check if expired
        if self._is_expired(timestamp):
            # Delete expired entry
            self.client.delete(top_result.id)
            return None
        
        prediction_json = top_result.prediction_json
        if isinstance(prediction_json, bytes):
            prediction_json = prediction_json.decode()
        
        search_text_result = top_result.search_text
        if isinstance(search_text_result, bytes):
            search_text_result = search_text_result.decode()
        
        return {
            "prediction": json.loads(prediction_json),
            "similarity": similarity,
            "cached_at": timestamp,
            "original_description": search_text_result,
            "cache_hit": True
        }
    
    def store(
        self,
        description: str,
        country_of_origin: str,
        unit_of_measure: str,
        prediction: dict
    ) -> str:
        """
        Store a prediction in the cache.
        
        Returns:
            The ID of the stored entry
        """
        search_text = self._create_search_text(
            description, country_of_origin, unit_of_measure
        )
        
        embedding = self._get_embedding(search_text)
        embedding_bytes = np.array(embedding, dtype=np.float32).tobytes()
        doc_id = f"{self.KEY_PREFIX}{uuid.uuid4()}"
        
        self.client.hset(
            doc_id,
            mapping={
                "search_text": search_text,
                "description": description,
                "country_of_origin": country_of_origin,
                "unit_of_measure": unit_of_measure,
                "unit_price": prediction.get("unit_price") or 0,
                "currency": prediction.get("currency", ""),
                "timestamp": datetime.now().isoformat(),
                "prediction_json": json.dumps(prediction),
                "embedding": embedding_bytes
            }
        )
        
        # Set TTL for auto-expiry
        self.client.expire(doc_id, timedelta(days=CACHE_EXPIRY_DAYS))
        
        return doc_id
    
    def get_stats(self) -> dict:
        """Get cache statistics."""
        try:
            info = self.client.ft(self.INDEX_NAME).info()
            total_entries = int(info.get("num_docs", 0))
        except redis.ResponseError:
            total_entries = 0
        
        return {
            "total_entries": total_entries,
            "similarity_threshold": SIMILARITY_THRESHOLD,
            "expiry_days": CACHE_EXPIRY_DAYS
        }
    
    def clear(self) -> int:
        """Clear all entries from this index only. Returns count of deleted entries."""
        count = 0
        
        try:
            batch_size = 100
            
            while True:
                # Always query from offset 0 since we're deleting as we go
                query = Query("*").paging(0, batch_size)
                results = self.client.ft(self.INDEX_NAME).search(query)
                
                if not results.docs:
                    break
                
                # Delete only documents belonging to this index
                doc_ids = [doc.id for doc in results.docs]
                if doc_ids:
                    self.client.delete(*doc_ids)
                    count += len(doc_ids)
                    
        except redis.ResponseError:
            # Index doesn't exist, nothing to clear
            pass
        
        return count


class CachedPriceService:
    """
    Price research service with Redis vector caching.
    
    Wraps the PriceResearchAgent with automatic cache lookup/storage:
    1. On request, search for similar items in Redis vector DB
    2. If similarity >= THRESHOLD, return cached result
    3. Otherwise, call the agent and cache the response
    """
    
    def __init__(self, cache: Optional[PriceCache] = None, agent=None):
        """
        Initialize the cached price service.
        
        Args:
            cache: PriceCache instance (optional, creates one if not provided)
            agent: PriceResearchAgent instance (optional, creates one if not provided)
        """
        self.cache = cache or get_cache()
        
        # Lazy import to avoid circular dependency
        if agent is None:
            from price_agent import PriceResearchAgent
            self.agent = PriceResearchAgent()
        else:
            self.agent = agent
    
    def get_price(
        self,
        description: str,
        country_of_origin: str,
        unit_of_measure: str,
        quantity: float,
        preferred_currency: Optional[str] = None,
    ) -> dict:
        """
        Get price prediction with caching.
        
        First checks Redis cache for similar items. If found with similarity
        above threshold, returns cached result. Otherwise calls the agent
        and caches the response.
        
        Args:
            description: Product/goods description
            country_of_origin: ISO3166 country code for origin
            unit_of_measure: Unit of measure (kg, tonne, etc.)
            quantity: Quantity of goods
            preferred_currency: Preferred currency code (optional)
            
        Returns:
            Dict containing the prediction and cache metadata:
            {
                "prediction": {...},  # The price prediction
                "cache_hit": bool,    # Whether result came from cache
                "similarity": float,  # Similarity score (if cache hit)
                "cached_at": str,     # Timestamp (if cache hit)
            }
        """
        # Step 1: Check cache for similar item (COO-based agnostic)
        logger.info(f"Checking cache for: {description[:50]}...")
        
        cached_result = self.cache.search(
            description=description,
            country_of_origin=country_of_origin,
            unit_of_measure=unit_of_measure,
        )
        
        if cached_result is not None:
            # Cache hit! Return cached prediction
            logger.info(
                f"Cache HIT! Similarity: {cached_result['similarity']:.2%}, "
                f"cached_at: {cached_result['cached_at']}"
            )
            return {
                "prediction": cached_result["prediction"],
                "cache_hit": True,
                "similarity": cached_result["similarity"],
                "cached_at": cached_result["cached_at"],
                "original_description": cached_result.get("original_description"),
            }
        
        # Step 2: Cache miss - call the agent
        logger.info("Cache MISS - calling agent for price research...")
        
        agent_result = self.agent.research_price(
            country_of_origin=country_of_origin,
            description=description,
            quantity=quantity,
            unit_of_measure=unit_of_measure,
            preferred_currency=preferred_currency,
        )
        
        # Convert to dict for storage
        prediction_dict = self.agent.to_dict(agent_result)
        
        # Step 3: Store in cache (only if we got a valid result)
        # if agent_result.error is None and agent_result.unit_price is not None:
        #     doc_id = self.cache.store(
        #         description=description,
        #         country_of_origin=country_of_origin,
        #         unit_of_measure=unit_of_measure,
        #         prediction=prediction_dict,
        #     )
        #     logger.info(f"Stored new prediction in cache with ID: {doc_id}")
        # else:
        #     logger.warning(
        #         f"Not caching result due to error or missing price: "
        #         f"error={agent_result.error}, unit_price={agent_result.unit_price}"
        #     )
        
        return {
            "prediction": prediction_dict,
            "cache_hit": False,
            "similarity": None,
            "cached_at": None,
        }
    
    def get_stats(self) -> dict:
        """Get cache statistics."""
        return self.cache.get_stats()
    
    def clear_cache(self) -> int:
        """Clear all cached entries."""
        return self.cache.clear()


# Singleton instances
_cache_instance: Optional[PriceCache] = None
_service_instance: Optional[CachedPriceService] = None


def get_cache() -> PriceCache:
    """Get or create the singleton cache instance."""
    global _cache_instance
    if _cache_instance is None:
        _cache_instance = PriceCache()
    return _cache_instance


def get_cached_price_service() -> CachedPriceService:
    """Get or create the singleton cached price service instance."""
    global _service_instance
    if _service_instance is None:
        _service_instance = CachedPriceService()
    return _service_instance


if __name__ == "__main__":
    import sys
    
    # Check command line args for test mode
    if len(sys.argv) > 1 and sys.argv[1] == "--test-cache":
        # Test cache directly without agent
        cache = PriceCache()
        
        test_prediction = {
            "unit_price": 1.25,
            "currency": "USD",
            "unit_of_measure": "kg",
            "confidence": 0.85,
            "notes": "Test prediction"
        }
        
        doc_id = cache.store(
            description="LOW DENSITY POLYETHYLENE (LDPE) LOTRENE MG70",
            country_of_origin="QA",
            unit_of_measure="kg",
            prediction=test_prediction
        )
        print(f"Stored prediction with ID: {doc_id}")
        
        result = cache.search(
            description="LDPE Polyethylene Lotrene",
            country_of_origin="QA",
            unit_of_measure="kg"
        )
        
        if result:
            print(f"Cache HIT! Similarity: {result['similarity']:.2%}")
            print(f"Cached price: {result['prediction']['unit_price']} {result['prediction']['currency']}")
        else:
            print("Cache MISS - no similar entry found")
        
        print(f"\nCache stats: {cache.get_stats()}")
    
    else:
        # Test the full cached service
        print("=" * 60)
        print("Testing CachedPriceService")
        print("=" * 60)
        
        service = get_cached_price_service()
        
        # First call - should be a cache miss, will call agent
        print("\n--- First call (expecting cache MISS) ---")
        result1 = service.get_price(
            description="LOW DENSITY POLYETHYLENE (LDPE) LOTRENE MG70",
            country_of_origin="QA",
            unit_of_measure="kg",
            quantity=5000,
        )
        
        print(f"Cache hit: {result1['cache_hit']}")
        if result1['prediction'].get('unit_price'):
            print(f"Price: {result1['prediction']['unit_price']:.2f} {result1['prediction'].get('currency', 'USD')}")
        
        # Second call with similar description - should be a cache hit
        print("\n--- Second call with similar description (expecting cache HIT) ---")
        result2 = service.get_price(
            description="LDPE Polyethylene Lotrene MG70 resin",
            country_of_origin="QA",
            unit_of_measure="kg",
            quantity=5000,
        )
        
        print(f"Cache hit: {result2['cache_hit']}")
        if result2['cache_hit']:
            print(f"Similarity: {result2['similarity']:.2%}")
            print(f"Cached at: {result2['cached_at']}")
        if result2['prediction'].get('unit_price'):
            print(f"Price: {result2['prediction']['unit_price']:.2f} {result2['prediction'].get('currency', 'USD')}")
        
        print(f"\nCache stats: {service.get_stats()}")


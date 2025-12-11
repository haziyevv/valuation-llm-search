"""
Price Cache - Semantic Vector Database
=======================================
Uses Redis Stack to store and retrieve price predictions based on
semantic similarity of product descriptions.

Features:
- Stores predictions with embeddings of description + context
- Finds similar past predictions to avoid redundant LLM calls
- Configurable similarity threshold and cache expiry
"""

import os
import json
import uuid
from datetime import datetime, timedelta
from typing import Optional

import redis
from redis.commands.search.field import VectorField, TextField, NumericField
from redis.commands.search.index import IndexDefinition, IndexType
from redis.commands.search.query import Query
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

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
                TextField("country_of_destination"),
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
        country_of_destination: str,
        unit_of_measure: str
    ) -> str:
        """Create searchable text combining description and trade context."""
        return f"{description} | Origin: {country_of_origin} | Destination: {country_of_destination} | Unit: {unit_of_measure}"
    
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
        country_of_destination: str,
        unit_of_measure: str,
        n_results: int = 1
    ) -> Optional[dict]:
        """
        Search for a similar cached prediction.
        
        Returns:
            Cached prediction dict if found and similar enough, None otherwise
        """
        search_text = self._create_search_text(
            description, country_of_origin, country_of_destination, unit_of_measure
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
        country_of_destination: str,
        unit_of_measure: str,
        prediction: dict
    ) -> str:
        """
        Store a prediction in the cache.
        
        Returns:
            The ID of the stored entry
        """
        search_text = self._create_search_text(
            description, country_of_origin, country_of_destination, unit_of_measure
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
                "country_of_destination": country_of_destination,
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
        """Clear all entries from the cache. Returns count of deleted entries."""
        count = 0
        cursor = 0
        
        while True:
            cursor, keys = self.client.scan(cursor, match=f"{self.KEY_PREFIX}*", count=100)
            if keys:
                count += len(keys)
                self.client.delete(*keys)
            if cursor == 0:
                break
        
        return count


# Singleton instance
_cache_instance: Optional[PriceCache] = None


def get_cache() -> PriceCache:
    """Get or create the singleton cache instance."""
    global _cache_instance
    if _cache_instance is None:
        _cache_instance = PriceCache()
    return _cache_instance


if __name__ == "__main__":
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
        country_of_destination="PK",
        unit_of_measure="kg",
        prediction=test_prediction
    )
    print(f"Stored prediction with ID: {doc_id}")
    
    result = cache.search(
        description="LDPE Polyethylene Lotrene",
        country_of_origin="QA",
        country_of_destination="PK",
        unit_of_measure="kg"
    )
    
    if result:
        print(f"Cache HIT! Similarity: {result['similarity']:.2%}")
        print(f"Cached price: {result['prediction']['unit_price']} {result['prediction']['currency']}")
    else:
        print("Cache MISS - no similar entry found")
    
    print(f"\nCache stats: {cache.get_stats()}")

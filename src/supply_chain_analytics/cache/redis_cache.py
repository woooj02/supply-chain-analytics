"""
Redis caching layer for high-performance data access.
Implements caching strategies, TTL management, and cache invalidation.
"""
import asyncio
import hashlib
import json
from datetime import datetime, timedelta
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Union
import pickle

import redis.asyncio as aioredis
from redis.asyncio import Redis
from loguru import logger

from config.settings import settings
from supply_chain_analytics.core.logger import LoggerSetup


class RedisCache:
    """
    Redis cache manager with:
    - Async operations
    - Connection pooling
    - Multiple serialization formats
    - TTL management
    - Cache statistics
    - Circuit breaker pattern
    """
    
    _instance = None
    _client: Optional[Redis] = None
    _enabled: bool = True
    _failure_count: int = 0
    _failure_threshold: int = 5
    _circuit_open: bool = False
    _circuit_reset_time: Optional[datetime] = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    @classmethod
    async def initialize(cls) -> None:
        """Initialize Redis connection pool."""
        if cls._client is not None:
            return
        
        try:
            cls._client = aioredis.from_url(
                settings.redis.connection_url,
                encoding="utf-8",
                decode_responses=False,
                max_connections=20,
                socket_timeout=5,
                socket_connect_timeout=5,
                retry_on_timeout=True,
                health_check_interval=30,
            )
            
            # Test connection
            await cls._client.ping()
            logger.info("Redis cache connected successfully")
            
        except Exception as e:
            logger.warning(f"Redis connection failed, caching disabled: {e}")
            cls._enabled = False
            cls._client = None
    
    @classmethod
    async def close(cls) -> None:
        """Close Redis connection."""
        if cls._client:
            await cls._client.close()
            cls._client = None
            logger.info("Redis cache connection closed")
    
    @classmethod
    def _check_circuit(cls) -> bool:
        """Check if circuit breaker is open."""
        if not cls._circuit_open:
            return True
        
        if cls._circuit_reset_time and datetime.utcnow() > cls._circuit_reset_time:
            cls._circuit_open = False
            cls._failure_count = 0
            logger.info("Circuit breaker reset")
            return True
        
        return False
    
    @classmethod
    async def get(
        cls,
        key: str,
        deserialize: str = "json",
        default: Any = None,
    ) -> Optional[Any]:
        """Retrieve value from supply_chain_analytics.cache."""
        if not cls._enabled or not cls._client or not cls._check_circuit():
            return default
        
        try:
            value = await cls._client.get(key)
            
            if value is None:
                return default
            
            if deserialize == "json":
                return json.loads(value)
            elif deserialize == "pickle":
                return pickle.loads(value)
            else:
                return value.decode("utf-8") if isinstance(value, bytes) else value
        
        except Exception as e:
            logger.warning(f"Cache get error for key '{key}': {e}")
            cls._failure_count += 1
            if cls._failure_count >= cls._failure_threshold:
                cls._circuit_open = True
                cls._circuit_reset_time = datetime.utcnow() + timedelta(seconds=60)
                logger.error("Circuit breaker opened")
            return default
    
    @classmethod
    async def set(
        cls,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
        serialize: str = "json",
        nx: bool = False,
    ) -> bool:
        """Store value in cache."""
        if not cls._enabled or not cls._client or not cls._check_circuit():
            return False
        
        try:
            if serialize == "json":
                data = json.dumps(value, default=str)
            elif serialize == "pickle":
                data = pickle.dumps(value)
            else:
                data = str(value)
            
            await cls._client.set(key, data, ex=ttl, nx=nx)
            return True
        
        except Exception as e:
            logger.warning(f"Cache set error for key '{key}': {e}")
            return False
    
    @classmethod
    async def delete(cls, *keys: str) -> int:
        """Delete keys from supply_chain_analytics.cache."""
        if not cls._enabled or not cls._client:
            return 0
        
        try:
            return await cls._client.delete(*keys)
        except Exception as e:
            logger.warning(f"Cache delete error: {e}")
            return 0
    
    @classmethod
    async def exists(cls, *keys: str) -> int:
        """Check if keys exist."""
        if not cls._enabled or not cls._client:
            return 0
        
        try:
            return await cls._client.exists(*keys)
        except Exception:
            return 0
    
    @classmethod
    async def increment(cls, key: str, amount: int = 1) -> Optional[int]:
        """Increment a counter."""
        if not cls._enabled or not cls._client:
            return None
        
        try:
            return await cls._client.incrby(key, amount)
        except Exception:
            return None
    
    @classmethod
    async def get_or_set(
        cls,
        key: str,
        factory: Callable,
        ttl: int = 300,
        serialize: str = "json",
    ) -> Any:
        """Get from cache or compute and store."""
        cached = await cls.get(key, deserialize=serialize)
        
        if cached is not None:
            return cached
        
        value = await factory() if asyncio.iscoroutinefunction(factory) else factory()
        
        if value is not None:
            await cls.set(key, value, ttl=ttl, serialize=serialize)
        
        return value
    
    @classmethod
    async def invalidate_pattern(cls, pattern: str) -> int:
        """Invalidate all keys matching a pattern."""
        if not cls._enabled or not cls._client:
            return 0
        
        try:
            keys = []
            async for key in cls._client.scan_iter(match=pattern):
                keys.append(key)
            
            if keys:
                return await cls._client.delete(*keys)
            return 0
        except Exception as e:
            logger.warning(f"Cache pattern invalidation error: {e}")
            return 0
    
    @classmethod
    async def get_stats(cls) -> Dict[str, Any]:
        """Get cache statistics."""
        if not cls._enabled or not cls._client:
            return {"enabled": False}
        
        try:
            info = await cls._client.info()
            return {
                "enabled": True,
                "connected_clients": info.get("connected_clients", 0),
                "used_memory_human": info.get("used_memory_human", "0"),
                "uptime_in_seconds": info.get("uptime_in_seconds", 0),
                "keyspace_hits": info.get("keyspace_hits", 0),
                "keyspace_misses": info.get("keyspace_misses", 0),
                "hit_ratio": cls._calculate_hit_ratio(info),
                "circuit_open": cls._circuit_open,
                "failure_count": cls._failure_count,
            }
        except Exception:
            return {"enabled": True, "error": "Failed to get stats"}
    
    @classmethod
    def _calculate_hit_ratio(cls, info: Dict) -> float:
        """Calculate cache hit ratio."""
        hits = int(info.get("keyspace_hits", 0))
        misses = int(info.get("keyspace_misses", 0))
        total = hits + misses
        if total > 0:
            return round(hits / total * 100, 2)
        return 0.0


def cached(
    ttl: int = 300,
    key_prefix: str = "",
    serialize: str = "json",
):
    """Decorator for caching function results."""
    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            if not RedisCache._enabled:
                return await func(*args, **kwargs)
            
            # Generate cache key
            key_parts = [key_prefix, func.__name__]
            key_parts.append(hashlib.md5(
                f"{args}{kwargs}".encode()
            ).hexdigest())
            cache_key = ":".join(key_parts)
            
            # Try cache
            result = await RedisCache.get(cache_key, deserialize=serialize)
            
            if result is not None:
                logger.debug(f"Cache hit: {cache_key}")
                return result
            
            # Execute function
            result = await func(*args, **kwargs)
            
            # Store in cache
            await RedisCache.set(cache_key, result, ttl=ttl, serialize=serialize)
            
            return result
        
        return wrapper
    return decorator


# Singleton instance
cache = RedisCache()
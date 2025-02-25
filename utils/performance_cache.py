import os
import json
import hashlib
import time
from pathlib import Path
from typing import Dict, Any, Optional, Union, Callable
from loguru import logger


class ProcessingCache:
    """
    Persistent cache for expensive operations like API calls and text processing.
    
    Features:
    - File-based persistence
    - TTL (time-to-live) support
    - Size limits and LRU eviction
    - Optional compression
    
    This cache can significantly reduce processing time and costs for batched operations
    by avoiding redundant calls for identical or similar inputs.
    """
    
    def __init__(
        self,
        cache_dir: Union[str, Path] = "cache",
        max_size: int = 1000,
        ttl: Optional[int] = None,  # Time to live in seconds (None = no expiration)
        compress: bool = False,
        namespace: str = "default"
    ):
        """
        Initialize the cache.
        
        Args:
            cache_dir: Directory for cache files
            max_size: Maximum number of entries in memory cache
            ttl: Time-to-live in seconds (None = no expiration)
            compress: Whether to compress cached data
            namespace: Namespace for cache isolation
        """
        self.cache_dir = Path(cache_dir) / namespace
        self.max_size = max_size
        self.ttl = ttl
        self.compress = compress
        self.memory_cache: Dict[str, Dict[str, Any]] = {}
        self.access_times: Dict[str, float] = {}
        
        # Initialize cache directory
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Cache initialized: {self.cache_dir} (max_size={max_size}, ttl={ttl})")

    def _get_cache_path(self, key: str) -> Path:
        """Get file path for a cache key."""
        hashed_key = hashlib.md5(key.encode('utf-8')).hexdigest()
        return self.cache_dir / f"{hashed_key}.json"

    def _is_expired(self, timestamp: float) -> bool:
        """Check if a cache entry is expired based on TTL."""
        if self.ttl is None:
            return False
        return (time.time() - timestamp) > self.ttl

    def _evict_if_needed(self) -> None:
        """Evict least recently used entries if cache exceeds max size."""
        if len(self.memory_cache) <= self.max_size:
            return
            
        # Sort by access time (oldest first)
        sorted_keys = sorted(self.access_times.items(), key=lambda x: x[1])
        
        # Remove oldest entries to get back to 75% capacity
        entries_to_remove = len(self.memory_cache) - int(self.max_size * 0.75)
        for key, _ in sorted_keys[:entries_to_remove]:
            self.memory_cache.pop(key, None)
            self.access_times.pop(key, None)
            
        logger.debug(f"Cache eviction: removed {entries_to_remove} entries")

    def get(self, key: str) -> Optional[Any]:
        """
        Get a value from the cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found or expired
        """
        # Try memory cache first
        if key in self.memory_cache:
            entry = self.memory_cache[key]
            
            # Check expiration
            if self._is_expired(entry.get('timestamp', 0)):
                self.memory_cache.pop(key, None)
                self.access_times.pop(key, None)
                logger.debug(f"Cache entry expired: {key}")
                return None
                
            # Update access time
            self.access_times[key] = time.time()
            logger.debug(f"Cache hit (memory): {key}")
            return entry.get('value')
            
        # Try disk cache
        cache_path = self._get_cache_path(key)
        if cache_path.exists():
            try:
                with open(cache_path, 'r', encoding='utf-8') as f:
                    entry = json.load(f)
                    
                # Check expiration
                if self._is_expired(entry.get('timestamp', 0)):
                    cache_path.unlink(missing_ok=True)
                    logger.debug(f"Cache entry expired (disk): {key}")
                    return None
                    
                # Add to memory cache
                self.memory_cache[key] = entry
                self.access_times[key] = time.time()
                
                # Evict if needed
                self._evict_if_needed()
                
                logger.debug(f"Cache hit (disk): {key}")
                return entry.get('value')
                
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"Error reading cache file {cache_path}: {str(e)}")
                
        logger.debug(f"Cache miss: {key}")
        return None

    def set(self, key: str, value: Any) -> None:
        """
        Store a value in the cache.
        
        Args:
            key: Cache key
            value: Value to store
        """
        # Create cache entry with timestamp
        entry = {
            'value': value,
            'timestamp': time.time()
        }
        
        # Update memory cache
        self.memory_cache[key] = entry
        self.access_times[key] = time.time()
        
        # Evict if needed
        self._evict_if_needed()
        
        # Write to disk cache
        cache_path = self._get_cache_path(key)
        try:
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(entry, f)
            logger.debug(f"Cache set: {key}")
        except IOError as e:
            logger.warning(f"Error writing cache file {cache_path}: {str(e)}")

    def invalidate(self, key: str) -> None:
        """
        Invalidate a specific cache entry.
        
        Args:
            key: Cache key to invalidate
        """
        # Remove from memory cache
        self.memory_cache.pop(key, None)
        self.access_times.pop(key, None)
        
        # Remove from disk cache
        cache_path = self._get_cache_path(key)
        cache_path.unlink(missing_ok=True)
        logger.debug(f"Cache invalidated: {key}")

    def clear(self) -> None:
        """Clear all cache entries."""
        # Clear memory cache
        self.memory_cache.clear()
        self.access_times.clear()
        
        # Clear disk cache
        for cache_file in self.cache_dir.glob('*.json'):
            cache_file.unlink(missing_ok=True)
        logger.info(f"Cache cleared: {self.cache_dir}")

    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        disk_entries = len(list(self.cache_dir.glob('*.json')))
        memory_entries = len(self.memory_cache)
        
        return {
            'memory_entries': memory_entries,
            'disk_entries': disk_entries,
            'max_size': self.max_size,
            'ttl': self.ttl,
            'cache_dir': str(self.cache_dir)
        }


def cached(
    cache_instance: ProcessingCache, 
    key_func: Optional[Callable] = None,
    ttl: Optional[int] = None
):
    """
    Decorator for caching function results.
    
    Args:
        cache_instance: ProcessingCache instance
        key_func: Function to generate cache key from args and kwargs
        ttl: Time-to-live override for this specific function
        
    Returns:
        Decorator function
    """
    def decorator(func):
        def _generate_key(*args, **kwargs):
            if key_func:
                return key_func(*args, **kwargs)
            
            # Default key generation
            key_parts = [func.__name__]
            
            # Add args
            for arg in args:
                if isinstance(arg, (str, int, float, bool)):
                    key_parts.append(str(arg))
                else:
                    # For complex objects, use their hash or repr
                    key_parts.append(str(hash(str(arg))))
            
            # Add kwargs (sorted for consistency)
            for k in sorted(kwargs.keys()):
                v = kwargs[k]
                if isinstance(v, (str, int, float, bool)):
                    key_parts.append(f"{k}={v}")
                else:
                    key_parts.append(f"{k}={hash(str(v))}")
            
            return ":".join(key_parts)
        
        async def async_wrapper(*args, **kwargs):
            # Generate cache key
            key = _generate_key(*args, **kwargs)
            
            # Try to get from cache
            cached_result = cache_instance.get(key)
            if cached_result is not None:
                return cached_result
            
            # Not in cache, call function
            result = await func(*args, **kwargs)
            
            # Store in cache
            cache_instance.set(key, result)
            
            return result
            
        def sync_wrapper(*args, **kwargs):
            # Generate cache key
            key = _generate_key(*args, **kwargs)
            
            # Try to get from cache
            cached_result = cache_instance.get(key)
            if cached_result is not None:
                return cached_result
            
            # Not in cache, call function
            result = func(*args, **kwargs)
            
            # Store in cache
            cache_instance.set(key, result)
            
            return result
        
        # Return appropriate wrapper based on function type
        import inspect
        if inspect.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
            
    return decorator
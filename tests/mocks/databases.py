"""
Mock databases and caches for testing.

These mocks provide in-memory storage without requiring
actual database or cache infrastructure.
"""

from typing import Dict, Any, Optional, List
from pathlib import Path
from datetime import datetime, timedelta
import json


class MockDatabase:
    """Mock database for testing."""
    
    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path
        self._data: Dict[str, Dict[str, Any]] = {}
        self._query_count = 0
        self._should_fail = False
        
    def connect(self):
        """Mock connection."""
        if self._should_fail:
            raise ConnectionError("Mock database connection failed")
        return self
        
    def disconnect(self):
        """Mock disconnection."""
        pass
        
    def insert(self, table: str, data: Dict[str, Any]) -> int:
        """Insert data into table."""
        self._query_count += 1
        
        if table not in self._data:
            self._data[table] = {}
            
        # Generate ID
        record_id = len(self._data[table]) + 1
        data["id"] = record_id
        data["created_at"] = datetime.now()
        
        self._data[table][record_id] = data
        return record_id
        
    def select(self, table: str, filters: Optional[Dict] = None) -> List[Dict]:
        """Select data from table."""
        self._query_count += 1
        
        if table not in self._data:
            return []
            
        results = list(self._data[table].values())
        
        # Apply filters
        if filters:
            filtered = []
            for record in results:
                match = all(
                    record.get(key) == value
                    for key, value in filters.items()
                )
                if match:
                    filtered.append(record)
            results = filtered
            
        return results
        
    def update(self, table: str, record_id: int, data: Dict[str, Any]) -> bool:
        """Update record."""
        self._query_count += 1
        
        if table not in self._data or record_id not in self._data[table]:
            return False
            
        self._data[table][record_id].update(data)
        self._data[table][record_id]["updated_at"] = datetime.now()
        return True
        
    def delete(self, table: str, record_id: int) -> bool:
        """Delete record."""
        self._query_count += 1
        
        if table not in self._data or record_id not in self._data[table]:
            return False
            
        del self._data[table][record_id]
        return True
        
    def clear(self, table: Optional[str] = None):
        """Clear data."""
        if table:
            self._data[table] = {}
        else:
            self._data = {}
            
    def get_query_count(self) -> int:
        """Get number of queries executed."""
        return self._query_count
        
    def reset_query_count(self):
        """Reset query counter."""
        self._query_count = 0
        
    def set_should_fail(self, should_fail: bool):
        """Configure whether database should fail."""
        self._should_fail = should_fail


class MockCache:
    """Mock cache for testing."""
    
    def __init__(self, default_ttl: int = 300):
        self.default_ttl = default_ttl
        self._data: Dict[str, Any] = {}
        self._expiry: Dict[str, datetime] = {}
        self._hit_count = 0
        self._miss_count = 0
        
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        # Check expiry
        if key in self._expiry:
            if datetime.now() > self._expiry[key]:
                del self._data[key]
                del self._expiry[key]
                self._miss_count += 1
                return None
                
        if key in self._data:
            self._hit_count += 1
            return self._data[key]
            
        self._miss_count += 1
        return None
        
    def set(self, key: str, value: Any, ttl: Optional[int] = None):
        """Set value in cache."""
        self._data[key] = value
        
        ttl = ttl or self.default_ttl
        self._expiry[key] = datetime.now() + timedelta(seconds=ttl)
        
    def delete(self, key: str) -> bool:
        """Delete key from cache."""
        if key in self._data:
            del self._data[key]
            if key in self._expiry:
                del self._expiry[key]
            return True
        return False
        
    def clear(self):
        """Clear all cache."""
        self._data = {}
        self._expiry = {}
        
    def exists(self, key: str) -> bool:
        """Check if key exists."""
        return self.get(key) is not None
        
    def get_hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self._hit_count + self._miss_count
        if total == 0:
            return 0.0
        return self._hit_count / total
        
    def get_stats(self) -> Dict[str, int]:
        """Get cache statistics."""
        return {
            "hits": self._hit_count,
            "misses": self._miss_count,
            "size": len(self._data),
            "hit_rate": self.get_hit_rate()
        }
        
    def reset_stats(self):
        """Reset statistics."""
        self._hit_count = 0
        self._miss_count = 0


class MockRedisCache(MockCache):
    """Mock Redis cache with additional Redis-specific methods."""
    
    def incr(self, key: str, amount: int = 1) -> int:
        """Increment counter."""
        current = self.get(key) or 0
        new_value = current + amount
        self.set(key, new_value)
        return new_value
        
    def decr(self, key: str, amount: int = 1) -> int:
        """Decrement counter."""
        return self.incr(key, -amount)
        
    def lpush(self, key: str, *values):
        """Push to list (left)."""
        current = self.get(key) or []
        current = list(values) + current
        self.set(key, current)
        
    def rpush(self, key: str, *values):
        """Push to list (right)."""
        current = self.get(key) or []
        current.extend(values)
        self.set(key, current)
        
    def lrange(self, key: str, start: int, end: int) -> List:
        """Get range from list."""
        current = self.get(key) or []
        if end == -1:
            return current[start:]
        return current[start:end+1]


class MockModelRegistry:
    """Mock model registry for testing."""
    
    def __init__(self, storage_path: Optional[Path] = None):
        self.storage_path = storage_path
        self._models: Dict[str, Dict[str, Any]] = {}
        
    def register_model(
        self,
        name: str,
        version: str,
        metadata: Dict[str, Any]
    ) -> str:
        """Register a model."""
        model_id = f"{name}:{version}"
        self._models[model_id] = {
            "name": name,
            "version": version,
            "metadata": metadata,
            "registered_at": datetime.now()
        }
        return model_id
        
    def get_model(self, name: str, version: Optional[str] = None) -> Optional[Dict]:
        """Get model metadata."""
        if version:
            model_id = f"{name}:{version}"
            return self._models.get(model_id)
            
        # Get latest version
        matching = [
            m for m in self._models.values()
            if m["name"] == name
        ]
        if not matching:
            return None
            
        return max(matching, key=lambda m: m["registered_at"])
        
    def list_models(self, name: Optional[str] = None) -> List[Dict]:
        """List all models."""
        if name:
            return [
                m for m in self._models.values()
                if m["name"] == name
            ]
        return list(self._models.values())
        
    def delete_model(self, name: str, version: str) -> bool:
        """Delete a model."""
        model_id = f"{name}:{version}"
        if model_id in self._models:
            del self._models[model_id]
            return True
        return False

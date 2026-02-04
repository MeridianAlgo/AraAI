"""
Horizontal Scaling Support
Stateless API design, distributed caching, worker pools, and service discovery
"""

import json
import hashlib
from typing import Dict, Any, Optional, List, Callable
from datetime import datetime
import threading

from ara.utils import get_logger

logger = get_logger(__name__)


class StatelessAPIHandler:
    """
    Stateless API request handler for load balancing
    Ensures no session state is stored on individual servers
    """

    def __init__(self, session_store=None):
        """
        Initialize stateless API handler

        Args:
            session_store: External session store (Redis, etc.)
        """
        self.session_store = session_store
        logger.info("Initialized StatelessAPIHandler")

    def create_request_context(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create request context from incoming request
        All state is derived from request data

        Args:
            request_data: Request data

        Returns:
            Request context
        """
        context = {
            "request_id": self._generate_request_id(request_data),
            "timestamp": datetime.now().isoformat(),
            "data": request_data,
        }

        return context

    def _generate_request_id(self, request_data: Dict[str, Any]) -> str:
        """Generate unique request ID"""
        data_str = json.dumps(request_data, sort_keys=True)
        timestamp = datetime.now().isoformat()
        combined = f"{data_str}:{timestamp}"
        return hashlib.sha256(combined.encode()).hexdigest()[:16]

    def store_session_data(self, session_id: str, data: Dict[str, Any]) -> bool:
        """
        Store session data in external store

        Args:
            session_id: Session identifier
            data: Session data

        Returns:
            True if successful
        """
        if self.session_store is None:
            logger.warning("No session store configured")
            return False

        try:
            self.session_store.set(f"session:{session_id}", data, ttl=3600)
            return True
        except Exception as e:
            logger.error(f"Failed to store session data: {e}")
            return False

    def get_session_data(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve session data from external store

        Args:
            session_id: Session identifier

        Returns:
            Session data or None
        """
        if self.session_store is None:
            return None

        try:
            return self.session_store.get(f"session:{session_id}")
        except Exception as e:
            logger.error(f"Failed to get session data: {e}")
            return None


class DistributedCache:
    """
    Distributed caching with Redis Cluster support
    """

    def __init__(
        self,
        cluster_nodes: Optional[List[Dict[str, Any]]] = None,
        password: Optional[str] = None,
    ):
        """
        Initialize distributed cache

        Args:
            cluster_nodes: List of cluster node configs
            password: Redis password
        """
        self.cluster_nodes = cluster_nodes or [{"host": "localhost", "port": 6379}]
        self.password = password
        self._client = None
        self._is_cluster = len(self.cluster_nodes) > 1

        self._initialize_client()

        logger.info(
            "Initialized DistributedCache",
            cluster_mode=self._is_cluster,
            nodes=len(self.cluster_nodes),
        )

    def _initialize_client(self):
        """Initialize Redis client or cluster"""
        try:
            import redis
            from redis.cluster import RedisCluster

            if self._is_cluster:
                # Redis Cluster mode
                startup_nodes = [
                    redis.cluster.ClusterNode(node["host"], node["port"])
                    for node in self.cluster_nodes
                ]

                self._client = RedisCluster(
                    startup_nodes=startup_nodes,
                    password=self.password,
                    decode_responses=False,
                    skip_full_coverage_check=True,
                )
                logger.info("Connected to Redis Cluster")
            else:
                # Single node mode
                node = self.cluster_nodes[0]
                self._client = redis.Redis(
                    host=node["host"],
                    port=node["port"],
                    password=self.password,
                    decode_responses=False,
                )
                logger.info(f"Connected to Redis at {node['host']}:{node['port']}")

            # Test connection
            self._client.ping()

        except ImportError:
            logger.error("redis-py not installed. Install with: pip install redis")
            self._client = None
        except Exception as e:
            logger.error(f"Failed to initialize Redis client: {e}")
            self._client = None

    def get(self, key: str) -> Optional[Any]:
        """Get value from distributed cache"""
        if self._client is None:
            return None

        try:
            import pickle

            value = self._client.get(key)
            return pickle.loads(value) if value else None
        except Exception as e:
            logger.warning(f"Distributed cache get error: {e}")
            return None

    def set(self, key: str, value: Any, ttl: int = 3600) -> bool:
        """Set value in distributed cache"""
        if self._client is None:
            return False

        try:
            import pickle

            serialized = pickle.dumps(value)
            self._client.setex(key, ttl, serialized)
            return True
        except Exception as e:
            logger.warning(f"Distributed cache set error: {e}")
            return False

    def delete(self, key: str) -> bool:
        """Delete key from distributed cache"""
        if self._client is None:
            return False

        try:
            return self._client.delete(key) > 0
        except Exception as e:
            logger.warning(f"Distributed cache delete error: {e}")
            return False

    def get_cluster_info(self) -> Dict[str, Any]:
        """Get cluster information"""
        if self._client is None:
            return {"connected": False}

        try:
            if self._is_cluster:
                info = self._client.cluster_info()
                nodes = self._client.cluster_nodes()
                return {
                    "connected": True,
                    "cluster_mode": True,
                    "cluster_info": info,
                    "nodes": nodes,
                }
            else:
                info = self._client.info()
                return {"connected": True, "cluster_mode": False, "info": info}
        except Exception as e:
            logger.error(f"Failed to get cluster info: {e}")
            return {"connected": False, "error": str(e)}


class WorkerPool:
    """
    Worker pool for async task processing (Celery-compatible)
    """

    def __init__(
        self,
        broker_url: str = "redis://localhost:6379/0",
        backend_url: str = "redis://localhost:6379/1",
        max_workers: int = 4,
    ):
        """
        Initialize worker pool

        Args:
            broker_url: Message broker URL
            backend_url: Result backend URL
            max_workers: Maximum number of workers
        """
        self.broker_url = broker_url
        self.backend_url = backend_url
        self.max_workers = max_workers
        self._app = None
        self._tasks: Dict[str, Callable] = {}

        self._initialize_celery()

        logger.info("Initialized WorkerPool", broker=broker_url, max_workers=max_workers)

    def _initialize_celery(self):
        """Initialize Celery app"""
        try:
            from celery import Celery

            self._app = Celery("ara_worker", broker=self.broker_url, backend=self.backend_url)

            # Configure Celery
            self._app.conf.update(
                task_serializer="json",
                accept_content=["json"],
                result_serializer="json",
                timezone="UTC",
                enable_utc=True,
                worker_prefetch_multiplier=1,
                worker_max_tasks_per_child=1000,
            )

            logger.info("Initialized Celery app")

        except ImportError:
            logger.warning(
                "Celery not installed. Worker pool will use threading. "
                "Install with: pip install celery"
            )
            self._app = None

    def register_task(self, name: str, func: Callable) -> None:
        """
        Register a task

        Args:
            name: Task name
            func: Task function
        """
        if self._app:
            # Register with Celery
            self._app.task(name=name)(func)

        self._tasks[name] = func
        logger.info(f"Registered task: {name}")

    def submit_task(self, task_name: str, *args, **kwargs) -> Optional[str]:
        """
        Submit a task for execution

        Args:
            task_name: Name of registered task
            *args: Task arguments
            **kwargs: Task keyword arguments

        Returns:
            Task ID or None
        """
        if task_name not in self._tasks:
            logger.error(f"Task not registered: {task_name}")
            return None

        if self._app:
            # Submit to Celery
            try:
                result = self._app.send_task(task_name, args=args, kwargs=kwargs)
                logger.info(f"Submitted task {task_name}: {result.id}")
                return result.id
            except Exception as e:
                logger.error(f"Failed to submit task: {e}")
                return None
        else:
            # Fallback to threading
            import threading

            def run_task():
                try:
                    func = self._tasks[task_name]
                    func(*args, **kwargs)
                except Exception as e:
                    logger.error(f"Task execution failed: {e}")

            thread = threading.Thread(target=run_task)
            thread.start()

            task_id = f"thread_{threading.get_ident()}"
            logger.info(f"Submitted task {task_name} to thread: {task_id}")
            return task_id

    def get_task_result(self, task_id: str, timeout: Optional[float] = None) -> Any:
        """
        Get task result

        Args:
            task_id: Task ID
            timeout: Timeout in seconds

        Returns:
            Task result
        """
        if self._app:
            try:
                from celery.result import AsyncResult

                result = AsyncResult(task_id, app=self._app)
                return result.get(timeout=timeout)
            except Exception as e:
                logger.error(f"Failed to get task result: {e}")
                return None
        else:
            logger.warning("Task result retrieval not supported in threading mode")
            return None

    def get_task_status(self, task_id: str) -> str:
        """
        Get task status

        Args:
            task_id: Task ID

        Returns:
            Task status
        """
        if self._app:
            try:
                from celery.result import AsyncResult

                result = AsyncResult(task_id, app=self._app)
                return result.status
            except Exception as e:
                logger.error(f"Failed to get task status: {e}")
                return "UNKNOWN"
        else:
            return "UNKNOWN"


class ServiceDiscovery:
    """
    Service discovery for microservices architecture
    """

    def __init__(
        self,
        registry_backend: str = "redis",
        registry_url: str = "redis://localhost:6379/2",
    ):
        """
        Initialize service discovery

        Args:
            registry_backend: Registry backend type
            registry_url: Registry URL
        """
        self.registry_backend = registry_backend
        self.registry_url = registry_url
        self._registry = None
        self._services: Dict[str, List[Dict[str, Any]]] = {}
        self._lock = threading.RLock()

        self._initialize_registry()

        logger.info("Initialized ServiceDiscovery", backend=registry_backend)

    def _initialize_registry(self):
        """Initialize service registry"""
        if self.registry_backend == "redis":
            try:
                import redis

                # Parse Redis URL
                if self.registry_url.startswith("redis://"):
                    url_parts = self.registry_url.replace("redis://", "").split("/")
                    host_port = url_parts[0].split(":")
                    host = host_port[0]
                    port = int(host_port[1]) if len(host_port) > 1 else 6379
                    db = int(url_parts[1]) if len(url_parts) > 1 else 0

                    self._registry = redis.Redis(host=host, port=port, db=db, decode_responses=True)

                    # Test connection
                    self._registry.ping()
                    logger.info("Connected to Redis service registry")

            except ImportError:
                logger.warning("redis-py not installed, using in-memory registry")
                self._registry = None
            except Exception as e:
                logger.error(f"Failed to connect to Redis registry: {e}")
                self._registry = None
        else:
            logger.warning(f"Unknown registry backend: {self.registry_backend}")

    def register_service(
        self,
        service_name: str,
        host: str,
        port: int,
        metadata: Optional[Dict[str, Any]] = None,
        ttl: int = 60,
    ) -> bool:
        """
        Register a service instance

        Args:
            service_name: Service name
            host: Service host
            port: Service port
            metadata: Additional metadata
            ttl: Time-to-live in seconds

        Returns:
            True if successful
        """
        service_info = {
            "host": host,
            "port": port,
            "metadata": metadata or {},
            "registered_at": datetime.now().isoformat(),
        }

        service_id = f"{service_name}:{host}:{port}"

        if self._registry:
            try:
                # Store in Redis
                key = f"service:{service_id}"
                self._registry.setex(key, ttl, json.dumps(service_info))

                # Add to service set
                self._registry.sadd(f"services:{service_name}", service_id)

                logger.info(f"Registered service: {service_id}")
                return True

            except Exception as e:
                logger.error(f"Failed to register service: {e}")
                return False
        else:
            # In-memory registry
            with self._lock:
                if service_name not in self._services:
                    self._services[service_name] = []

                # Remove existing entry
                self._services[service_name] = [
                    s
                    for s in self._services[service_name]
                    if not (s["host"] == host and s["port"] == port)
                ]

                # Add new entry
                self._services[service_name].append(service_info)

                logger.info(f"Registered service (in-memory): {service_id}")
                return True

    def discover_service(self, service_name: str) -> List[Dict[str, Any]]:
        """
        Discover service instances

        Args:
            service_name: Service name

        Returns:
            List of service instances
        """
        if self._registry:
            try:
                # Get service IDs from Redis
                service_ids = self._registry.smembers(f"services:{service_name}")

                instances = []
                for service_id in service_ids:
                    key = f"service:{service_id}"
                    data = self._registry.get(key)
                    if data:
                        instances.append(json.loads(data))

                return instances

            except Exception as e:
                logger.error(f"Failed to discover services: {e}")
                return []
        else:
            # In-memory registry
            with self._lock:
                return self._services.get(service_name, [])

    def deregister_service(self, service_name: str, host: str, port: int) -> bool:
        """
        Deregister a service instance

        Args:
            service_name: Service name
            host: Service host
            port: Service port

        Returns:
            True if successful
        """
        service_id = f"{service_name}:{host}:{port}"

        if self._registry:
            try:
                # Remove from Redis
                key = f"service:{service_id}"
                self._registry.delete(key)
                self._registry.srem(f"services:{service_name}", service_id)

                logger.info(f"Deregistered service: {service_id}")
                return True

            except Exception as e:
                logger.error(f"Failed to deregister service: {e}")
                return False
        else:
            # In-memory registry
            with self._lock:
                if service_name in self._services:
                    self._services[service_name] = [
                        s
                        for s in self._services[service_name]
                        if not (s["host"] == host and s["port"] == port)
                    ]

                    logger.info(f"Deregistered service (in-memory): {service_id}")
                    return True
                return False

    def get_service_endpoint(self, service_name: str) -> Optional[str]:
        """
        Get a service endpoint (simple round-robin)

        Args:
            service_name: Service name

        Returns:
            Service endpoint URL or None
        """
        instances = self.discover_service(service_name)

        if not instances:
            logger.warning(f"No instances found for service: {service_name}")
            return None

        # Simple round-robin (in production, use more sophisticated load balancing)
        instance = instances[0]
        endpoint = f"http://{instance['host']}:{instance['port']}"

        return endpoint

    def health_check(self, service_name: str) -> Dict[str, Any]:
        """
        Perform health check on service instances

        Args:
            service_name: Service name

        Returns:
            Health check results
        """
        instances = self.discover_service(service_name)

        return {
            "service_name": service_name,
            "instance_count": len(instances),
            "instances": instances,
            "healthy": len(instances) > 0,
        }


class LoadBalancer:
    """
    Simple load balancer for distributing requests
    """

    def __init__(self, strategy: str = "round_robin"):
        """
        Initialize load balancer

        Args:
            strategy: Load balancing strategy ('round_robin', 'least_connections', 'random')
        """
        self.strategy = strategy
        self._current_index = 0
        self._connection_counts: Dict[str, int] = {}
        self._lock = threading.RLock()

        logger.info(f"Initialized LoadBalancer (strategy={strategy})")

    def select_backend(self, backends: List[str]) -> Optional[str]:
        """
        Select a backend server

        Args:
            backends: List of backend URLs

        Returns:
            Selected backend URL
        """
        if not backends:
            return None

        with self._lock:
            if self.strategy == "round_robin":
                backend = backends[self._current_index % len(backends)]
                self._current_index += 1
                return backend

            elif self.strategy == "least_connections":
                # Select backend with fewest connections
                backend = min(backends, key=lambda b: self._connection_counts.get(b, 0))
                return backend

            elif self.strategy == "random":
                import random

                return random.choice(backends)

            else:
                # Default to round-robin
                backend = backends[self._current_index % len(backends)]
                self._current_index += 1
                return backend

    def increment_connections(self, backend: str) -> None:
        """Increment connection count for backend"""
        with self._lock:
            self._connection_counts[backend] = self._connection_counts.get(backend, 0) + 1

    def decrement_connections(self, backend: str) -> None:
        """Decrement connection count for backend"""
        with self._lock:
            if backend in self._connection_counts:
                self._connection_counts[backend] = max(0, self._connection_counts[backend] - 1)

    def get_stats(self) -> Dict[str, Any]:
        """Get load balancer statistics"""
        with self._lock:
            return {
                "strategy": self.strategy,
                "connection_counts": dict(self._connection_counts),
                "total_connections": sum(self._connection_counts.values()),
            }

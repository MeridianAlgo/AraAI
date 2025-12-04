"""
Distributed tracing with OpenTelemetry
Provides tracing capabilities for monitoring request flows
"""

from typing import Optional, Dict, Any, Callable
from functools import wraps
import inspect
import time

try:
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    from opentelemetry.trace import Status, StatusCode
    OPENTELEMETRY_AVAILABLE = True
except ImportError:
    OPENTELEMETRY_AVAILABLE = False
    trace = None

from ara.utils.logging import get_logger

logger = get_logger(__name__)


class TracingManager:
    """
    Manages distributed tracing with OpenTelemetry
    """
    
    def __init__(
        self,
        service_name: str = "ara-ai",
        service_version: str = "1.0.0",
        otlp_endpoint: Optional[str] = None,
        console_export: bool = False
    ):
        """
        Initialize tracing manager
        
        Args:
            service_name: Name of the service
            service_version: Version of the service
            otlp_endpoint: OTLP collector endpoint (e.g., "http://localhost:4317")
            console_export: Whether to export traces to console
        """
        self.enabled = OPENTELEMETRY_AVAILABLE
        self.tracer = None
        
        if not self.enabled:
            logger.warning(
                "OpenTelemetry not available. Install with: pip install opentelemetry-api opentelemetry-sdk opentelemetry-exporter-otlp"
            )
            return
        
        try:
            # Create resource
            resource = Resource.create({
                "service.name": service_name,
                "service.version": service_version
            })
            
            # Create tracer provider
            provider = TracerProvider(resource=resource)
            
            # Add exporters
            if otlp_endpoint:
                otlp_exporter = OTLPSpanExporter(endpoint=otlp_endpoint)
                provider.add_span_processor(BatchSpanProcessor(otlp_exporter))
                logger.info(f"OTLP tracing enabled: {otlp_endpoint}")
            
            if console_export:
                console_exporter = ConsoleSpanExporter()
                provider.add_span_processor(BatchSpanProcessor(console_exporter))
            
            # Set global tracer provider
            trace.set_tracer_provider(provider)
            
            # Get tracer
            self.tracer = trace.get_tracer(__name__)
            
            logger.info("Distributed tracing initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize tracing: {e}")
            self.enabled = False
    
    def start_span(
        self,
        name: str,
        attributes: Optional[Dict[str, Any]] = None
    ):
        """
        Start a new span
        
        Args:
            name: Span name
            attributes: Span attributes
            
        Returns:
            Span context manager or dummy context
        """
        if not self.enabled or not self.tracer:
            return DummySpan()
        
        span = self.tracer.start_as_current_span(name)
        
        if attributes:
            for key, value in attributes.items():
                span.set_attribute(key, value)
        
        return span
    
    def trace_function(
        self,
        span_name: Optional[str] = None,
        attributes: Optional[Dict[str, Any]] = None
    ):
        """
        Decorator to trace function execution
        
        Args:
            span_name: Custom span name (uses function name if None)
            attributes: Additional span attributes
        """
        def decorator(func: Callable) -> Callable:
            name = span_name or f"{func.__module__}.{func.__name__}"
            
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                if not self.enabled or not self.tracer:
                    return await func(*args, **kwargs)
                
                with self.tracer.start_as_current_span(name) as span:
                    # Add attributes
                    if attributes:
                        for key, value in attributes.items():
                            span.set_attribute(key, value)
                    
                    # Add function info
                    span.set_attribute("function.name", func.__name__)
                    span.set_attribute("function.module", func.__module__)
                    
                    try:
                        result = await func(*args, **kwargs)
                        span.set_status(Status(StatusCode.OK))
                        return result
                    except Exception as e:
                        span.set_status(Status(StatusCode.ERROR, str(e)))
                        span.record_exception(e)
                        raise
            
            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                if not self.enabled or not self.tracer:
                    return func(*args, **kwargs)
                
                with self.tracer.start_as_current_span(name) as span:
                    # Add attributes
                    if attributes:
                        for key, value in attributes.items():
                            span.set_attribute(key, value)
                    
                    # Add function info
                    span.set_attribute("function.name", func.__name__)
                    span.set_attribute("function.module", func.__module__)
                    
                    try:
                        result = func(*args, **kwargs)
                        span.set_status(Status(StatusCode.OK))
                        return result
                    except Exception as e:
                        span.set_status(Status(StatusCode.ERROR, str(e)))
                        span.record_exception(e)
                        raise
            
            if inspect.iscoroutinefunction(func):
                return async_wrapper
            return sync_wrapper
        
        return decorator


class DummySpan:
    """Dummy span for when tracing is disabled"""
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass
    
    def set_attribute(self, key: str, value: Any):
        pass
    
    def set_status(self, status):
        pass
    
    def record_exception(self, exception: Exception):
        pass


# Global tracing manager
_tracing_manager: Optional[TracingManager] = None


def get_tracing_manager() -> TracingManager:
    """
    Get global tracing manager
    
    Returns:
        TracingManager instance
    """
    global _tracing_manager
    if _tracing_manager is None:
        _tracing_manager = TracingManager()
    return _tracing_manager


def init_tracing(
    service_name: str = "ara-ai",
    service_version: str = "1.0.0",
    otlp_endpoint: Optional[str] = None,
    console_export: bool = False
):
    """
    Initialize distributed tracing
    
    Args:
        service_name: Name of the service
        service_version: Version of the service
        otlp_endpoint: OTLP collector endpoint
        console_export: Whether to export traces to console
    """
    global _tracing_manager
    _tracing_manager = TracingManager(
        service_name=service_name,
        service_version=service_version,
        otlp_endpoint=otlp_endpoint,
        console_export=console_export
    )


def trace(span_name: Optional[str] = None, **attributes):
    """
    Decorator to trace function execution
    
    Args:
        span_name: Custom span name
        **attributes: Span attributes
    """
    manager = get_tracing_manager()
    return manager.trace_function(span_name, attributes)


def start_span(name: str, **attributes):
    """
    Start a new span
    
    Args:
        name: Span name
        **attributes: Span attributes
        
    Returns:
        Span context manager
    """
    manager = get_tracing_manager()
    return manager.start_span(name, attributes)

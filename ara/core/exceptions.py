"""
Exception hierarchy for ARA AI system
Provides structured error handling
"""


class AraAIException(Exception):
    """Base exception for all ARA AI errors"""
    
    def __init__(self, message: str, details: dict = None):
        self.message = message
        self.details = details or {}
        super().__init__(self.message)
    
    def __str__(self):
        if self.details:
            return f"{self.message} | Details: {self.details}"
        return self.message


class DataProviderError(AraAIException):
    """Data fetching or processing errors"""
    pass


class ModelError(AraAIException):
    """Model training or prediction errors"""
    pass


class ValidationError(AraAIException):
    """Input validation errors"""
    pass


class CacheError(AraAIException):
    """Cache-related errors"""
    pass


class APIError(AraAIException):
    """API request/response errors"""
    pass


class ConfigurationError(AraAIException):
    """Configuration-related errors"""
    pass


class FeatureEngineeringError(AraAIException):
    """Feature calculation errors"""
    pass


class BacktestError(AraAIException):
    """Backtesting-related errors"""
    pass


class PortfolioError(AraAIException):
    """Portfolio optimization errors"""
    pass


class AlertError(AraAIException):
    """Alert and notification errors"""
    pass

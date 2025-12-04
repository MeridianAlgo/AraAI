"""
Alert condition evaluation engine
Evaluates alert conditions against market data and predictions
"""

from typing import Dict, Any, Optional
from datetime import datetime, timedelta

from ara.alerts.models import AlertCondition, ConditionOperator
from ara.core.exceptions import ValidationError
from ara.utils.logging import get_logger

logger = get_logger(__name__)


class ConditionEvaluator:
    """
    Evaluates alert conditions against data
    """
    
    def __init__(self):
        self._previous_values: Dict[str, float] = {}  # For cross detection
    
    def evaluate(
        self,
        condition: AlertCondition,
        data: Dict[str, Any],
        symbol: str
    ) -> tuple[bool, Optional[float]]:
        """
        Evaluate a condition against data
        
        Args:
            condition: Alert condition to evaluate
            data: Data dictionary containing field values
            symbol: Symbol being evaluated (for cross detection)
            
        Returns:
            Tuple of (condition_met, actual_value)
        """
        try:
            # Get the field value from data
            field_value = self._get_field_value(data, condition.field)
            
            if field_value is None:
                logger.warning(
                    f"Field '{condition.field}' not found in data",
                    symbol=symbol
                )
                return False, None
            
            # Evaluate based on operator
            if condition.operator == ConditionOperator.GREATER_THAN:
                result = field_value > condition.value
            
            elif condition.operator == ConditionOperator.LESS_THAN:
                result = field_value < condition.value
            
            elif condition.operator == ConditionOperator.GREATER_EQUAL:
                result = field_value >= condition.value
            
            elif condition.operator == ConditionOperator.LESS_EQUAL:
                result = field_value <= condition.value
            
            elif condition.operator == ConditionOperator.EQUAL:
                result = abs(field_value - condition.value) < 1e-6
            
            elif condition.operator == ConditionOperator.NOT_EQUAL:
                result = abs(field_value - condition.value) >= 1e-6
            
            elif condition.operator == ConditionOperator.CROSSES_ABOVE:
                result = self._check_cross_above(
                    symbol, condition.field, field_value, condition.value
                )
            
            elif condition.operator == ConditionOperator.CROSSES_BELOW:
                result = self._check_cross_below(
                    symbol, condition.field, field_value, condition.value
                )
            
            elif condition.operator == ConditionOperator.PERCENT_CHANGE:
                result = self._check_percent_change(
                    data, condition.field, condition.value, condition.timeframe
                )
            
            else:
                raise ValidationError(
                    f"Unsupported operator: {condition.operator}",
                    {"operator": condition.operator.value}
                )
            
            # Update previous value for cross detection
            if condition.operator in [ConditionOperator.CROSSES_ABOVE, ConditionOperator.CROSSES_BELOW]:
                self._update_previous_value(symbol, condition.field, field_value)
            
            return result, field_value
            
        except Exception as e:
            logger.error(
                f"Error evaluating condition: {e}",
                symbol=symbol,
                condition=str(condition)
            )
            return False, None
    
    def _get_field_value(self, data: Dict[str, Any], field: str) -> Optional[float]:
        """
        Extract field value from data dictionary
        Supports nested fields with dot notation (e.g., "prediction.price")
        """
        try:
            # Handle nested fields
            if '.' in field:
                parts = field.split('.')
                value = data
                for part in parts:
                    if isinstance(value, dict):
                        value = value.get(part)
                    else:
                        return None
                return float(value) if value is not None else None
            
            # Direct field access
            value = data.get(field)
            return float(value) if value is not None else None
            
        except (ValueError, TypeError):
            return None
    
    def _check_cross_above(
        self,
        symbol: str,
        field: str,
        current_value: float,
        threshold: float
    ) -> bool:
        """
        Check if value crossed above threshold
        (previous <= threshold and current > threshold)
        """
        key = f"{symbol}:{field}"
        previous_value = self._previous_values.get(key)
        
        if previous_value is None:
            # First evaluation, no cross yet
            return False
        
        return previous_value <= threshold and current_value > threshold
    
    def _check_cross_below(
        self,
        symbol: str,
        field: str,
        current_value: float,
        threshold: float
    ) -> bool:
        """
        Check if value crossed below threshold
        (previous >= threshold and current < threshold)
        """
        key = f"{symbol}:{field}"
        previous_value = self._previous_values.get(key)
        
        if previous_value is None:
            # First evaluation, no cross yet
            return False
        
        return previous_value >= threshold and current_value < threshold
    
    def _check_percent_change(
        self,
        data: Dict[str, Any],
        field: str,
        threshold_percent: float,
        timeframe: Optional[str]
    ) -> bool:
        """
        Check if percent change exceeds threshold
        Requires historical data in the data dict
        """
        current_value = self._get_field_value(data, field)
        if current_value is None:
            return False
        
        # Try to get historical value
        historical_field = f"{field}_historical"
        if timeframe:
            historical_field = f"{field}_{timeframe}_ago"
        
        historical_value = self._get_field_value(data, historical_field)
        
        if historical_value is None or historical_value == 0:
            return False
        
        percent_change = ((current_value - historical_value) / abs(historical_value)) * 100
        
        return abs(percent_change) >= abs(threshold_percent)
    
    def _update_previous_value(self, symbol: str, field: str, value: float) -> None:
        """Update previous value for cross detection"""
        key = f"{symbol}:{field}"
        self._previous_values[key] = value
    
    def clear_history(self, symbol: Optional[str] = None) -> None:
        """
        Clear evaluation history
        
        Args:
            symbol: If provided, clear only for this symbol. Otherwise clear all.
        """
        if symbol:
            keys_to_remove = [k for k in self._previous_values.keys() if k.startswith(f"{symbol}:")]
            for key in keys_to_remove:
                del self._previous_values[key]
        else:
            self._previous_values.clear()

"""
Data models for alert system
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, Any, Optional, List
from uuid import uuid4


class AlertStatus(Enum):
    """Alert status"""
    ACTIVE = "active"
    PAUSED = "paused"
    TRIGGERED = "triggered"
    DELETED = "deleted"


class AlertPriority(Enum):
    """Alert priority levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class NotificationChannel(Enum):
    """Notification delivery channels"""
    EMAIL = "email"
    SMS = "sms"
    WEBHOOK = "webhook"


class ConditionOperator(Enum):
    """Condition comparison operators"""
    GREATER_THAN = ">"
    LESS_THAN = "<"
    GREATER_EQUAL = ">="
    LESS_EQUAL = "<="
    EQUAL = "=="
    NOT_EQUAL = "!="
    CROSSES_ABOVE = "crosses_above"
    CROSSES_BELOW = "crosses_below"
    PERCENT_CHANGE = "percent_change"


@dataclass
class AlertCondition:
    """
    Alert condition definition
    
    Examples:
        - price > 200
        - predicted_return < -5
        - confidence >= 0.8
        - price crosses_above 150
        - percent_change > 10
    """
    field: str  # e.g., "price", "predicted_return", "confidence"
    operator: ConditionOperator
    value: float
    timeframe: Optional[str] = None  # e.g., "1d", "1h" for percent_change
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "field": self.field,
            "operator": self.operator.value,
            "value": self.value,
            "timeframe": self.timeframe
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AlertCondition':
        """Create from dictionary"""
        return cls(
            field=data["field"],
            operator=ConditionOperator(data["operator"]),
            value=data["value"],
            timeframe=data.get("timeframe")
        )
    
    def __str__(self) -> str:
        """String representation"""
        if self.operator == ConditionOperator.PERCENT_CHANGE:
            return f"{self.field} {self.operator.value} {self.value}% in {self.timeframe}"
        return f"{self.field} {self.operator.value} {self.value}"


@dataclass
class Alert:
    """
    Alert definition
    """
    id: str = field(default_factory=lambda: str(uuid4()))
    name: str = ""
    symbol: str = ""
    condition: AlertCondition = None
    channels: List[NotificationChannel] = field(default_factory=list)
    priority: AlertPriority = AlertPriority.MEDIUM
    status: AlertStatus = AlertStatus.ACTIVE
    message_template: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    last_triggered: Optional[datetime] = None
    trigger_count: int = 0
    cooldown_minutes: int = 60  # Minimum time between notifications
    
    # Notification settings
    email_recipients: List[str] = field(default_factory=list)
    sms_recipients: List[str] = field(default_factory=list)
    webhook_url: Optional[str] = None
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "id": self.id,
            "name": self.name,
            "symbol": self.symbol,
            "condition": self.condition.to_dict() if self.condition else None,
            "channels": [ch.value for ch in self.channels],
            "priority": self.priority.value,
            "status": self.status.value,
            "message_template": self.message_template,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "last_triggered": self.last_triggered.isoformat() if self.last_triggered else None,
            "trigger_count": self.trigger_count,
            "cooldown_minutes": self.cooldown_minutes,
            "email_recipients": self.email_recipients,
            "sms_recipients": self.sms_recipients,
            "webhook_url": self.webhook_url,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Alert':
        """Create from dictionary"""
        return cls(
            id=data.get("id", str(uuid4())),
            name=data["name"],
            symbol=data["symbol"],
            condition=AlertCondition.from_dict(data["condition"]) if data.get("condition") else None,
            channels=[NotificationChannel(ch) for ch in data.get("channels", [])],
            priority=AlertPriority(data.get("priority", "medium")),
            status=AlertStatus(data.get("status", "active")),
            message_template=data.get("message_template"),
            created_at=datetime.fromisoformat(data["created_at"]) if "created_at" in data else datetime.utcnow(),
            updated_at=datetime.fromisoformat(data["updated_at"]) if "updated_at" in data else datetime.utcnow(),
            last_triggered=datetime.fromisoformat(data["last_triggered"]) if data.get("last_triggered") else None,
            trigger_count=data.get("trigger_count", 0),
            cooldown_minutes=data.get("cooldown_minutes", 60),
            email_recipients=data.get("email_recipients", []),
            sms_recipients=data.get("sms_recipients", []),
            webhook_url=data.get("webhook_url"),
            metadata=data.get("metadata", {})
        )
    
    def can_trigger(self) -> bool:
        """Check if alert can be triggered (respects cooldown)"""
        if self.status != AlertStatus.ACTIVE:
            return False
        
        if self.last_triggered is None:
            return True
        
        minutes_since_last = (datetime.utcnow() - self.last_triggered).total_seconds() / 60
        return minutes_since_last >= self.cooldown_minutes
    
    def mark_triggered(self) -> None:
        """Mark alert as triggered"""
        self.last_triggered = datetime.utcnow()
        self.trigger_count += 1
        self.updated_at = datetime.utcnow()


@dataclass
class AlertHistory:
    """
    Alert trigger history record
    """
    id: str = field(default_factory=lambda: str(uuid4()))
    alert_id: str = ""
    symbol: str = ""
    triggered_at: datetime = field(default_factory=datetime.utcnow)
    condition_met: str = ""
    actual_value: float = 0.0
    threshold_value: float = 0.0
    message: str = ""
    channels_notified: List[NotificationChannel] = field(default_factory=list)
    notification_status: Dict[str, bool] = field(default_factory=dict)  # channel -> success
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "id": self.id,
            "alert_id": self.alert_id,
            "symbol": self.symbol,
            "triggered_at": self.triggered_at.isoformat(),
            "condition_met": self.condition_met,
            "actual_value": self.actual_value,
            "threshold_value": self.threshold_value,
            "message": self.message,
            "channels_notified": [ch.value for ch in self.channels_notified],
            "notification_status": {k: v for k, v in self.notification_status.items()},
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AlertHistory':
        """Create from dictionary"""
        return cls(
            id=data.get("id", str(uuid4())),
            alert_id=data["alert_id"],
            symbol=data["symbol"],
            triggered_at=datetime.fromisoformat(data["triggered_at"]) if "triggered_at" in data else datetime.utcnow(),
            condition_met=data.get("condition_met", ""),
            actual_value=data.get("actual_value", 0.0),
            threshold_value=data.get("threshold_value", 0.0),
            message=data.get("message", ""),
            channels_notified=[NotificationChannel(ch) for ch in data.get("channels_notified", [])],
            notification_status=data.get("notification_status", {}),
            metadata=data.get("metadata", {})
        )

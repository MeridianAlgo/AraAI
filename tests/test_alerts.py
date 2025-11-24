"""
Tests for alert and notification system
"""

import pytest
import tempfile
from pathlib import Path
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock

from ara.alerts import (
    AlertManager,
    AlertCondition,
    ConditionOperator,
    NotificationChannel,
    AlertPriority,
    AlertStatus,
    Alert,
    AlertHistory
)
from ara.alerts.evaluator import ConditionEvaluator
from ara.alerts.notifiers import EmailNotifier, SMSNotifier, WebhookNotifier
from ara.core.exceptions import ValidationError


class TestAlertModels:
    """Test alert data models"""
    
    def test_alert_condition_creation(self):
        """Test creating alert conditions"""
        condition = AlertCondition(
            field="price",
            operator=ConditionOperator.GREATER_THAN,
            value=200.0
        )
        
        assert condition.field == "price"
        assert condition.operator == ConditionOperator.GREATER_THAN
        assert condition.value == 200.0
        assert str(condition) == "price > 200.0"
    
    def test_alert_condition_serialization(self):
        """Test condition to/from dict"""
        condition = AlertCondition(
            field="price",
            operator=ConditionOperator.CROSSES_ABOVE,
            value=150.0
        )
        
        data = condition.to_dict()
        restored = AlertCondition.from_dict(data)
        
        assert restored.field == condition.field
        assert restored.operator == condition.operator
        assert restored.value == condition.value
    
    def test_alert_creation(self):
        """Test creating alerts"""
        condition = AlertCondition(
            field="price",
            operator=ConditionOperator.GREATER_THAN,
            value=200.0
        )
        
        alert = Alert(
            name="Test Alert",
            symbol="AAPL",
            condition=condition,
            channels=[NotificationChannel.EMAIL],
            priority=AlertPriority.HIGH,
            email_recipients=["test@example.com"]
        )
        
        assert alert.name == "Test Alert"
        assert alert.symbol == "AAPL"
        assert alert.status == AlertStatus.ACTIVE
        assert alert.priority == AlertPriority.HIGH
        assert len(alert.channels) == 1
    
    def test_alert_can_trigger(self):
        """Test alert cooldown logic"""
        alert = Alert(
            name="Test",
            symbol="AAPL",
            cooldown_minutes=60
        )
        
        # Should trigger on first check
        assert alert.can_trigger() is True
        
        # Mark as triggered
        alert.mark_triggered()
        
        # Should not trigger immediately
        assert alert.can_trigger() is False
        
        # Simulate time passing
        alert.last_triggered = datetime.utcnow() - timedelta(minutes=61)
        assert alert.can_trigger() is True
    
    def test_alert_serialization(self):
        """Test alert to/from dict"""
        condition = AlertCondition(
            field="price",
            operator=ConditionOperator.GREATER_THAN,
            value=200.0
        )
        
        alert = Alert(
            name="Test Alert",
            symbol="AAPL",
            condition=condition,
            channels=[NotificationChannel.EMAIL],
            email_recipients=["test@example.com"]
        )
        
        data = alert.to_dict()
        restored = Alert.from_dict(data)
        
        assert restored.name == alert.name
        assert restored.symbol == alert.symbol
        assert restored.condition.field == alert.condition.field


class TestConditionEvaluator:
    """Test condition evaluation engine"""
    
    def test_greater_than(self):
        """Test greater than operator"""
        evaluator = ConditionEvaluator()
        condition = AlertCondition(
            field="price",
            operator=ConditionOperator.GREATER_THAN,
            value=200.0
        )
        
        # Should trigger
        result, value = evaluator.evaluate(condition, {"price": 205.0}, "AAPL")
        assert result is True
        assert value == 205.0
        
        # Should not trigger
        result, value = evaluator.evaluate(condition, {"price": 195.0}, "AAPL")
        assert result is False
        assert value == 195.0
    
    def test_less_than(self):
        """Test less than operator"""
        evaluator = ConditionEvaluator()
        condition = AlertCondition(
            field="price",
            operator=ConditionOperator.LESS_THAN,
            value=100.0
        )
        
        result, value = evaluator.evaluate(condition, {"price": 95.0}, "AAPL")
        assert result is True
        
        result, value = evaluator.evaluate(condition, {"price": 105.0}, "AAPL")
        assert result is False
    
    def test_equal(self):
        """Test equal operator"""
        evaluator = ConditionEvaluator()
        condition = AlertCondition(
            field="confidence",
            operator=ConditionOperator.EQUAL,
            value=0.85
        )
        
        result, value = evaluator.evaluate(condition, {"confidence": 0.85}, "AAPL")
        assert result is True
        
        result, value = evaluator.evaluate(condition, {"confidence": 0.86}, "AAPL")
        assert result is False
    
    def test_crosses_above(self):
        """Test crosses above operator"""
        evaluator = ConditionEvaluator()
        condition = AlertCondition(
            field="price",
            operator=ConditionOperator.CROSSES_ABOVE,
            value=200.0
        )
        
        # First evaluation - no previous value
        result, value = evaluator.evaluate(condition, {"price": 195.0}, "AAPL")
        assert result is False
        
        # Second evaluation - still below
        result, value = evaluator.evaluate(condition, {"price": 198.0}, "AAPL")
        assert result is False
        
        # Third evaluation - crosses above
        result, value = evaluator.evaluate(condition, {"price": 205.0}, "AAPL")
        assert result is True
        
        # Fourth evaluation - already above
        result, value = evaluator.evaluate(condition, {"price": 210.0}, "AAPL")
        assert result is False
    
    def test_crosses_below(self):
        """Test crosses below operator"""
        evaluator = ConditionEvaluator()
        condition = AlertCondition(
            field="price",
            operator=ConditionOperator.CROSSES_BELOW,
            value=200.0
        )
        
        # Start above threshold
        result, value = evaluator.evaluate(condition, {"price": 205.0}, "AAPL")
        assert result is False
        
        # Cross below
        result, value = evaluator.evaluate(condition, {"price": 195.0}, "AAPL")
        assert result is True
        
        # Already below
        result, value = evaluator.evaluate(condition, {"price": 190.0}, "AAPL")
        assert result is False
    
    def test_nested_field_access(self):
        """Test accessing nested fields"""
        evaluator = ConditionEvaluator()
        condition = AlertCondition(
            field="prediction.confidence",
            operator=ConditionOperator.GREATER_THAN,
            value=0.8
        )
        
        data = {
            "prediction": {
                "confidence": 0.85,
                "price": 205.0
            }
        }
        
        result, value = evaluator.evaluate(condition, data, "AAPL")
        assert result is True
        assert value == 0.85
    
    def test_missing_field(self):
        """Test handling missing fields"""
        evaluator = ConditionEvaluator()
        condition = AlertCondition(
            field="nonexistent",
            operator=ConditionOperator.GREATER_THAN,
            value=100.0
        )
        
        result, value = evaluator.evaluate(condition, {"price": 105.0}, "AAPL")
        assert result is False
        assert value is None


class TestAlertManager:
    """Test alert manager"""
    
    @pytest.fixture
    def temp_storage(self):
        """Create temporary storage directory"""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)
    
    @pytest.fixture
    def manager(self, temp_storage):
        """Create alert manager with temp storage"""
        return AlertManager(storage_path=temp_storage)
    
    def test_create_alert(self, manager):
        """Test creating alerts"""
        condition = AlertCondition(
            field="price",
            operator=ConditionOperator.GREATER_THAN,
            value=200.0
        )
        
        alert = manager.create_alert(
            name="Test Alert",
            symbol="AAPL",
            condition=condition,
            channels=[NotificationChannel.EMAIL],
            email_recipients=["test@example.com"]
        )
        
        assert alert.id is not None
        assert alert.name == "Test Alert"
        assert alert.symbol == "AAPL"
        assert alert.status == AlertStatus.ACTIVE
    
    def test_create_alert_validation(self, manager):
        """Test alert creation validation"""
        condition = AlertCondition(
            field="price",
            operator=ConditionOperator.GREATER_THAN,
            value=200.0
        )
        
        # Missing email recipients
        with pytest.raises(ValidationError):
            manager.create_alert(
                name="Test",
                symbol="AAPL",
                condition=condition,
                channels=[NotificationChannel.EMAIL]
            )
        
        # Missing webhook URL
        with pytest.raises(ValidationError):
            manager.create_alert(
                name="Test",
                symbol="AAPL",
                condition=condition,
                channels=[NotificationChannel.WEBHOOK]
            )
    
    def test_list_alerts(self, manager):
        """Test listing alerts"""
        condition = AlertCondition(
            field="price",
            operator=ConditionOperator.GREATER_THAN,
            value=200.0
        )
        
        # Create multiple alerts
        manager.create_alert(
            name="Alert 1",
            symbol="AAPL",
            condition=condition,
            channels=[NotificationChannel.EMAIL],
            email_recipients=["test@example.com"]
        )
        
        manager.create_alert(
            name="Alert 2",
            symbol="MSFT",
            condition=condition,
            channels=[NotificationChannel.EMAIL],
            email_recipients=["test@example.com"],
            priority=AlertPriority.HIGH
        )
        
        # List all
        all_alerts = manager.list_alerts()
        assert len(all_alerts) == 2
        
        # Filter by symbol
        aapl_alerts = manager.list_alerts(symbol="AAPL")
        assert len(aapl_alerts) == 1
        assert aapl_alerts[0].symbol == "AAPL"
        
        # Filter by priority
        high_alerts = manager.list_alerts(priority=AlertPriority.HIGH)
        assert len(high_alerts) == 1
        assert high_alerts[0].priority == AlertPriority.HIGH
    
    def test_update_alert(self, manager):
        """Test updating alerts"""
        condition = AlertCondition(
            field="price",
            operator=ConditionOperator.GREATER_THAN,
            value=200.0
        )
        
        alert = manager.create_alert(
            name="Test Alert",
            symbol="AAPL",
            condition=condition,
            channels=[NotificationChannel.EMAIL],
            email_recipients=["test@example.com"]
        )
        
        # Update priority
        updated = manager.update_alert(alert.id, priority=AlertPriority.CRITICAL)
        assert updated.priority == AlertPriority.CRITICAL
        
        # Verify persistence
        retrieved = manager.get_alert(alert.id)
        assert retrieved.priority == AlertPriority.CRITICAL
    
    def test_delete_alert(self, manager):
        """Test deleting alerts"""
        condition = AlertCondition(
            field="price",
            operator=ConditionOperator.GREATER_THAN,
            value=200.0
        )
        
        alert = manager.create_alert(
            name="Test Alert",
            symbol="AAPL",
            condition=condition,
            channels=[NotificationChannel.EMAIL],
            email_recipients=["test@example.com"]
        )
        
        # Delete
        result = manager.delete_alert(alert.id)
        assert result is True
        
        # Verify status
        deleted = manager.get_alert(alert.id)
        assert deleted.status == AlertStatus.DELETED
    
    def test_pause_resume_alert(self, manager):
        """Test pausing and resuming alerts"""
        condition = AlertCondition(
            field="price",
            operator=ConditionOperator.GREATER_THAN,
            value=200.0
        )
        
        alert = manager.create_alert(
            name="Test Alert",
            symbol="AAPL",
            condition=condition,
            channels=[NotificationChannel.EMAIL],
            email_recipients=["test@example.com"]
        )
        
        # Pause
        manager.pause_alert(alert.id)
        paused = manager.get_alert(alert.id)
        assert paused.status == AlertStatus.PAUSED
        
        # Resume
        manager.resume_alert(alert.id)
        resumed = manager.get_alert(alert.id)
        assert resumed.status == AlertStatus.ACTIVE
    
    def test_evaluate_alerts(self, manager):
        """Test evaluating alerts"""
        condition = AlertCondition(
            field="price",
            operator=ConditionOperator.GREATER_THAN,
            value=200.0
        )
        
        alert = manager.create_alert(
            name="Test Alert",
            symbol="AAPL",
            condition=condition,
            channels=[NotificationChannel.EMAIL],
            email_recipients=["test@example.com"]
        )
        
        # Data that triggers alert
        data = {"price": 205.0, "volume": 1000000}
        triggered = manager.evaluate_alerts("AAPL", data)
        
        assert len(triggered) == 1
        assert triggered[0].alert_id == alert.id
        assert triggered[0].actual_value == 205.0
        
        # Verify alert was marked as triggered
        updated_alert = manager.get_alert(alert.id)
        assert updated_alert.trigger_count == 1
        assert updated_alert.last_triggered is not None
    
    def test_alert_cooldown(self, manager):
        """Test alert cooldown prevents repeated triggers"""
        condition = AlertCondition(
            field="price",
            operator=ConditionOperator.GREATER_THAN,
            value=200.0
        )
        
        alert = manager.create_alert(
            name="Test Alert",
            symbol="AAPL",
            condition=condition,
            channels=[NotificationChannel.EMAIL],
            email_recipients=["test@example.com"],
            cooldown_minutes=60
        )
        
        data = {"price": 205.0}
        
        # First trigger
        triggered1 = manager.evaluate_alerts("AAPL", data)
        assert len(triggered1) == 1
        
        # Second trigger (should be blocked by cooldown)
        triggered2 = manager.evaluate_alerts("AAPL", data)
        assert len(triggered2) == 0
    
    def test_get_history(self, manager):
        """Test retrieving alert history"""
        condition = AlertCondition(
            field="price",
            operator=ConditionOperator.GREATER_THAN,
            value=200.0
        )
        
        alert = manager.create_alert(
            name="Test Alert",
            symbol="AAPL",
            condition=condition,
            channels=[NotificationChannel.EMAIL],
            email_recipients=["test@example.com"],
            cooldown_minutes=1
        )
        
        # Trigger multiple times
        data = {"price": 205.0}
        manager.evaluate_alerts("AAPL", data)
        
        # Get history
        history = manager.get_history(alert_id=alert.id)
        assert len(history) >= 1
        assert history[0].alert_id == alert.id
    
    def test_get_statistics(self, manager):
        """Test getting statistics"""
        condition = AlertCondition(
            field="price",
            operator=ConditionOperator.GREATER_THAN,
            value=200.0
        )
        
        alert = manager.create_alert(
            name="Test Alert",
            symbol="AAPL",
            condition=condition,
            channels=[NotificationChannel.EMAIL],
            email_recipients=["test@example.com"]
        )
        
        # Global stats
        stats = manager.get_statistics()
        assert stats["total_alerts"] >= 1
        assert stats["active_alerts"] >= 1
        
        # Alert-specific stats
        alert_stats = manager.get_statistics(alert_id=alert.id)
        assert alert_stats["alert_id"] == alert.id
        assert alert_stats["alert_name"] == "Test Alert"
    
    def test_persistence(self, temp_storage):
        """Test alert persistence across manager instances"""
        condition = AlertCondition(
            field="price",
            operator=ConditionOperator.GREATER_THAN,
            value=200.0
        )
        
        # Create manager and alert
        manager1 = AlertManager(storage_path=temp_storage)
        alert = manager1.create_alert(
            name="Test Alert",
            symbol="AAPL",
            condition=condition,
            channels=[NotificationChannel.EMAIL],
            email_recipients=["test@example.com"]
        )
        alert_id = alert.id
        
        # Create new manager instance
        manager2 = AlertManager(storage_path=temp_storage)
        
        # Verify alert was loaded
        loaded_alert = manager2.get_alert(alert_id)
        assert loaded_alert is not None
        assert loaded_alert.name == "Test Alert"


class TestNotifiers:
    """Test notification delivery"""
    
    def test_email_notifier_format(self):
        """Test email message formatting"""
        notifier = EmailNotifier()
        
        condition = AlertCondition(
            field="price",
            operator=ConditionOperator.GREATER_THAN,
            value=200.0
        )
        
        alert = Alert(
            name="Test Alert",
            symbol="AAPL",
            condition=condition,
            channels=[NotificationChannel.EMAIL],
            priority=AlertPriority.HIGH
        )
        
        history = AlertHistory(
            alert_id=alert.id,
            symbol="AAPL",
            condition_met="price > 200.0",
            actual_value=205.0,
            threshold_value=200.0
        )
        
        message = notifier.format_message(alert, history)
        assert "Test Alert" in message
        assert "AAPL" in message
        assert "205.00" in message
    
    def test_webhook_notifier_payload(self):
        """Test webhook payload format"""
        notifier = WebhookNotifier()
        
        condition = AlertCondition(
            field="price",
            operator=ConditionOperator.GREATER_THAN,
            value=200.0
        )
        
        alert = Alert(
            name="Test Alert",
            symbol="AAPL",
            condition=condition,
            channels=[NotificationChannel.WEBHOOK],
            webhook_url="https://example.com/webhook"
        )
        
        history = AlertHistory(
            alert_id=alert.id,
            symbol="AAPL",
            condition_met="price > 200.0",
            actual_value=205.0,
            threshold_value=200.0
        )
        
        # Mock requests.post
        with patch('requests.post') as mock_post:
            mock_post.return_value.status_code = 200
            result = notifier.send(alert, history)
            
            assert result is True
            assert mock_post.called
            
            # Verify payload structure
            call_args = mock_post.call_args
            payload = call_args[1]['json']
            assert 'alert' in payload
            assert 'trigger' in payload
            assert payload['alert']['symbol'] == 'AAPL'


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

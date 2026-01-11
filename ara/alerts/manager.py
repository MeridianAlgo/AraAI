"""
Alert Manager
Central management for alerts, evaluation, and notifications
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from threading import Lock

from ara.alerts.models import (
    Alert,
    AlertCondition,
    AlertStatus,
    AlertPriority,
    NotificationChannel,
    AlertHistory,
)
from ara.alerts.evaluator import ConditionEvaluator
from ara.alerts.notifiers import (
    EmailNotifier,
    SMSNotifier,
    WebhookNotifier,
    BaseNotifier,
)
from ara.core.exceptions import ValidationError
from ara.utils.logging import get_logger

logger = get_logger(__name__)


class AlertManager:
    """
    Manages alerts, evaluates conditions, and sends notifications

    Features:
    - Create, update, delete alerts
    - Evaluate alert conditions against data
    - Send notifications via multiple channels
    - Track alert history
    - Rate limiting to prevent alert fatigue
    """

    def __init__(
        self,
        storage_path: Optional[Path] = None,
        email_notifier: Optional[EmailNotifier] = None,
        sms_notifier: Optional[SMSNotifier] = None,
        webhook_notifier: Optional[WebhookNotifier] = None,
    ):
        """
        Initialize Alert Manager

        Args:
            storage_path: Path to store alerts and history (default: ./alerts)
            email_notifier: Email notifier instance
            sms_notifier: SMS notifier instance
            webhook_notifier: Webhook notifier instance
        """
        self.storage_path = storage_path or Path("alerts")
        self.storage_path.mkdir(parents=True, exist_ok=True)

        self.alerts_file = self.storage_path / "alerts.json"
        self.history_file = self.storage_path / "history.json"

        # Initialize components
        self.evaluator = ConditionEvaluator()
        self.notifiers: Dict[NotificationChannel, BaseNotifier] = {}

        if email_notifier:
            self.notifiers[NotificationChannel.EMAIL] = email_notifier
        else:
            self.notifiers[NotificationChannel.EMAIL] = EmailNotifier()

        if sms_notifier:
            self.notifiers[NotificationChannel.SMS] = sms_notifier
        else:
            self.notifiers[NotificationChannel.SMS] = SMSNotifier()

        if webhook_notifier:
            self.notifiers[NotificationChannel.WEBHOOK] = webhook_notifier
        else:
            self.notifiers[NotificationChannel.WEBHOOK] = WebhookNotifier()

        # In-memory storage
        self.alerts: Dict[str, Alert] = {}
        self.history: List[AlertHistory] = []

        # Thread safety
        self._lock = Lock()

        # Load existing data
        self._load_alerts()
        self._load_history()

        logger.info("AlertManager initialized", storage_path=str(self.storage_path))

    def create_alert(
        self,
        name: str,
        symbol: str,
        condition: AlertCondition,
        channels: List[NotificationChannel],
        priority: AlertPriority = AlertPriority.MEDIUM,
        email_recipients: Optional[List[str]] = None,
        sms_recipients: Optional[List[str]] = None,
        webhook_url: Optional[str] = None,
        message_template: Optional[str] = None,
        cooldown_minutes: int = 60,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Alert:
        """
        Create a new alert

        Args:
            name: Alert name
            symbol: Symbol to monitor
            condition: Alert condition
            channels: Notification channels
            priority: Alert priority
            email_recipients: Email addresses for notifications
            sms_recipients: Phone numbers for SMS notifications
            webhook_url: Webhook URL for notifications
            message_template: Custom message template
            cooldown_minutes: Minimum minutes between notifications
            metadata: Additional metadata

        Returns:
            Created Alert object
        """
        # Validate inputs
        if not name or not symbol:
            raise ValidationError("Alert name and symbol are required")

        if not channels:
            raise ValidationError("At least one notification channel is required")

        if NotificationChannel.EMAIL in channels and not email_recipients:
            raise ValidationError("Email recipients required for email notifications")

        if NotificationChannel.SMS in channels and not sms_recipients:
            raise ValidationError("SMS recipients required for SMS notifications")

        if NotificationChannel.WEBHOOK in channels and not webhook_url:
            raise ValidationError("Webhook URL required for webhook notifications")

        # Create alert
        alert = Alert(
            name=name,
            symbol=symbol,
            condition=condition,
            channels=channels,
            priority=priority,
            email_recipients=email_recipients or [],
            sms_recipients=sms_recipients or [],
            webhook_url=webhook_url,
            message_template=message_template,
            cooldown_minutes=cooldown_minutes,
            metadata=metadata or {},
        )

        with self._lock:
            self.alerts[alert.id] = alert
            self._save_alerts()

        logger.info(
            f"Alert created: {alert.name}",
            alert_id=alert.id,
            symbol=symbol,
            condition=str(condition),
        )

        return alert

    def get_alert(self, alert_id: str) -> Optional[Alert]:
        """Get alert by ID"""
        return self.alerts.get(alert_id)

    def list_alerts(
        self,
        symbol: Optional[str] = None,
        status: Optional[AlertStatus] = None,
        priority: Optional[AlertPriority] = None,
    ) -> List[Alert]:
        """
        List alerts with optional filtering

        Args:
            symbol: Filter by symbol
            status: Filter by status
            priority: Filter by priority

        Returns:
            List of matching alerts
        """
        alerts = list(self.alerts.values())

        if symbol:
            alerts = [a for a in alerts if a.symbol == symbol]

        if status:
            alerts = [a for a in alerts if a.status == status]

        if priority:
            alerts = [a for a in alerts if a.priority == priority]

        return alerts

    def update_alert(self, alert_id: str, **kwargs) -> Alert:
        """
        Update alert properties

        Args:
            alert_id: Alert ID
            **kwargs: Properties to update

        Returns:
            Updated Alert object
        """
        alert = self.get_alert(alert_id)
        if not alert:
            raise ValidationError(f"Alert not found: {alert_id}")

        with self._lock:
            # Update allowed fields
            allowed_fields = [
                "name",
                "condition",
                "channels",
                "priority",
                "status",
                "email_recipients",
                "sms_recipients",
                "webhook_url",
                "message_template",
                "cooldown_minutes",
                "metadata",
            ]

            for key, value in kwargs.items():
                if key in allowed_fields and hasattr(alert, key):
                    setattr(alert, key, value)

            alert.updated_at = datetime.utcnow()
            self._save_alerts()

        logger.info(f"Alert updated: {alert.name}", alert_id=alert_id)

        return alert

    def delete_alert(self, alert_id: str) -> bool:
        """
        Delete an alert

        Args:
            alert_id: Alert ID

        Returns:
            True if deleted successfully
        """
        alert = self.get_alert(alert_id)
        if not alert:
            return False

        with self._lock:
            alert.status = AlertStatus.DELETED
            alert.updated_at = datetime.utcnow()
            self._save_alerts()

        logger.info(f"Alert deleted: {alert.name}", alert_id=alert_id)

        return True

    def pause_alert(self, alert_id: str) -> bool:
        """Pause an alert"""
        return self.update_alert(alert_id, status=AlertStatus.PAUSED) is not None

    def resume_alert(self, alert_id: str) -> bool:
        """Resume a paused alert"""
        return self.update_alert(alert_id, status=AlertStatus.ACTIVE) is not None

    def evaluate_alerts(self, symbol: str, data: Dict[str, Any]) -> List[AlertHistory]:
        """
        Evaluate all active alerts for a symbol

        Args:
            symbol: Symbol to evaluate
            data: Data dictionary containing field values

        Returns:
            List of triggered alert history records
        """
        triggered_histories = []

        # Get active alerts for this symbol
        active_alerts = [
            a
            for a in self.alerts.values()
            if a.symbol == symbol and a.status == AlertStatus.ACTIVE
        ]

        for alert in active_alerts:
            try:
                # Check if alert can be triggered (respects cooldown)
                if not alert.can_trigger():
                    continue

                # Evaluate condition
                condition_met, actual_value = self.evaluator.evaluate(
                    alert.condition, data, symbol
                )

                if condition_met:
                    # Create history record
                    history = AlertHistory(
                        alert_id=alert.id,
                        symbol=symbol,
                        condition_met=str(alert.condition),
                        actual_value=actual_value or 0.0,
                        threshold_value=alert.condition.value,
                        metadata=data.copy(),
                    )

                    # Send notifications
                    notification_status = self._send_notifications(alert, history)
                    history.channels_notified = list(notification_status.keys())
                    history.notification_status = {
                        ch.value: success for ch, success in notification_status.items()
                    }

                    # Format message
                    history.message = self.notifiers[
                        NotificationChannel.EMAIL
                    ].format_message(alert, history)

                    # Update alert
                    with self._lock:
                        alert.mark_triggered()
                        self._save_alerts()

                    # Save history
                    with self._lock:
                        self.history.append(history)
                        self._save_history()

                    triggered_histories.append(history)

                    logger.info(
                        f"Alert triggered: {alert.name}",
                        alert_id=alert.id,
                        symbol=symbol,
                        actual_value=actual_value,
                    )

            except Exception as e:
                logger.error(
                    f"Error evaluating alert {alert.id}: {e}",
                    alert_id=alert.id,
                    symbol=symbol,
                )

        return triggered_histories

    def _send_notifications(
        self, alert: Alert, history: AlertHistory
    ) -> Dict[NotificationChannel, bool]:
        """
        Send notifications via configured channels

        Returns:
            Dictionary mapping channel to success status
        """
        results = {}

        for channel in alert.channels:
            notifier = self.notifiers.get(channel)
            if notifier:
                try:
                    success = notifier.send(alert, history)
                    results[channel] = success
                except Exception as e:
                    logger.error(
                        f"Error sending {channel.value} notification: {e}",
                        alert_id=alert.id,
                    )
                    results[channel] = False
            else:
                logger.warning(f"No notifier configured for channel: {channel.value}")
                results[channel] = False

        return results

    def get_history(
        self,
        alert_id: Optional[str] = None,
        symbol: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 100,
    ) -> List[AlertHistory]:
        """
        Get alert history with optional filtering

        Args:
            alert_id: Filter by alert ID
            symbol: Filter by symbol
            start_date: Filter by start date
            end_date: Filter by end date
            limit: Maximum number of records to return

        Returns:
            List of alert history records
        """
        history = self.history.copy()

        if alert_id:
            history = [h for h in history if h.alert_id == alert_id]

        if symbol:
            history = [h for h in history if h.symbol == symbol]

        if start_date:
            history = [h for h in history if h.triggered_at >= start_date]

        if end_date:
            history = [h for h in history if h.triggered_at <= end_date]

        # Sort by triggered_at descending
        history.sort(key=lambda h: h.triggered_at, reverse=True)

        return history[:limit]

    def get_statistics(self, alert_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get alert statistics

        Args:
            alert_id: Optional alert ID to get stats for specific alert

        Returns:
            Dictionary with statistics
        """
        if alert_id:
            alert = self.get_alert(alert_id)
            if not alert:
                raise ValidationError(f"Alert not found: {alert_id}")

            alert_history = [h for h in self.history if h.alert_id == alert_id]

            return {
                "alert_id": alert_id,
                "alert_name": alert.name,
                "total_triggers": alert.trigger_count,
                "last_triggered": (
                    alert.last_triggered.isoformat() if alert.last_triggered else None
                ),
                "history_count": len(alert_history),
                "status": alert.status.value,
                "created_at": alert.created_at.isoformat(),
            }
        else:
            # Global statistics
            active_alerts = len(
                [a for a in self.alerts.values() if a.status == AlertStatus.ACTIVE]
            )
            paused_alerts = len(
                [a for a in self.alerts.values() if a.status == AlertStatus.PAUSED]
            )

            return {
                "total_alerts": len(self.alerts),
                "active_alerts": active_alerts,
                "paused_alerts": paused_alerts,
                "total_triggers": sum(a.trigger_count for a in self.alerts.values()),
                "history_count": len(self.history),
                "symbols_monitored": len(set(a.symbol for a in self.alerts.values())),
            }

    def _load_alerts(self) -> None:
        """Load alerts from storage"""
        if not self.alerts_file.exists():
            return

        try:
            with open(self.alerts_file, "r") as f:
                data = json.load(f)

            self.alerts = {
                alert_id: Alert.from_dict(alert_data)
                for alert_id, alert_data in data.items()
            }

            logger.info(f"Loaded {len(self.alerts)} alerts from storage")

        except Exception as e:
            logger.error(f"Failed to load alerts: {e}")

    def _save_alerts(self) -> None:
        """Save alerts to storage"""
        try:
            data = {
                alert_id: alert.to_dict() for alert_id, alert in self.alerts.items()
            }

            with open(self.alerts_file, "w") as f:
                json.dump(data, f, indent=2)

        except Exception as e:
            logger.error(f"Failed to save alerts: {e}")

    def _load_history(self) -> None:
        """Load history from storage"""
        if not self.history_file.exists():
            return

        try:
            with open(self.history_file, "r") as f:
                data = json.load(f)

            self.history = [AlertHistory.from_dict(h) for h in data]

            logger.info(f"Loaded {len(self.history)} history records from storage")

        except Exception as e:
            logger.error(f"Failed to load history: {e}")

    def _save_history(self) -> None:
        """Save history to storage (keep last 10000 records)"""
        try:
            # Keep only recent history to prevent file from growing too large
            recent_history = sorted(
                self.history, key=lambda h: h.triggered_at, reverse=True
            )[:10000]

            data = [h.to_dict() for h in recent_history]

            with open(self.history_file, "w") as f:
                json.dump(data, f, indent=2)

        except Exception as e:
            logger.error(f"Failed to save history: {e}")

    def cleanup_old_history(self, days: int = 90) -> int:
        """
        Remove history older than specified days

        Args:
            days: Number of days to keep

        Returns:
            Number of records removed
        """
        cutoff_date = datetime.utcnow() - timedelta(days=days)

        with self._lock:
            original_count = len(self.history)
            self.history = [h for h in self.history if h.triggered_at >= cutoff_date]
            removed_count = original_count - len(self.history)

            if removed_count > 0:
                self._save_history()

        logger.info(f"Cleaned up {removed_count} old history records")

        return removed_count

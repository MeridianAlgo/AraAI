# Alert and Notification System

Comprehensive alert management system with multi-channel notifications for the ARA AI prediction platform.

## Features

- **Alert Management**: Create, update, delete, pause/resume alerts
- **Condition Evaluation**: Flexible condition engine supporting multiple operators
- **Multi-Channel Notifications**: Email (SMTP), SMS (Twilio), and Webhooks
- **Rate Limiting**: Configurable cooldown periods to prevent alert fatigue
- **Alert History**: Track all triggered alerts with detailed metadata
- **Priority Levels**: Low, Medium, High, Critical
- **Custom Templates**: Customize notification messages

## Quick Start

### Basic Usage

```python
from ara.alerts import AlertManager, AlertCondition, ConditionOperator, NotificationChannel, AlertPriority

# Initialize manager
manager = AlertManager()

# Create an alert
condition = AlertCondition(
    field="price",
    operator=ConditionOperator.GREATER_THAN,
    value=200.0
)

alert = manager.create_alert(
    name="AAPL Price Alert",
    symbol="AAPL",
    condition=condition,
    channels=[NotificationChannel.EMAIL],
    priority=AlertPriority.HIGH,
    email_recipients=["trader@example.com"],
    cooldown_minutes=60
)

# Evaluate alerts against data
data = {
    "price": 205.50,
    "volume": 1000000,
    "predicted_return": 5.2
}

triggered = manager.evaluate_alerts("AAPL", data)
print(f"Triggered {len(triggered)} alerts")
```

### Condition Operators

The system supports various condition operators:

- **Comparison**: `>`, `<`, `>=`, `<=`, `==`, `!=`
- **Cross Detection**: `crosses_above`, `crosses_below`
- **Percent Change**: `percent_change` (requires timeframe)

```python
# Price crosses above 200
condition = AlertCondition(
    field="price",
    operator=ConditionOperator.CROSSES_ABOVE,
    value=200.0
)

# Percent change exceeds 10% in 1 day
condition = AlertCondition(
    field="price",
    operator=ConditionOperator.PERCENT_CHANGE,
    value=10.0,
    timeframe="1d"
)

# Confidence below threshold
condition = AlertCondition(
    field="confidence",
    operator=ConditionOperator.LESS_THAN,
    value=0.7
)
```

### Notification Channels

#### Email (SMTP)

Configure via environment variables:

```bash
export SMTP_HOST=smtp.gmail.com
export SMTP_PORT=587
export SMTP_USERNAME=your-email@gmail.com
export SMTP_PASSWORD=your-app-password
export SMTP_FROM_EMAIL=alerts@yourcompany.com
```

```python
alert = manager.create_alert(
    name="Price Alert",
    symbol="BTC-USD",
    condition=condition,
    channels=[NotificationChannel.EMAIL],
    email_recipients=["trader1@example.com", "trader2@example.com"]
)
```

#### SMS (Twilio)

Configure via environment variables:

```bash
export TWILIO_ACCOUNT_SID=your-account-sid
export TWILIO_AUTH_TOKEN=your-auth-token
export TWILIO_PHONE_NUMBER=+1234567890
```

Install Twilio SDK:
```bash
pip install twilio
```

```python
alert = manager.create_alert(
    name="Critical Alert",
    symbol="AAPL",
    condition=condition,
    channels=[NotificationChannel.SMS],
    priority=AlertPriority.CRITICAL,
    sms_recipients=["+1234567890", "+0987654321"]
)
```

#### Webhooks

```python
alert = manager.create_alert(
    name="Webhook Alert",
    symbol="TSLA",
    condition=condition,
    channels=[NotificationChannel.WEBHOOK],
    webhook_url="https://your-app.com/webhook/alerts"
)
```

Webhook payload format:
```json
{
  "alert": {
    "id": "alert-uuid",
    "name": "Price Alert",
    "symbol": "TSLA",
    "priority": "high",
    "condition": "price > 200"
  },
  "trigger": {
    "triggered_at": "2024-01-15T10:30:00Z",
    "condition_met": "price > 200",
    "actual_value": 205.5,
    "threshold_value": 200.0
  },
  "metadata": {
    "volume": 1000000,
    "predicted_return": 5.2
  }
}
```

### Alert Management

```python
# List all alerts
alerts = manager.list_alerts()

# Filter alerts
active_alerts = manager.list_alerts(status=AlertStatus.ACTIVE)
aapl_alerts = manager.list_alerts(symbol="AAPL")
high_priority = manager.list_alerts(priority=AlertPriority.HIGH)

# Get specific alert
alert = manager.get_alert(alert_id)

# Update alert
manager.update_alert(
    alert_id,
    priority=AlertPriority.CRITICAL,
    cooldown_minutes=30
)

# Pause/Resume
manager.pause_alert(alert_id)
manager.resume_alert(alert_id)

# Delete alert
manager.delete_alert(alert_id)
```

### Alert History

```python
# Get all history
history = manager.get_history(limit=100)

# Filter history
alert_history = manager.get_history(alert_id=alert_id)
symbol_history = manager.get_history(symbol="AAPL")
recent_history = manager.get_history(
    start_date=datetime.now() - timedelta(days=7)
)

# Get statistics
stats = manager.get_statistics()
print(f"Total alerts: {stats['total_alerts']}")
print(f"Active alerts: {stats['active_alerts']}")
print(f"Total triggers: {stats['total_triggers']}")

# Alert-specific stats
alert_stats = manager.get_statistics(alert_id=alert_id)
print(f"Trigger count: {alert_stats['total_triggers']}")
print(f"Last triggered: {alert_stats['last_triggered']}")

# Cleanup old history
removed = manager.cleanup_old_history(days=90)
print(f"Removed {removed} old records")
```

### Custom Message Templates

```python
template = """
🔔 Alert: {symbol}
Condition: {condition}
Current Value: ${actual_value:.2f}
Threshold: ${threshold_value:.2f}
Priority: {priority}

Take action immediately!
"""

alert = manager.create_alert(
    name="Custom Alert",
    symbol="AAPL",
    condition=condition,
    channels=[NotificationChannel.EMAIL],
    email_recipients=["trader@example.com"],
    message_template=template
)
```

### Rate Limiting

Prevent alert fatigue with cooldown periods:

```python
alert = manager.create_alert(
    name="Rate Limited Alert",
    symbol="BTC-USD",
    condition=condition,
    channels=[NotificationChannel.EMAIL],
    email_recipients=["trader@example.com"],
    cooldown_minutes=120  # Wait 2 hours between notifications
)
```

## Integration with Prediction System

```python
from ara.api.prediction_engine import PredictionEngine
from ara.alerts import AlertManager, AlertCondition, ConditionOperator, NotificationChannel

# Initialize
prediction_engine = PredictionEngine()
alert_manager = AlertManager()

# Create alert for prediction confidence
condition = AlertCondition(
    field="confidence",
    operator=ConditionOperator.GREATER_EQUAL,
    value=0.85
)

alert = alert_manager.create_alert(
    name="High Confidence Prediction",
    symbol="AAPL",
    condition=condition,
    channels=[NotificationChannel.EMAIL],
    email_recipients=["trader@example.com"]
)

# Make prediction and evaluate alerts
result = await prediction_engine.predict("AAPL", days=5)

# Prepare data for alert evaluation
alert_data = {
    "price": result.current_price,
    "confidence": result.confidence.overall,
    "predicted_return": result.predictions[0].predicted_return,
    "regime": result.regime.current_regime.value
}

# Evaluate alerts
triggered = alert_manager.evaluate_alerts("AAPL", alert_data)
```

## Advanced Features

### Nested Field Access

Access nested fields using dot notation:

```python
condition = AlertCondition(
    field="prediction.confidence",
    operator=ConditionOperator.GREATER_THAN,
    value=0.8
)

data = {
    "prediction": {
        "confidence": 0.85,
        "price": 205.50
    }
}
```

### Multiple Channels

Send notifications via multiple channels:

```python
alert = manager.create_alert(
    name="Multi-Channel Alert",
    symbol="TSLA",
    condition=condition,
    channels=[
        NotificationChannel.EMAIL,
        NotificationChannel.SMS,
        NotificationChannel.WEBHOOK
    ],
    email_recipients=["trader@example.com"],
    sms_recipients=["+1234567890"],
    webhook_url="https://your-app.com/webhook"
)
```

### Priority-Based Routing

Route different priorities to different channels:

```python
# Critical alerts via SMS
critical_alert = manager.create_alert(
    name="Critical Price Movement",
    symbol="BTC-USD",
    condition=critical_condition,
    channels=[NotificationChannel.SMS, NotificationChannel.EMAIL],
    priority=AlertPriority.CRITICAL,
    sms_recipients=["+1234567890"],
    email_recipients=["trader@example.com"],
    cooldown_minutes=15  # More frequent for critical
)

# Low priority via email only
info_alert = manager.create_alert(
    name="Daily Summary",
    symbol="BTC-USD",
    condition=info_condition,
    channels=[NotificationChannel.EMAIL],
    priority=AlertPriority.LOW,
    email_recipients=["trader@example.com"],
    cooldown_minutes=1440  # Once per day
)
```

## Configuration

### Storage

By default, alerts are stored in `./alerts/`:
- `alerts.json`: Alert definitions
- `history.json`: Alert history (last 10,000 records)

Custom storage location:

```python
from pathlib import Path

manager = AlertManager(storage_path=Path("/var/lib/ara/alerts"))
```

### Custom Notifiers

Implement custom notification channels:

```python
from ara.alerts.notifiers import BaseNotifier

class SlackNotifier(BaseNotifier):
    def __init__(self, webhook_url):
        self.webhook_url = webhook_url
    
    def send(self, alert, history):
        # Implement Slack notification
        payload = {
            "text": self.format_message(alert, history)
        }
        response = requests.post(self.webhook_url, json=payload)
        return response.status_code == 200

# Use custom notifier
slack_notifier = SlackNotifier("https://hooks.slack.com/...")
manager = AlertManager(webhook_notifier=slack_notifier)
```

## Error Handling

The system handles errors gracefully:

- Failed notifications are logged but don't stop other channels
- Invalid conditions are caught during evaluation
- Storage failures are logged with fallback to in-memory
- Network errors trigger retries with exponential backoff

```python
try:
    alert = manager.create_alert(...)
except ValidationError as e:
    print(f"Invalid alert configuration: {e}")

# Check notification status
triggered = manager.evaluate_alerts("AAPL", data)
for history in triggered:
    for channel, success in history.notification_status.items():
        if not success:
            print(f"Failed to send {channel} notification")
```

## Best Practices

1. **Set Appropriate Cooldowns**: Prevent alert fatigue with reasonable cooldown periods
2. **Use Priority Levels**: Route critical alerts to immediate channels (SMS)
3. **Monitor History**: Regularly review alert history to tune conditions
4. **Clean Up Old Data**: Periodically cleanup old history records
5. **Test Notifications**: Test each channel before relying on it
6. **Secure Credentials**: Use environment variables for sensitive credentials
7. **Rate Limit Webhooks**: Implement rate limiting on webhook endpoints

## Requirements

- Python 3.11+
- `requests` for webhooks
- `twilio` for SMS (optional)
- SMTP server access for email

## See Also

- [API Documentation](../api/README.md)
- [Prediction Engine](../api/prediction_engine.py)
- [Configuration Guide](../config/README.md)

"""
Alert and Notification System Demo
Demonstrates alert creation, evaluation, and notifications
"""

import asyncio
from datetime import datetime, timedelta
from ara.alerts import (
    AlertManager,
    AlertCondition,
    ConditionOperator,
    NotificationChannel,
    AlertPriority,
    AlertStatus
)


def demo_basic_alerts():
    """Demonstrate basic alert creation and management"""
    print("=" * 60)
    print("Basic Alert Management Demo")
    print("=" * 60)
    
    # Initialize manager
    manager = AlertManager()
    
    # Create a simple price alert
    print("\n1. Creating price alert...")
    condition = AlertCondition(
        field="price",
        operator=ConditionOperator.GREATER_THAN,
        value=200.0
    )
    
    alert = manager.create_alert(
        name="AAPL Price Above 200",
        symbol="AAPL",
        condition=condition,
        channels=[NotificationChannel.EMAIL],
        priority=AlertPriority.MEDIUM,
        email_recipients=["trader@example.com"],
        cooldown_minutes=60
    )
    
    print(f"   Created alert: {alert.name} (ID: {alert.id})")
    print(f"   Condition: {alert.condition}")
    print(f"   Status: {alert.status.value}")
    
    # List alerts
    print("\n2. Listing all alerts...")
    alerts = manager.list_alerts()
    print(f"   Total alerts: {len(alerts)}")
    for a in alerts:
        print(f"   - {a.name} ({a.symbol}): {a.status.value}")
    
    # Update alert
    print("\n3. Updating alert priority...")
    manager.update_alert(alert.id, priority=AlertPriority.HIGH)
    updated_alert = manager.get_alert(alert.id)
    print(f"   New priority: {updated_alert.priority.value}")
    
    # Pause and resume
    print("\n4. Pausing alert...")
    manager.pause_alert(alert.id)
    paused_alert = manager.get_alert(alert.id)
    print(f"   Status: {paused_alert.status.value}")
    
    print("\n5. Resuming alert...")
    manager.resume_alert(alert.id)
    resumed_alert = manager.get_alert(alert.id)
    print(f"   Status: {resumed_alert.status.value}")


def demo_condition_operators():
    """Demonstrate different condition operators"""
    print("\n" + "=" * 60)
    print("Condition Operators Demo")
    print("=" * 60)
    
    manager = AlertManager()
    
    # Greater than
    print("\n1. Greater Than Operator")
    condition1 = AlertCondition(
        field="price",
        operator=ConditionOperator.GREATER_THAN,
        value=150.0
    )
    print(f"   Condition: {condition1}")
    
    # Crosses above
    print("\n2. Crosses Above Operator")
    condition2 = AlertCondition(
        field="price",
        operator=ConditionOperator.CROSSES_ABOVE,
        value=200.0
    )
    print(f"   Condition: {condition2}")
    
    # Percent change
    print("\n3. Percent Change Operator")
    condition3 = AlertCondition(
        field="price",
        operator=ConditionOperator.PERCENT_CHANGE,
        value=10.0,
        timeframe="1d"
    )
    print(f"   Condition: {condition3}")
    
    # Less than or equal
    print("\n4. Less Than or Equal Operator")
    condition4 = AlertCondition(
        field="confidence",
        operator=ConditionOperator.LESS_EQUAL,
        value=0.7
    )
    print(f"   Condition: {condition4}")


def demo_alert_evaluation():
    """Demonstrate alert evaluation"""
    print("\n" + "=" * 60)
    print("Alert Evaluation Demo")
    print("=" * 60)
    
    manager = AlertManager()
    
    # Create multiple alerts
    print("\n1. Creating alerts...")
    
    # Price alert
    price_condition = AlertCondition(
        field="price",
        operator=ConditionOperator.GREATER_THAN,
        value=200.0
    )
    
    price_alert = manager.create_alert(
        name="High Price Alert",
        symbol="AAPL",
        condition=price_condition,
        channels=[NotificationChannel.EMAIL],
        email_recipients=["trader@example.com"]
    )
    print(f"   Created: {price_alert.name}")
    
    # Confidence alert
    confidence_condition = AlertCondition(
        field="confidence",
        operator=ConditionOperator.GREATER_EQUAL,
        value=0.85
    )
    
    confidence_alert = manager.create_alert(
        name="High Confidence Alert",
        symbol="AAPL",
        condition=confidence_condition,
        channels=[NotificationChannel.EMAIL],
        email_recipients=["trader@example.com"]
    )
    print(f"   Created: {confidence_alert.name}")
    
    # Evaluate with data that triggers price alert
    print("\n2. Evaluating alerts (price = 205, confidence = 0.75)...")
    data1 = {
        "price": 205.0,
        "confidence": 0.75,
        "volume": 1000000
    }
    
    triggered1 = manager.evaluate_alerts("AAPL", data1)
    print(f"   Triggered {len(triggered1)} alert(s)")
    for history in triggered1:
        print(f"   - {history.condition_met}")
        print(f"     Actual: {history.actual_value:.2f}, Threshold: {history.threshold_value:.2f}")
    
    # Evaluate with data that triggers confidence alert
    print("\n3. Evaluating alerts (price = 195, confidence = 0.90)...")
    data2 = {
        "price": 195.0,
        "confidence": 0.90,
        "volume": 1500000
    }
    
    triggered2 = manager.evaluate_alerts("AAPL", data2)
    print(f"   Triggered {len(triggered2)} alert(s)")
    for history in triggered2:
        print(f"   - {history.condition_met}")
        print(f"     Actual: {history.actual_value:.2f}, Threshold: {history.threshold_value:.2f}")
    
    # Evaluate with data that triggers both
    print("\n4. Evaluating alerts (price = 210, confidence = 0.88)...")
    data3 = {
        "price": 210.0,
        "confidence": 0.88,
        "volume": 2000000
    }
    
    triggered3 = manager.evaluate_alerts("AAPL", data3)
    print(f"   Triggered {len(triggered3)} alert(s)")
    for history in triggered3:
        print(f"   - {history.condition_met}")
        print(f"     Actual: {history.actual_value:.2f}, Threshold: {history.threshold_value:.2f}")


def demo_multi_channel():
    """Demonstrate multi-channel notifications"""
    print("\n" + "=" * 60)
    print("Multi-Channel Notifications Demo")
    print("=" * 60)
    
    manager = AlertManager()
    
    print("\n1. Creating multi-channel alert...")
    print("   Note: Configure SMTP and Twilio credentials for actual notifications")
    
    condition = AlertCondition(
        field="price",
        operator=ConditionOperator.GREATER_THAN,
        value=1000.0
    )
    
    alert = manager.create_alert(
        name="Critical BTC Alert",
        symbol="BTC-USD",
        condition=condition,
        channels=[
            NotificationChannel.EMAIL,
            NotificationChannel.WEBHOOK
        ],
        priority=AlertPriority.CRITICAL,
        email_recipients=["trader@example.com"],
        webhook_url="https://example.com/webhook/alerts",
        cooldown_minutes=30
    )
    
    print(f"   Created: {alert.name}")
    print(f"   Channels: {[ch.value for ch in alert.channels]}")
    print(f"   Priority: {alert.priority.value}")
    
    # Simulate evaluation
    print("\n2. Simulating alert trigger...")
    data = {
        "price": 45000.0,
        "volume": 1000000000
    }
    
    triggered = manager.evaluate_alerts("BTC-USD", data)
    if triggered:
        print(f"   Alert triggered!")
        for history in triggered:
            print(f"   Channels notified: {[ch.value for ch in history.channels_notified]}")
            print(f"   Notification status: {history.notification_status}")


def demo_alert_history():
    """Demonstrate alert history and statistics"""
    print("\n" + "=" * 60)
    print("Alert History and Statistics Demo")
    print("=" * 60)
    
    manager = AlertManager()
    
    # Create and trigger some alerts
    print("\n1. Creating and triggering alerts...")
    
    condition = AlertCondition(
        field="price",
        operator=ConditionOperator.GREATER_THAN,
        value=100.0
    )
    
    alert = manager.create_alert(
        name="Test Alert",
        symbol="AAPL",
        condition=condition,
        channels=[NotificationChannel.EMAIL],
        email_recipients=["trader@example.com"],
        cooldown_minutes=1  # Short cooldown for demo
    )
    
    # Trigger multiple times
    for i in range(3):
        data = {"price": 105.0 + i * 5}
        triggered = manager.evaluate_alerts("AAPL", data)
        if triggered:
            print(f"   Trigger {i+1}: price = {data['price']}")
    
    # Get history
    print("\n2. Retrieving alert history...")
    history = manager.get_history(alert_id=alert.id)
    print(f"   Total history records: {len(history)}")
    for h in history:
        print(f"   - {h.triggered_at.strftime('%Y-%m-%d %H:%M:%S')}: {h.condition_met}")
    
    # Get statistics
    print("\n3. Alert statistics...")
    stats = manager.get_statistics(alert_id=alert.id)
    print(f"   Alert: {stats['alert_name']}")
    print(f"   Total triggers: {stats['total_triggers']}")
    print(f"   Last triggered: {stats['last_triggered']}")
    print(f"   Status: {stats['status']}")
    
    # Global statistics
    print("\n4. Global statistics...")
    global_stats = manager.get_statistics()
    print(f"   Total alerts: {global_stats['total_alerts']}")
    print(f"   Active alerts: {global_stats['active_alerts']}")
    print(f"   Total triggers: {global_stats['total_triggers']}")
    print(f"   Symbols monitored: {global_stats['symbols_monitored']}")


def demo_custom_templates():
    """Demonstrate custom message templates"""
    print("\n" + "=" * 60)
    print("Custom Message Templates Demo")
    print("=" * 60)
    
    manager = AlertManager()
    
    print("\n1. Creating alert with custom template...")
    
    template = """
🚨 URGENT ALERT 🚨

Symbol: {symbol}
Condition Met: {condition}

Current Value: ${actual_value:.2f}
Threshold: ${threshold_value:.2f}

Priority: {priority}
Time: {triggered_at}

Action Required: Review your positions immediately!
"""
    
    condition = AlertCondition(
        field="price",
        operator=ConditionOperator.LESS_THAN,
        value=150.0
    )
    
    alert = manager.create_alert(
        name="Stop Loss Alert",
        symbol="AAPL",
        condition=condition,
        channels=[NotificationChannel.EMAIL],
        email_recipients=["trader@example.com"],
        message_template=template,
        priority=AlertPriority.CRITICAL
    )
    
    print(f"   Created: {alert.name}")
    print(f"   Template configured: Yes")
    
    # Trigger alert to see formatted message
    print("\n2. Triggering alert to see formatted message...")
    data = {"price": 145.0}
    triggered = manager.evaluate_alerts("AAPL", data)
    
    if triggered:
        print("\n   Formatted message:")
        print("   " + "-" * 50)
        print("   " + triggered[0].message.replace("\n", "\n   "))
        print("   " + "-" * 50)


def main():
    """Run all demos"""
    print("\n" + "=" * 60)
    print("ARA AI Alert System - Comprehensive Demo")
    print("=" * 60)
    
    try:
        demo_basic_alerts()
        demo_condition_operators()
        demo_alert_evaluation()
        demo_multi_channel()
        demo_alert_history()
        demo_custom_templates()
        
        print("\n" + "=" * 60)
        print("Demo completed successfully!")
        print("=" * 60)
        print("\nNext steps:")
        print("1. Configure SMTP credentials for email notifications")
        print("2. Configure Twilio credentials for SMS notifications")
        print("3. Set up webhook endpoints for custom integrations")
        print("4. Integrate with prediction engine for automated alerts")
        print("\nSee ara/alerts/README.md for detailed documentation")
        
    except Exception as e:
        print(f"\nError during demo: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

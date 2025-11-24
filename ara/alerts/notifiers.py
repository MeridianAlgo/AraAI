"""
Notification delivery implementations
Supports email (SMTP), SMS (Twilio), and webhooks
"""

import smtplib
import requests
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Dict, Any, Optional, List
from abc import ABC, abstractmethod

from ara.alerts.models import Alert, AlertHistory, NotificationChannel
from ara.core.exceptions import APIError
from ara.utils.logging import get_logger
from ara.config import get_config

logger = get_logger(__name__)


class BaseNotifier(ABC):
    """Base class for notification delivery"""
    
    @abstractmethod
    def send(self, alert: Alert, history: AlertHistory) -> bool:
        """
        Send notification
        
        Args:
            alert: Alert that was triggered
            history: Alert history record
            
        Returns:
            True if notification sent successfully
        """
        pass
    
    def format_message(self, alert: Alert, history: AlertHistory) -> str:
        """
        Format notification message
        
        Args:
            alert: Alert that was triggered
            history: Alert history record
            
        Returns:
            Formatted message string
        """
        if alert.message_template:
            # Use custom template
            return alert.message_template.format(
                symbol=alert.symbol,
                condition=history.condition_met,
                actual_value=history.actual_value,
                threshold_value=history.threshold_value,
                priority=alert.priority.value,
                triggered_at=history.triggered_at.strftime('%Y-%m-%d %H:%M:%S UTC'),
                **history.metadata
            )
        
        # Default message format
        return (
            f"🔔 Alert: {alert.name}\n"
            f"Symbol: {alert.symbol}\n"
            f"Condition: {history.condition_met}\n"
            f"Actual Value: {history.actual_value:.2f}\n"
            f"Threshold: {history.threshold_value:.2f}\n"
            f"Priority: {alert.priority.value.upper()}\n"
            f"Time: {history.triggered_at.strftime('%Y-%m-%d %H:%M:%S UTC')}"
        )


class EmailNotifier(BaseNotifier):
    """
    Email notification via SMTP
    
    Configuration (environment variables or config):
        - SMTP_HOST: SMTP server hostname
        - SMTP_PORT: SMTP server port (default: 587)
        - SMTP_USERNAME: SMTP username
        - SMTP_PASSWORD: SMTP password
        - SMTP_FROM_EMAIL: Sender email address
        - SMTP_USE_TLS: Use TLS (default: True)
    """
    
    def __init__(
        self,
        smtp_host: Optional[str] = None,
        smtp_port: Optional[int] = None,
        smtp_username: Optional[str] = None,
        smtp_password: Optional[str] = None,
        from_email: Optional[str] = None,
        use_tls: bool = True
    ):
        import os
        
        self.smtp_host = smtp_host or os.getenv('SMTP_HOST', 'smtp.gmail.com')
        self.smtp_port = smtp_port or int(os.getenv('SMTP_PORT', '587'))
        self.smtp_username = smtp_username or os.getenv('SMTP_USERNAME', '')
        self.smtp_password = smtp_password or os.getenv('SMTP_PASSWORD', '')
        self.from_email = from_email or os.getenv('SMTP_FROM_EMAIL', self.smtp_username)
        self.use_tls = use_tls
        
        if not self.smtp_username or not self.smtp_password:
            logger.warning("SMTP credentials not configured. Email notifications will fail.")
    
    def send(self, alert: Alert, history: AlertHistory) -> bool:
        """Send email notification"""
        if not alert.email_recipients:
            logger.warning(f"No email recipients configured for alert {alert.id}")
            return False
        
        if not self.smtp_username or not self.smtp_password:
            logger.error("SMTP credentials not configured")
            return False
        
        try:
            # Create message
            msg = MIMEMultipart('alternative')
            msg['Subject'] = f"[{alert.priority.value.upper()}] Alert: {alert.name}"
            msg['From'] = self.from_email
            msg['To'] = ', '.join(alert.email_recipients)
            
            # Plain text version
            text_content = self.format_message(alert, history)
            text_part = MIMEText(text_content, 'plain')
            msg.attach(text_part)
            
            # HTML version
            html_content = self._format_html_message(alert, history)
            html_part = MIMEText(html_content, 'html')
            msg.attach(html_part)
            
            # Send email
            with smtplib.SMTP(self.smtp_host, self.smtp_port) as server:
                if self.use_tls:
                    server.starttls()
                server.login(self.smtp_username, self.smtp_password)
                server.send_message(msg)
            
            logger.info(
                f"Email notification sent for alert {alert.id}",
                recipients=alert.email_recipients,
                symbol=alert.symbol
            )
            return True
            
        except Exception as e:
            logger.error(
                f"Failed to send email notification: {e}",
                alert_id=alert.id,
                error=str(e)
            )
            return False
    
    def _format_html_message(self, alert: Alert, history: AlertHistory) -> str:
        """Format HTML email message"""
        priority_colors = {
            'low': '#28a745',
            'medium': '#ffc107',
            'high': '#fd7e14',
            'critical': '#dc3545'
        }
        color = priority_colors.get(alert.priority.value, '#6c757d')
        
        return f"""
        <html>
        <body style="font-family: Arial, sans-serif; padding: 20px;">
            <div style="border-left: 4px solid {color}; padding-left: 15px;">
                <h2 style="color: {color}; margin-top: 0;">🔔 Alert: {alert.name}</h2>
                <table style="border-collapse: collapse; width: 100%; max-width: 600px;">
                    <tr>
                        <td style="padding: 8px; font-weight: bold;">Symbol:</td>
                        <td style="padding: 8px;">{alert.symbol}</td>
                    </tr>
                    <tr style="background-color: #f8f9fa;">
                        <td style="padding: 8px; font-weight: bold;">Condition:</td>
                        <td style="padding: 8px;">{history.condition_met}</td>
                    </tr>
                    <tr>
                        <td style="padding: 8px; font-weight: bold;">Actual Value:</td>
                        <td style="padding: 8px;">{history.actual_value:.2f}</td>
                    </tr>
                    <tr style="background-color: #f8f9fa;">
                        <td style="padding: 8px; font-weight: bold;">Threshold:</td>
                        <td style="padding: 8px;">{history.threshold_value:.2f}</td>
                    </tr>
                    <tr>
                        <td style="padding: 8px; font-weight: bold;">Priority:</td>
                        <td style="padding: 8px; color: {color}; font-weight: bold;">
                            {alert.priority.value.upper()}
                        </td>
                    </tr>
                    <tr style="background-color: #f8f9fa;">
                        <td style="padding: 8px; font-weight: bold;">Time:</td>
                        <td style="padding: 8px;">{history.triggered_at.strftime('%Y-%m-%d %H:%M:%S UTC')}</td>
                    </tr>
                </table>
            </div>
            <p style="color: #6c757d; font-size: 12px; margin-top: 20px;">
                This is an automated alert from ARA AI Prediction System.
            </p>
        </body>
        </html>
        """


class SMSNotifier(BaseNotifier):
    """
    SMS notification via Twilio
    
    Configuration (environment variables):
        - TWILIO_ACCOUNT_SID: Twilio account SID
        - TWILIO_AUTH_TOKEN: Twilio auth token
        - TWILIO_PHONE_NUMBER: Twilio phone number (sender)
    """
    
    def __init__(
        self,
        account_sid: Optional[str] = None,
        auth_token: Optional[str] = None,
        from_phone: Optional[str] = None
    ):
        import os
        
        self.account_sid = account_sid or os.getenv('TWILIO_ACCOUNT_SID', '')
        self.auth_token = auth_token or os.getenv('TWILIO_AUTH_TOKEN', '')
        self.from_phone = from_phone or os.getenv('TWILIO_PHONE_NUMBER', '')
        
        if not self.account_sid or not self.auth_token or not self.from_phone:
            logger.warning("Twilio credentials not configured. SMS notifications will fail.")
        
        self.client = None
        if self.account_sid and self.auth_token:
            try:
                from twilio.rest import Client
                self.client = Client(self.account_sid, self.auth_token)
            except ImportError:
                logger.warning("Twilio library not installed. Install with: pip install twilio")
    
    def send(self, alert: Alert, history: AlertHistory) -> bool:
        """Send SMS notification"""
        if not alert.sms_recipients:
            logger.warning(f"No SMS recipients configured for alert {alert.id}")
            return False
        
        if not self.client:
            logger.error("Twilio client not initialized")
            return False
        
        try:
            # Format message (SMS has character limits)
            message = self._format_sms_message(alert, history)
            
            # Send to all recipients
            success_count = 0
            for phone_number in alert.sms_recipients:
                try:
                    self.client.messages.create(
                        body=message,
                        from_=self.from_phone,
                        to=phone_number
                    )
                    success_count += 1
                except Exception as e:
                    logger.error(
                        f"Failed to send SMS to {phone_number}: {e}",
                        alert_id=alert.id
                    )
            
            if success_count > 0:
                logger.info(
                    f"SMS notifications sent for alert {alert.id}",
                    recipients_count=success_count,
                    symbol=alert.symbol
                )
                return True
            
            return False
            
        except Exception as e:
            logger.error(
                f"Failed to send SMS notification: {e}",
                alert_id=alert.id,
                error=str(e)
            )
            return False
    
    def _format_sms_message(self, alert: Alert, history: AlertHistory) -> str:
        """Format SMS message (keep it short)"""
        priority_emoji = {
            'low': 'ℹ️',
            'medium': '⚠️',
            'high': '🔥',
            'critical': '🚨'
        }
        emoji = priority_emoji.get(alert.priority.value, '🔔')
        
        return (
            f"{emoji} {alert.name}\n"
            f"{alert.symbol}: {history.condition_met}\n"
            f"Value: {history.actual_value:.2f} (threshold: {history.threshold_value:.2f})"
        )


class WebhookNotifier(BaseNotifier):
    """
    Webhook notification via HTTP POST
    Sends JSON payload to configured webhook URL
    """
    
    def __init__(self, timeout: int = 10, max_retries: int = 3):
        self.timeout = timeout
        self.max_retries = max_retries
    
    def send(self, alert: Alert, history: AlertHistory) -> bool:
        """Send webhook notification"""
        if not alert.webhook_url:
            logger.warning(f"No webhook URL configured for alert {alert.id}")
            return False
        
        try:
            # Prepare payload
            payload = {
                "alert": {
                    "id": alert.id,
                    "name": alert.name,
                    "symbol": alert.symbol,
                    "priority": alert.priority.value,
                    "condition": str(alert.condition)
                },
                "trigger": {
                    "triggered_at": history.triggered_at.isoformat(),
                    "condition_met": history.condition_met,
                    "actual_value": history.actual_value,
                    "threshold_value": history.threshold_value
                },
                "metadata": history.metadata
            }
            
            # Send webhook with retries
            for attempt in range(self.max_retries):
                try:
                    response = requests.post(
                        alert.webhook_url,
                        json=payload,
                        timeout=self.timeout,
                        headers={'Content-Type': 'application/json'}
                    )
                    
                    if response.status_code in [200, 201, 202, 204]:
                        logger.info(
                            f"Webhook notification sent for alert {alert.id}",
                            url=alert.webhook_url,
                            status_code=response.status_code,
                            symbol=alert.symbol
                        )
                        return True
                    else:
                        logger.warning(
                            f"Webhook returned non-success status: {response.status_code}",
                            alert_id=alert.id,
                            attempt=attempt + 1
                        )
                        
                except requests.exceptions.Timeout:
                    logger.warning(
                        f"Webhook timeout (attempt {attempt + 1}/{self.max_retries})",
                        alert_id=alert.id
                    )
                except requests.exceptions.RequestException as e:
                    logger.warning(
                        f"Webhook request failed (attempt {attempt + 1}/{self.max_retries}): {e}",
                        alert_id=alert.id
                    )
                
                # Wait before retry (exponential backoff)
                if attempt < self.max_retries - 1:
                    import time
                    time.sleep(2 ** attempt)
            
            logger.error(
                f"Failed to send webhook notification after {self.max_retries} attempts",
                alert_id=alert.id
            )
            return False
            
        except Exception as e:
            logger.error(
                f"Failed to send webhook notification: {e}",
                alert_id=alert.id,
                error=str(e)
            )
            return False

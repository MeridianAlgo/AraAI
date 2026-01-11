"""
Security Audit Logger Module

Provides comprehensive security event logging for audit trails and
compliance requirements.
"""

import json
import logging
from datetime import datetime
from typing import Any, Dict, Optional
from pathlib import Path
from enum import Enum


class SecurityEventType(Enum):
    """Types of security events"""

    # Authentication events
    LOGIN_SUCCESS = "login_success"
    LOGIN_FAILURE = "login_failure"
    LOGOUT = "logout"
    TOKEN_GENERATED = "token_generated"
    TOKEN_EXPIRED = "token_expired"
    TOKEN_REVOKED = "token_revoked"

    # Authorization events
    ACCESS_GRANTED = "access_granted"
    ACCESS_DENIED = "access_denied"
    PERMISSION_CHANGED = "permission_changed"

    # API key events
    API_KEY_CREATED = "api_key_created"
    API_KEY_DELETED = "api_key_deleted"
    API_KEY_ROTATED = "api_key_rotated"
    API_KEY_COMPROMISED = "api_key_compromised"

    # Data access events
    DATA_READ = "data_read"
    DATA_WRITE = "data_write"
    DATA_DELETE = "data_delete"
    DATA_EXPORT = "data_export"

    # Security events
    RATE_LIMIT_EXCEEDED = "rate_limit_exceeded"
    SUSPICIOUS_ACTIVITY = "suspicious_activity"
    INJECTION_ATTEMPT = "injection_attempt"
    SQL_INJECTION_ATTEMPT = "sql_injection_attempt"
    XSS_ATTEMPT = "xss_attempt"
    BRUTE_FORCE_ATTEMPT = "brute_force_attempt"

    # Model events
    MODEL_ACCESSED = "model_accessed"
    MODEL_MODIFIED = "model_modified"
    MODEL_DEPLOYED = "model_deployed"

    # Configuration events
    CONFIG_CHANGED = "config_changed"
    ENCRYPTION_KEY_ROTATED = "encryption_key_rotated"


class SecurityEventSeverity(Enum):
    """Severity levels for security events"""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class SecurityAuditLogger:
    """Logs security events for audit trails"""

    def __init__(self, log_file: Optional[Path] = None, enable_console: bool = False):
        """
        Initialize security audit logger

        Args:
            log_file: Path to audit log file (None = default location)
            enable_console: Whether to also log to console
        """
        if log_file is None:
            log_dir = Path.home() / ".ara" / "logs"
            log_dir.mkdir(parents=True, exist_ok=True)
            log_file = log_dir / "security_audit.log"

        self.log_file = log_file

        # Set up logger
        self.logger = logging.getLogger("ara.security.audit")
        self.logger.setLevel(logging.INFO)

        # File handler (JSON format)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(logging.Formatter("%(message)s"))
        self.logger.addHandler(file_handler)

        # Console handler (optional)
        if enable_console:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.WARNING)
            console_handler.setFormatter(
                logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
            )
            self.logger.addHandler(console_handler)

    def log_event(
        self,
        event_type: SecurityEventType,
        severity: SecurityEventSeverity = SecurityEventSeverity.INFO,
        user_id: Optional[str] = None,
        ip_address: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        success: bool = True,
    ) -> None:
        """
        Log a security event

        Args:
            event_type: Type of security event
            severity: Severity level
            user_id: User identifier (if applicable)
            ip_address: IP address of request
            details: Additional event details
            success: Whether the action was successful
        """
        event = {
            "timestamp": datetime.utcnow().isoformat(),
            "event_type": event_type.value,
            "severity": severity.value,
            "success": success,
        }

        if user_id:
            event["user_id"] = user_id

        if ip_address:
            event["ip_address"] = ip_address

        if details:
            event["details"] = details

        # Log as JSON
        log_message = json.dumps(event)

        # Log at appropriate level
        if severity == SecurityEventSeverity.CRITICAL:
            self.logger.critical(log_message)
        elif severity == SecurityEventSeverity.ERROR:
            self.logger.error(log_message)
        elif severity == SecurityEventSeverity.WARNING:
            self.logger.warning(log_message)
        else:
            self.logger.info(log_message)

    def log_authentication(
        self,
        success: bool,
        user_id: Optional[str] = None,
        ip_address: Optional[str] = None,
        method: str = "password",
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Log authentication attempt

        Args:
            success: Whether authentication succeeded
            user_id: User identifier
            ip_address: IP address
            method: Authentication method (password, api_key, token)
            details: Additional details
        """
        event_type = (
            SecurityEventType.LOGIN_SUCCESS
            if success
            else SecurityEventType.LOGIN_FAILURE
        )
        severity = (
            SecurityEventSeverity.INFO if success else SecurityEventSeverity.WARNING
        )

        event_details = {"method": method}
        if details:
            event_details.update(details)

        self.log_event(
            event_type=event_type,
            severity=severity,
            user_id=user_id,
            ip_address=ip_address,
            details=event_details,
            success=success,
        )

    def log_authorization(
        self,
        success: bool,
        user_id: str,
        resource: str,
        action: str,
        ip_address: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Log authorization check

        Args:
            success: Whether authorization succeeded
            user_id: User identifier
            resource: Resource being accessed
            action: Action being performed
            ip_address: IP address
            details: Additional details
        """
        event_type = (
            SecurityEventType.ACCESS_GRANTED
            if success
            else SecurityEventType.ACCESS_DENIED
        )
        severity = (
            SecurityEventSeverity.INFO if success else SecurityEventSeverity.WARNING
        )

        event_details = {"resource": resource, "action": action}
        if details:
            event_details.update(details)

        self.log_event(
            event_type=event_type,
            severity=severity,
            user_id=user_id,
            ip_address=ip_address,
            details=event_details,
            success=success,
        )

    def log_api_key_event(
        self,
        event_type: SecurityEventType,
        user_id: str,
        key_id: str,
        ip_address: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Log API key related event

        Args:
            event_type: Type of API key event
            user_id: User identifier
            key_id: API key identifier (masked)
            ip_address: IP address
            details: Additional details
        """
        event_details = {"key_id": key_id}
        if details:
            event_details.update(details)

        severity = (
            SecurityEventSeverity.CRITICAL
            if event_type == SecurityEventType.API_KEY_COMPROMISED
            else SecurityEventSeverity.INFO
        )

        self.log_event(
            event_type=event_type,
            severity=severity,
            user_id=user_id,
            ip_address=ip_address,
            details=event_details,
        )

    def log_suspicious_activity(
        self,
        activity_type: str,
        user_id: Optional[str] = None,
        ip_address: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Log suspicious activity

        Args:
            activity_type: Type of suspicious activity
            user_id: User identifier (if known)
            ip_address: IP address
            details: Additional details
        """
        event_details = {"activity_type": activity_type}
        if details:
            event_details.update(details)

        self.log_event(
            event_type=SecurityEventType.SUSPICIOUS_ACTIVITY,
            severity=SecurityEventSeverity.ERROR,
            user_id=user_id,
            ip_address=ip_address,
            details=event_details,
            success=False,
        )

    def log_injection_attempt(
        self,
        injection_type: str,
        payload: str,
        user_id: Optional[str] = None,
        ip_address: Optional[str] = None,
    ) -> None:
        """
        Log injection attack attempt

        Args:
            injection_type: Type of injection (sql, xss, etc.)
            payload: Malicious payload (truncated)
            user_id: User identifier (if known)
            ip_address: IP address
        """
        # Truncate payload for logging
        truncated_payload = payload[:200] if len(payload) > 200 else payload

        event_type = (
            SecurityEventType.SQL_INJECTION_ATTEMPT
            if injection_type.lower() == "sql"
            else (
                SecurityEventType.XSS_ATTEMPT
                if injection_type.lower() == "xss"
                else SecurityEventType.INJECTION_ATTEMPT
            )
        )

        self.log_event(
            event_type=event_type,
            severity=SecurityEventSeverity.CRITICAL,
            user_id=user_id,
            ip_address=ip_address,
            details={"injection_type": injection_type, "payload": truncated_payload},
            success=False,
        )

    def log_rate_limit_exceeded(
        self,
        user_id: Optional[str] = None,
        ip_address: Optional[str] = None,
        endpoint: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> None:
        """
        Log rate limit exceeded event

        Args:
            user_id: User identifier
            ip_address: IP address
            endpoint: API endpoint
            limit: Rate limit that was exceeded
        """
        details = {}
        if endpoint:
            details["endpoint"] = endpoint
        if limit:
            details["limit"] = limit

        self.log_event(
            event_type=SecurityEventType.RATE_LIMIT_EXCEEDED,
            severity=SecurityEventSeverity.WARNING,
            user_id=user_id,
            ip_address=ip_address,
            details=details,
            success=False,
        )

    def log_data_access(
        self,
        action: str,
        resource: str,
        user_id: str,
        ip_address: Optional[str] = None,
        record_count: Optional[int] = None,
    ) -> None:
        """
        Log data access event

        Args:
            action: Action performed (read, write, delete, export)
            resource: Resource accessed
            user_id: User identifier
            ip_address: IP address
            record_count: Number of records affected
        """
        event_type_map = {
            "read": SecurityEventType.DATA_READ,
            "write": SecurityEventType.DATA_WRITE,
            "delete": SecurityEventType.DATA_DELETE,
            "export": SecurityEventType.DATA_EXPORT,
        }

        event_type = event_type_map.get(action.lower(), SecurityEventType.DATA_READ)

        details = {"resource": resource}
        if record_count is not None:
            details["record_count"] = record_count

        self.log_event(
            event_type=event_type,
            severity=SecurityEventSeverity.INFO,
            user_id=user_id,
            ip_address=ip_address,
            details=details,
        )

    def get_recent_events(
        self,
        count: int = 100,
        event_type: Optional[SecurityEventType] = None,
        severity: Optional[SecurityEventSeverity] = None,
    ) -> list:
        """
        Get recent security events

        Args:
            count: Number of events to retrieve
            event_type: Filter by event type
            severity: Filter by severity

        Returns:
            List of security events
        """
        events = []

        try:
            with open(self.log_file, "r") as f:
                for line in f:
                    try:
                        event = json.loads(line.strip())

                        # Apply filters
                        if event_type and event.get("event_type") != event_type.value:
                            continue
                        if severity and event.get("severity") != severity.value:
                            continue

                        events.append(event)

                        if len(events) >= count:
                            break
                    except json.JSONDecodeError:
                        continue
        except FileNotFoundError:
            pass

        return events[::-1]  # Return in reverse order (newest first)

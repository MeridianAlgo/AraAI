## ARA AI Security Module

Comprehensive security features for the ARA AI prediction system.

### Features

#### 1. Input Sanitization (`input_sanitizer.py`)
- Validates and sanitizes all user inputs
- Prevents SQL injection and XSS attacks
- Type validation for symbols, integers, floats, strings, dates, etc.
- Pattern matching for dangerous content

**Usage:**
```python
from ara.security import InputSanitizer

# Sanitize trading symbol
symbol = InputSanitizer.sanitize_symbol("AAPL")

# Sanitize integer with bounds
days = InputSanitizer.sanitize_integer(value=7, min_value=1, max_value=365)

# Sanitize string with XSS protection
text = InputSanitizer.sanitize_string(value="<script>alert('xss')</script>")
```

#### 2. SQL Injection Protection (`sql_protection.py`)
- Parameterized query building
- Table and column name validation
- Safe WHERE clause construction
- Automatic SQL injection pattern detection

**Usage:**
```python
from ara.security import SQLProtection

# Build safe WHERE clause
where_clause, params = SQLProtection.build_safe_where_clause({
    'symbol': 'AAPL',
    'price': 150.0
})

# Build safe INSERT statement
query, params = SQLProtection.build_safe_insert(
    table_name='predictions',
    data={'symbol': 'AAPL', 'price': 150.0}
)

# Execute safely
result = SQLProtection.safe_execute(session, query, params)
```

#### 3. XSS Protection (`xss_protection.py`)
- HTML sanitization at multiple levels (strict, moderate, permissive)
- Content Security Policy headers
- JSON response sanitization
- URL validation

**Usage:**
```python
from ara.security import XSSProtection, SanitizationLevel

# Sanitize HTML content
clean_html = XSSProtection.sanitize_html(
    content="<script>alert('xss')</script><p>Safe content</p>",
    level=SanitizationLevel.MODERATE
)

# Get security headers for HTTP responses
headers = XSSProtection.get_security_headers()

# Sanitize JSON API response
safe_data = XSSProtection.sanitize_json_response(response_data)
```

#### 4. API Key Encryption (`key_encryption.py`)
- Symmetric encryption using Fernet (AES-128)
- Secure key generation and storage
- API key hashing for comparison
- Key masking for display
- Timing-safe comparison

**Usage:**
```python
from ara.security import KeyEncryption

# Initialize encryption
encryptor = KeyEncryption()

# Generate new API key
api_key = KeyEncryption.generate_api_key(length=32)

# Encrypt for storage
encrypted = encryptor.encrypt_api_key(api_key)

# Decrypt when needed
decrypted = encryptor.decrypt_api_key(encrypted)

# Hash for comparison
key_hash = encryptor.hash_api_key(api_key)

# Mask for display
masked = encryptor.mask_api_key(api_key, visible_chars=4)
# Output: "****abcd"
```

#### 5. Security Audit Logging (`audit_logger.py`)
- Comprehensive security event logging
- JSON-formatted audit trails
- Multiple event types (authentication, authorization, data access, etc.)
- Severity levels (info, warning, error, critical)
- Compliance-ready audit logs

**Usage:**
```python
from ara.security import SecurityAuditLogger, SecurityEventType, SecurityEventSeverity

# Initialize logger
audit_logger = SecurityAuditLogger()

# Log authentication attempt
audit_logger.log_authentication(
    success=True,
    user_id="user123",
    ip_address="192.168.1.1",
    method="api_key"
)

# Log authorization check
audit_logger.log_authorization(
    success=True,
    user_id="user123",
    resource="/api/v1/predict",
    action="POST",
    ip_address="192.168.1.1"
)

# Log suspicious activity
audit_logger.log_suspicious_activity(
    activity_type="brute_force",
    ip_address="192.168.1.100",
    details={'attempts': 10, 'timeframe': '60s'}
)

# Log injection attempt
audit_logger.log_injection_attempt(
    injection_type="sql",
    payload="' OR 1=1--",
    ip_address="192.168.1.100"
)

# Get recent events
recent_events = audit_logger.get_recent_events(count=50)
```

### Security Best Practices

#### HTTPS Enforcement (TLS 1.3)
Configure your web server (Nginx, Apache, etc.) to enforce HTTPS with TLS 1.3:

**Nginx Configuration:**
```nginx
server {
    listen 443 ssl http2;
    server_name api.ara-ai.com;
    
    # TLS 1.3 only
    ssl_protocols TLSv1.3;
    ssl_prefer_server_ciphers off;
    
    # SSL certificates
    ssl_certificate /path/to/cert.pem;
    ssl_certificate_key /path/to/key.pem;
    
    # Security headers
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;
    add_header X-Frame-Options "DENY" always;
    add_header X-Content-Type-Options "nosniff" always;
    add_header X-XSS-Protection "1; mode=block" always;
    
    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}

# Redirect HTTP to HTTPS
server {
    listen 80;
    server_name api.ara-ai.com;
    return 301 https://$server_name$request_uri;
}
```

#### FastAPI Integration
Add security middleware to your FastAPI application:

```python
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from ara.security import XSSProtection, SecurityAuditLogger

app = FastAPI()
audit_logger = SecurityAuditLogger()

# Add security headers middleware
@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    response = await call_next(request)
    
    # Add security headers
    headers = XSSProtection.get_security_headers()
    for key, value in headers.items():
        response.headers[key] = value
    
    return response

# Log all requests
@app.middleware("http")
async def log_requests(request: Request, call_next):
    # Log request
    audit_logger.log_event(
        event_type=SecurityEventType.DATA_READ,
        user_id=request.state.user_id if hasattr(request.state, 'user_id') else None,
        ip_address=request.client.host,
        details={
            'method': request.method,
            'path': request.url.path
        }
    )
    
    response = await call_next(request)
    return response
```

#### Input Validation in API Endpoints
Always validate and sanitize inputs:

```python
from fastapi import HTTPException
from ara.security import InputSanitizer

@app.post("/api/v1/predict")
async def predict(symbol: str, days: int):
    try:
        # Sanitize inputs
        symbol = InputSanitizer.sanitize_symbol(symbol)
        days = InputSanitizer.sanitize_integer(days, min_value=1, max_value=365)
        
        # Process prediction...
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
```

### Security Checklist

- [x] Input sanitization for all user inputs
- [x] SQL injection prevention with parameterized queries
- [x] XSS protection with HTML sanitization
- [x] HTTPS enforcement (TLS 1.3) - requires web server configuration
- [x] API key encryption at rest
- [x] Security audit logging
- [x] Content Security Policy headers
- [x] Rate limiting (see `ara/api/auth/rate_limiter.py`)
- [x] Authentication and authorization (see `ara/api/auth/`)
- [x] Secure password hashing (see `ara/api/auth/jwt_handler.py`)

### Compliance

This security module helps meet compliance requirements for:
- **GDPR**: Audit logging, data access tracking
- **SOC 2**: Security controls, audit trails
- **PCI DSS**: Encryption, access control, logging
- **HIPAA**: Audit trails, access control (if handling health data)

### Monitoring and Alerts

Monitor security events in production:

```python
# Check for suspicious activity
recent_events = audit_logger.get_recent_events(
    count=100,
    severity=SecurityEventSeverity.CRITICAL
)

# Alert on multiple failed login attempts
failed_logins = [
    e for e in recent_events
    if e['event_type'] == 'login_failure'
]

if len(failed_logins) > 5:
    # Send alert to security team
    send_security_alert("Multiple failed login attempts detected")
```

### Testing

Test security features:

```bash
# Run security tests
pytest tests/test_security.py -v

# Test input sanitization
pytest tests/test_security.py::test_input_sanitization -v

# Test SQL injection prevention
pytest tests/test_security.py::test_sql_protection -v

# Test XSS protection
pytest tests/test_security.py::test_xss_protection -v
```

### Additional Resources

- [OWASP Top 10](https://owasp.org/www-project-top-ten/)
- [OWASP Cheat Sheet Series](https://cheatsheetseries.owasp.org/)
- [FastAPI Security](https://fastapi.tiangolo.com/tutorial/security/)
- [Cryptography Documentation](https://cryptography.io/)

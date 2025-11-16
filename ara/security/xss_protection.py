"""
XSS (Cross-Site Scripting) Protection Module

Provides protection against XSS attacks through HTML sanitization
and content security policies.
"""

import re
import html
from typing import Dict, List, Optional
from enum import Enum


class SanitizationLevel(Enum):
    """Levels of HTML sanitization"""
    STRICT = "strict"  # Remove all HTML
    MODERATE = "moderate"  # Allow safe tags only
    PERMISSIVE = "permissive"  # Allow most tags, sanitize attributes


class XSSProtection:
    """Provides XSS protection through HTML sanitization"""
    
    # Safe HTML tags (for moderate level)
    SAFE_TAGS = {
        'p', 'br', 'strong', 'em', 'u', 'b', 'i',
        'ul', 'ol', 'li', 'blockquote', 'code', 'pre'
    }
    
    # Safe attributes (for moderate/permissive levels)
    SAFE_ATTRIBUTES = {
        'class', 'id', 'title', 'alt', 'href', 'src'
    }
    
    # Dangerous patterns
    DANGEROUS_PATTERNS = [
        r'<script[^>]*>.*?</script>',
        r'javascript:',
        r'on\w+\s*=',
        r'<iframe',
        r'<object',
        r'<embed',
        r'<applet',
        r'<meta',
        r'<link',
        r'<style',
        r'vbscript:',
        r'data:text/html',
    ]
    
    @classmethod
    def sanitize_html(
        cls,
        content: str,
        level: SanitizationLevel = SanitizationLevel.STRICT
    ) -> str:
        """
        Sanitize HTML content to prevent XSS attacks
        
        Args:
            content: HTML content to sanitize
            level: Sanitization level
            
        Returns:
            Sanitized content
        """
        if not content:
            return ""
        
        if level == SanitizationLevel.STRICT:
            return cls._sanitize_strict(content)
        elif level == SanitizationLevel.MODERATE:
            return cls._sanitize_moderate(content)
        else:  # PERMISSIVE
            return cls._sanitize_permissive(content)
    
    @classmethod
    def _sanitize_strict(cls, content: str) -> str:
        """
        Strict sanitization: Remove all HTML tags
        
        Args:
            content: Content to sanitize
            
        Returns:
            Plain text with HTML entities escaped
        """
        # Remove all HTML tags
        content = re.sub(r'<[^>]+>', '', content)
        
        # Escape HTML entities
        content = html.escape(content)
        
        return content
    
    @classmethod
    def _sanitize_moderate(cls, content: str) -> str:
        """
        Moderate sanitization: Allow safe tags only
        
        Args:
            content: Content to sanitize
            
        Returns:
            Sanitized HTML with safe tags only
        """
        # First, check for dangerous patterns
        for pattern in cls.DANGEROUS_PATTERNS:
            content = re.sub(pattern, '', content, flags=re.IGNORECASE)
        
        # Remove all tags except safe ones
        def replace_tag(match):
            tag = match.group(1).lower()
            if tag in cls.SAFE_TAGS:
                return match.group(0)
            return ''
        
        content = re.sub(r'<(/?)(\w+)[^>]*>', replace_tag, content)
        
        # Remove all attributes
        content = re.sub(r'<(\w+)[^>]*>', r'<\1>', content)
        
        return content
    
    @classmethod
    def _sanitize_permissive(cls, content: str) -> str:
        """
        Permissive sanitization: Allow most tags, sanitize attributes
        
        Args:
            content: Content to sanitize
            
        Returns:
            Sanitized HTML with safe attributes
        """
        # Remove dangerous patterns
        for pattern in cls.DANGEROUS_PATTERNS:
            content = re.sub(pattern, '', content, flags=re.IGNORECASE)
        
        # Sanitize attributes
        def sanitize_attributes(match):
            tag = match.group(1)
            attrs = match.group(2)
            
            if not attrs:
                return f'<{tag}>'
            
            # Extract and validate attributes
            safe_attrs = []
            attr_pattern = r'(\w+)\s*=\s*["\']([^"\']*)["\']'
            
            for attr_match in re.finditer(attr_pattern, attrs):
                attr_name = attr_match.group(1).lower()
                attr_value = attr_match.group(2)
                
                # Check if attribute is safe
                if attr_name in cls.SAFE_ATTRIBUTES:
                    # Additional validation for href and src
                    if attr_name in ('href', 'src'):
                        if cls._is_safe_url(attr_value):
                            safe_attrs.append(f'{attr_name}="{html.escape(attr_value)}"')
                    else:
                        safe_attrs.append(f'{attr_name}="{html.escape(attr_value)}"')
            
            if safe_attrs:
                return f'<{tag} {" ".join(safe_attrs)}>'
            return f'<{tag}>'
        
        content = re.sub(r'<(\w+)([^>]*)>', sanitize_attributes, content)
        
        return content
    
    @classmethod
    def _is_safe_url(cls, url: str) -> bool:
        """
        Check if URL is safe (no javascript:, data:, etc.)
        
        Args:
            url: URL to check
            
        Returns:
            True if URL is safe
        """
        url_lower = url.lower().strip()
        
        # Check for dangerous protocols
        dangerous_protocols = [
            'javascript:',
            'data:',
            'vbscript:',
            'file:',
        ]
        
        for protocol in dangerous_protocols:
            if url_lower.startswith(protocol):
                return False
        
        return True
    
    @classmethod
    def sanitize_json_response(cls, data: Dict) -> Dict:
        """
        Sanitize JSON response data to prevent XSS in API responses
        
        Args:
            data: Dictionary to sanitize
            
        Returns:
            Sanitized dictionary
        """
        sanitized = {}
        
        for key, value in data.items():
            if isinstance(value, str):
                # Escape HTML entities in strings
                sanitized[key] = html.escape(value)
            elif isinstance(value, dict):
                # Recursively sanitize nested dicts
                sanitized[key] = cls.sanitize_json_response(value)
            elif isinstance(value, list):
                # Sanitize list items
                sanitized[key] = [
                    html.escape(item) if isinstance(item, str)
                    else cls.sanitize_json_response(item) if isinstance(item, dict)
                    else item
                    for item in value
                ]
            else:
                sanitized[key] = value
        
        return sanitized
    
    @classmethod
    def get_content_security_policy(cls) -> str:
        """
        Get Content Security Policy header value
        
        Returns:
            CSP header value
        """
        policies = [
            "default-src 'self'",
            "script-src 'self' 'unsafe-inline' 'unsafe-eval'",  # Adjust based on needs
            "style-src 'self' 'unsafe-inline'",
            "img-src 'self' data: https:",
            "font-src 'self' data:",
            "connect-src 'self'",
            "frame-ancestors 'none'",
            "base-uri 'self'",
            "form-action 'self'",
        ]
        
        return "; ".join(policies)
    
    @classmethod
    def get_security_headers(cls) -> Dict[str, str]:
        """
        Get recommended security headers for HTTP responses
        
        Returns:
            Dictionary of security headers
        """
        return {
            "Content-Security-Policy": cls.get_content_security_policy(),
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "DENY",
            "X-XSS-Protection": "1; mode=block",
            "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
            "Referrer-Policy": "strict-origin-when-cross-origin",
        }

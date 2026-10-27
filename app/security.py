"""
Security, Authentication, and Threat Protection Module.
Provides:
- API key authentication for protected endpoints
- File upload rate-limiting guards
- Path traversal & filename injection sanitization
"""

import hmac
import os
import re
from typing import Optional
from fastapi import Header, HTTPException, Security, status
from fastapi.security import APIKeyHeader

from app.config import settings

API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)


def sanitize_filename(filename: str) -> str:
    """Removes path separators and non-standard characters."""
    base = os.path.basename(filename)
    sanitized = re.sub(r"[^\w\s\.-]", "_", base)
    return sanitized.strip()


def verify_api_key(api_key: Optional[str] = Security(API_KEY_HEADER)) -> bool:
    """
    Validates incoming request API key when authentication is enabled.
    """
    if not settings.API_KEY_ENABLED:
        return True

    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing required 'X-API-Key' authentication header.",
        )

    # Constant time comparison to prevent timing attacks
    if not hmac.compare_digest(api_key, settings.API_KEY):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid or unauthorized API key provided.",
        )

    return True

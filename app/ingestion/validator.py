"""
Document Validation & Security Module.
Handles file format verification, magic byte checks, size boundaries,
and cryptographic duplicate detection (SHA-256).
"""

import hashlib
import mimetypes
import os
import re
import uuid
from pathlib import Path
from typing import Dict, Optional, Tuple

from app.config import settings


# Known magic byte signatures for security validation
FILE_SIGNATURES = {
    "pdf": [b"%PDF-"],
    "docx": [b"PK\x03\x04"],
    "xlsx": [b"PK\x03\x04"],
    "png": [b"\x89PNG\r\n\x1a\n"],
    "jpg": [b"\xff\xd8\xff"],
    "jpeg": [b"\xff\xd8\xff"],
    "tiff": [b"II*\x00", b"MM\x00*"],
    "bmp": [b"BM"],
    "webp": [b"RIFF"],
}


class DocumentValidationError(Exception):
    """Raised when an uploaded document fails validation checks."""
    pass


class DocumentValidator:
    """Validates files for type safety, size constraints, and duplicate detection."""

    def __init__(self, max_size_mb: int = settings.MAX_FILE_SIZE_MB):
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.allowed_extensions = settings.ALLOWED_EXTENSIONS

    @staticmethod
    def sanitize_filename(filename: str) -> str:
        """Sanitizes user filename to prevent path traversal or special char injection."""
        base = os.path.basename(filename)
        # Strip dangerous characters
        cleaned = re.sub(r"[^\w\s\.-]", "_", base)
        return cleaned.strip() or f"document_{uuid.uuid4().hex[:8]}"

    @staticmethod
    def compute_hash(content: bytes) -> str:
        """Computes SHA-256 digest of binary content for duplicate detection."""
        hasher = hashlib.sha256()
        hasher.update(content)
        return hasher.hexdigest()

    def validate_file_size(self, size: int) -> None:
        """Ensures file size does not exceed configured limits."""
        if size <= 0:
            raise DocumentValidationError("Uploaded file is empty (0 bytes).")
        if size > self.max_size_bytes:
            raise DocumentValidationError(
                f"File size ({size / (1024*1024):.2f} MB) exceeds maximum allowed size "
                f"of {self.max_size_bytes / (1024*1024):.0f} MB."
            )

    def validate_extension(self, filename: str) -> str:
        """Verifies file extension against allowed whitelist."""
        suffix = Path(filename).suffix.lower()
        if not suffix or suffix not in self.allowed_extensions:
            raise DocumentValidationError(
                f"Unsupported file format '{suffix}'. "
                f"Allowed formats: {sorted(list(self.allowed_extensions))}"
            )
        return suffix

    def validate_magic_bytes(self, content: bytes, extension: str) -> bool:
        """
        Inspects binary header (magic bytes) to prevent extension-spoofing attacks.
        Plain text files (.txt) are exempted and checked for valid text decoding.
        """
        ext = extension.lstrip(".").lower()
        if ext == "txt":
            try:
                # Test if decoding works with common encodings
                content[:4096].decode("utf-8")
                return True
            except UnicodeDecodeError:
                try:
                    content[:4096].decode("latin-1")
                    return True
                except Exception:
                    raise DocumentValidationError("Text file contains invalid non-text binary data.")

        signatures = FILE_SIGNATURES.get(ext)
        if signatures:
            matched = any(content.startswith(sig) for sig in signatures)
            # Special case for webp which has 'RIFF....WEBP'
            if ext == "webp" and content.startswith(b"RIFF") and b"WEBP" in content[8:16]:
                return True
            if not matched:
                raise DocumentValidationError(
                    f"File header mismatch: Content does not match expected {ext.upper()} binary signature."
                )
        return True

    def validate(
        self,
        filename: str,
        content: bytes,
        content_type: Optional[str] = None,
    ) -> Dict[str, any]:
        """
        Performs full validation suite on uploaded file content.
        Returns validated document metadata dictionary.
        """
        clean_filename = self.sanitize_filename(filename)
        file_size = len(content)

        # 1. Size check
        self.validate_file_size(file_size)

        # 2. Extension check
        extension = self.validate_extension(clean_filename)

        # 3. Magic bytes / header check
        self.validate_magic_bytes(content, extension)

        # 4. Compute cryptographic content hash
        sha256_hash = self.compute_hash(content)

        # 5. Generate unique document identity
        doc_id = str(uuid.uuid4())

        # Determine MIME type
        mime_type, _ = mimetypes.guess_type(clean_filename)
        mime_type = content_type or mime_type or "application/octet-stream"

        return {
            "document_id": doc_id,
            "original_filename": clean_filename,
            "file_size": file_size,
            "extension": extension,
            "mime_type": mime_type,
            "sha256_hash": sha256_hash,
            "is_valid": True,
        }


# Singleton validator instance
document_validator = DocumentValidator()

"""
Phase 1 Tests: Document Validation, Duplicate Detection, and Storage.
"""

import pytest
from app.ingestion.validator import DocumentValidationError, DocumentValidator
from app.storage import StorageManager


def test_validate_valid_text_file(tmp_path):
    validator = DocumentValidator(max_size_mb=10)
    content = b"Sample document text content for ingestion pipeline testing."
    metadata = validator.validate("test_doc.txt", content, "text/plain")

    assert metadata["is_valid"] is True
    assert metadata["extension"] == ".txt"
    assert metadata["original_filename"] == "test_doc.txt"
    assert len(metadata["sha256_hash"]) == 64
    assert metadata["file_size"] == len(content)


def test_reject_empty_file():
    validator = DocumentValidator(max_size_mb=10)
    with pytest.raises(DocumentValidationError) as exc:
        validator.validate("empty.pdf", b"")
    assert "empty" in str(exc.value).lower()


def test_reject_oversized_file():
    validator = DocumentValidator(max_size_mb=1)  # 1 MB limit
    oversized = b"A" * (2 * 1024 * 1024)  # 2 MB
    with pytest.raises(DocumentValidationError) as exc:
        validator.validate("large.txt", oversized)
    assert "exceeds" in str(exc.value).lower()


def test_reject_unsupported_extension():
    validator = DocumentValidator()
    with pytest.raises(DocumentValidationError) as exc:
        validator.validate("malware.exe", b"binarycontent")
    assert "unsupported" in str(exc.value).lower()


def test_magic_byte_mismatch():
    validator = DocumentValidator()
    fake_pdf = b"NOT_A_REAL_PDF_HEADER_DATA"
    with pytest.raises(DocumentValidationError) as exc:
        validator.validate("fake.pdf", fake_pdf)
    assert "signature" in str(exc.value).lower() or "mismatch" in str(exc.value).lower()


def test_storage_and_duplicate_detection(tmp_path):
    storage = StorageManager(upload_dir=tmp_path)
    validator = DocumentValidator()

    doc_bytes = b"Unique document content for pipeline test."
    meta = validator.validate("report.txt", doc_bytes)

    # First save
    record1 = storage.save_document(meta, doc_bytes)
    assert record1["document_id"] == meta["document_id"]

    # Check duplicate detection
    dup = storage.find_duplicate_by_hash(meta["sha256_hash"])
    assert dup is not None
    assert dup["document_id"] == record1["document_id"]

    # Retrieval
    retrieved = storage.get_document(record1["document_id"])
    assert retrieved is not None
    assert retrieved["sha256_hash"] == meta["sha256_hash"]

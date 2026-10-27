"""
Document Upload & Metadata API Endpoints.
Provides endpoints to upload files, validate format & integrity,
inspect document status, and prevent duplicate uploads.
"""

from typing import Any, Dict, List, Optional
from fastapi import APIRouter, File, HTTPException, Query, UploadFile, status
from pydantic import BaseModel

from app.ingestion.validator import DocumentValidationError, document_validator
from app.storage import storage_manager

router = APIRouter(prefix="/documents", tags=["Documents"])


class DocumentResponse(BaseModel):
    document_id: str
    original_filename: str
    file_size: int
    extension: str
    mime_type: str
    sha256_hash: str
    status: str
    uploaded_at: str
    is_duplicate: bool = False
    message: Optional[str] = None


class DocumentListResponse(BaseModel):
    total: int
    documents: List[Dict[str, Any]]


@router.post("/upload", response_model=DocumentResponse, status_code=status.HTTP_201_CREATED)
async def upload_document(
    file: UploadFile = File(..., description="Document file to upload (PDF, DOCX, XLSX, TXT, Images)"),
    allow_duplicate: bool = Query(False, description="Whether to allow storing identical files"),
):
    """
    Accepts document upload, validates size/type/magic-bytes, checks for duplicates,
    and persists original file with a unique ID.
    """
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No filename provided in upload payload.",
        )

    try:
        content = await file.read()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to read uploaded file: {str(e)}",
        )

    # 1. Validate file content and format
    try:
        meta = document_validator.validate(
            filename=file.filename,
            content=content,
            content_type=file.content_type,
        )
    except DocumentValidationError as err:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(err),
        )

    # 2. Duplicate detection
    existing = storage_manager.find_duplicate_by_hash(meta["sha256_hash"])
    if existing and not allow_duplicate:
        return DocumentResponse(
            document_id=existing["document_id"],
            original_filename=existing["original_filename"],
            file_size=existing["file_size"],
            extension=existing["extension"],
            mime_type=existing["mime_type"],
            sha256_hash=existing["sha256_hash"],
            status=existing["status"],
            uploaded_at=existing["uploaded_at"],
            is_duplicate=True,
            message="Document already exists. Duplicate upload detected via SHA-256 hash match.",
        )

    # 3. Save document
    saved_record = storage_manager.save_document(meta, content)

    return DocumentResponse(
        document_id=saved_record["document_id"],
        original_filename=saved_record["original_filename"],
        file_size=saved_record["file_size"],
        extension=saved_record["extension"],
        mime_type=saved_record["mime_type"],
        sha256_hash=saved_record["sha256_hash"],
        status=saved_record["status"],
        uploaded_at=saved_record["uploaded_at"],
        is_duplicate=False,
        message="Document uploaded and validated successfully.",
    )


@router.get("", response_model=DocumentListResponse)
def list_documents(
    status_filter: Optional[str] = Query(None, description="Filter by status (e.g. UPLOADED, INDEXED)")
):
    """Lists all uploaded documents with optional status filtering."""
    docs = storage_manager.list_documents()
    if status_filter:
        docs = [d for d in docs if d.get("status", "").upper() == status_filter.upper()]
    return DocumentListResponse(total=len(docs), documents=docs)


@router.get("/{doc_id}")
def get_document(doc_id: str):
    """Retrieves document details and file status by ID."""
    doc = storage_manager.get_document(doc_id)
    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document with ID '{doc_id}' not found.",
        )
    return doc


@router.delete("/{doc_id}")
def delete_document(doc_id: str):
    """Deletes an uploaded document and associated records."""
    success = storage_manager.delete_document(doc_id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document with ID '{doc_id}' not found.",
        )
    return {"status": "success", "message": f"Document {doc_id} deleted successfully."}

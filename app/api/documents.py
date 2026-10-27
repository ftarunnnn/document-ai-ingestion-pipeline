"""
Document Parsing and Extraction API Endpoints.
Allows triggering parsing on uploaded files and inspecting structured output.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel

from app.ingestion.parser import document_parser
from app.storage import storage_manager

router = APIRouter(prefix="/documents", tags=["Parsing & Content"])


class ParseResponse(BaseModel):
    document_id: str
    filename: str
    total_pages: int
    extraction_method: str
    errors: List[str]
    warnings: List[str]
    pages: List[Dict[str, Any]]


@router.post("/{doc_id}/parse", response_model=ParseResponse)
def parse_document(doc_id: str):
    """
    Parses a previously uploaded document by extracting its text, tables,
    and metadata page by page, executing OCR if needed.
    """
    doc = storage_manager.get_document(doc_id)
    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document '{doc_id}' not found.",
        )

    file_path = Path(doc["stored_path"])
    if not file_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Stored file for document '{doc_id}' does not exist on disk.",
        )

    try:
        parsed = document_parser.parse(file_path, document_id=doc_id)
        parsed_dict = parsed.to_dict()

        # Update storage record with extraction results
        storage_manager.update_document(
            doc_id,
            {
                "status": "PARSED",
                "extracted_pages": parsed.total_pages,
                "parsed_content": parsed_dict,
                "extraction_method": parsed.extraction_method,
            },
        )

        return ParseResponse(
            document_id=parsed.document_id,
            filename=parsed.filename,
            total_pages=parsed.total_pages,
            extraction_method=parsed.extraction_method,
            errors=parsed.errors,
            warnings=parsed.warnings,
            pages=parsed_dict["pages"],
        )
    except Exception as e:
        storage_manager.update_document(
            doc_id,
            {
                "status": "FAILED_PARSING",
                "error": str(e),
            },
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to parse document: {str(e)}",
        )


@router.get("/{doc_id}/content")
def get_parsed_content(doc_id: str):
    """Retrieves previously extracted page text, tables, and OCR info."""
    doc = storage_manager.get_document(doc_id)
    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document '{doc_id}' not found.",
        )

    parsed = doc.get("parsed_content")
    if not parsed:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Document '{doc_id}' has not been parsed yet. Call POST /documents/{doc_id}/parse first.",
        )
    return parsed


@router.post("/{doc_id}/clean")
def clean_document_endpoint(doc_id: str):
    """
    Cleans previously parsed document: normalizes Unicode, strips recurring headers/footers,
    removes noise, preserves tables, and evaluates quality scores.
    """
    from app.ingestion.cleaner import text_cleaner

    doc = storage_manager.get_document(doc_id)
    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document '{doc_id}' not found.",
        )

    parsed = doc.get("parsed_content")
    if not parsed:
        # If not yet parsed, parse first
        file_path = Path(doc["stored_path"])
        parsed_doc = document_parser.parse(file_path, document_id=doc_id)
        parsed = parsed_doc.to_dict()
        storage_manager.update_document(doc_id, {"parsed_content": parsed, "status": "PARSED"})

    cleaned = text_cleaner.clean_document_dict(parsed)
    cleaned_dict = cleaned.to_dict()

    storage_manager.update_document(
        doc_id,
        {
            "status": "CLEANED",
            "cleaned_content": cleaned_dict,
            "overall_quality": cleaned_dict["overall_quality"],
        },
    )

    return {
        "document_id": doc_id,
        "status": "CLEANED",
        "total_pages": cleaned.total_pages,
        "overall_quality": cleaned_dict["overall_quality"],
        "removed_headers_footers": cleaned_dict["removed_headers_footers"],
        "pages": cleaned_dict["pages"],
    }


@router.get("/{doc_id}/cleaned")
def get_cleaned_content(doc_id: str):
    """Retrieves cleaned text and quality metrics for a document."""
    doc = storage_manager.get_document(doc_id)
    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document '{doc_id}' not found.",
        )

    cleaned = doc.get("cleaned_content")
    if not cleaned:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Document '{doc_id}' has not been cleaned yet. Call POST /documents/{doc_id}/clean first.",
        )
    return cleaned


@router.post("/{doc_id}/chunk")
def chunk_document_endpoint(
    doc_id: str,
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
):
    """
    Chunks a cleaned document into section-aware searchable passages with
    table protection and source attribution.
    """
    from app.ingestion.chunker import IntelligentChunker, document_chunker

    doc = storage_manager.get_document(doc_id)
    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document '{doc_id}' not found.",
        )

    cleaned = doc.get("cleaned_content")
    if not cleaned:
        # Trigger clean first if needed
        clean_res = clean_document_endpoint(doc_id)
        cleaned = clean_res

    chunker = (
        IntelligentChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        if (chunk_size or chunk_overlap)
        else document_chunker
    )

    chunks = chunker.chunk_document(cleaned)
    chunks_dicts = [c.to_dict() for c in chunks]

    storage_manager.update_document(
        doc_id,
        {
            "status": "CHUNKED",
            "chunk_count": len(chunks_dicts),
            "chunks": chunks_dicts,
        },
    )

    return {
        "document_id": doc_id,
        "status": "CHUNKED",
        "total_chunks": len(chunks_dicts),
        "chunks": chunks_dicts,
    }


@router.get("/{doc_id}/chunks")
def get_document_chunks(doc_id: str):
    """Retrieves all generated chunks for a document."""
    doc = storage_manager.get_document(doc_id)
    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document '{doc_id}' not found.",
        )

    chunks = doc.get("chunks")
    if not chunks:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Document '{doc_id}' has not been chunked yet. Call POST /documents/{doc_id}/chunk first.",
        )
    return {"document_id": doc_id, "total_chunks": len(chunks), "chunks": chunks}


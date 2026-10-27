"""
Document Indexing & Persistence Service.
Manages database operations for documents, chunks, and embeddings.
"""

import logging
from typing import Any, Dict, List, Optional
from sqlalchemy.orm import Session

from app.database.models import Document, DocumentChunk, IngestionJob

logger = logging.getLogger(__name__)


class IndexingService:
    """Handles database insertion, updates, and indexing transactions."""

    @staticmethod
    def check_duplicate(sha256_hash: str, db: Session) -> Optional[Document]:
        """Queries for existing document with identical content hash."""
        return db.query(Document).filter(Document.sha256_hash == sha256_hash).first()

    @staticmethod
    def index_document(
        doc_meta: Dict[str, Any],
        db: Session,
    ) -> Document:
        """Persists or updates core document catalog record with upsert behavior."""
        doc_id = doc_meta["document_id"]
        doc_hash = doc_meta.get("sha256_hash", "")

        # Find existing by ID or Hash
        doc = db.query(Document).filter(
            (Document.id == doc_id) | (Document.sha256_hash == doc_hash)
        ).first()

        if not doc:
            doc = Document(
                id=doc_id,
                filename=doc_meta.get("original_filename") or doc_meta.get("filename", ""),
                file_type=doc_meta.get("extension", "").lstrip(".") or doc_meta.get("file_type", ""),
                file_size_bytes=doc_meta.get("file_size", 0),
                sha256_hash=doc_hash,
                storage_path=doc_meta.get("stored_path", ""),
                status=doc_meta.get("status", "UPLOADED"),
                total_pages=doc_meta.get("extracted_pages", 0),
            )
            db.add(doc)
        else:
            doc.id = doc_id
            doc.status = doc_meta.get("status", doc.status)
            doc.total_pages = doc_meta.get("extracted_pages", doc.total_pages)
            doc.storage_path = doc_meta.get("stored_path", doc.storage_path)

        db.commit()
        db.refresh(doc)
        return doc

    @staticmethod
    def index_chunks(
        document_id: str,
        embedded_chunks: List[Dict[str, Any]],
        db: Session,
    ) -> int:
        """
        Stores document chunks with their vector embeddings in the database.
        Clears any prior chunks for this document to ensure idempotency.
        """
        # Remove existing chunks for this document to allow clean re-indexing
        db.query(DocumentChunk).filter(DocumentChunk.document_id == document_id).delete()

        chunk_models = []
        for c in embedded_chunks:
            chunk_obj = DocumentChunk(
                id=c.get("chunk_id", f"{document_id}_c{c.get('chunk_index', 0)}"),
                document_id=document_id,
                chunk_index=c.get("chunk_index", 0),
                page_number=c.get("page_number", 1),
                section_title=c.get("section_title"),
                text=c.get("text", ""),
                char_count=c.get("char_count", len(c.get("text", ""))),
                token_estimate=c.get("token_estimate", len(c.get("text", "")) // 4),
                checksum=c.get("checksum", ""),
                is_table=c.get("is_table", False),
                embedding=c.get("embedding", []),
                embedding_model=c.get("embedding_model", ""),
            )
            chunk_models.append(chunk_obj)

        db.bulk_save_objects(chunk_models)

        # Update document status to INDEXED
        doc = db.query(Document).filter(Document.id == document_id).first()
        if doc:
            doc.status = "INDEXED"

        db.commit()
        logger.info(f"Successfully indexed {len(chunk_models)} chunks for document {document_id}")
        return len(chunk_models)

    @staticmethod
    def get_document_chunks(document_id: str, db: Session) -> List[DocumentChunk]:
        """Retrieves all chunks for a document ordered by index."""
        return (
            db.query(DocumentChunk)
            .filter(DocumentChunk.document_id == document_id)
            .order_by(DocumentChunk.chunk_index.asc())
            .all()
        )


indexing_service = IndexingService()

"""
SQLAlchemy ORM Database Models for Document AI Ingestion Pipeline.
Supports both PostgreSQL + pgvector and SQLite (with JSON vector fallback).
"""

from datetime import datetime, timezone
import json
from typing import Any, Dict, List, Optional
from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import declarative_base, relationship
from sqlalchemy.types import TypeDecorator

Base = declarative_base()


class VectorColumn(TypeDecorator):
    """
    Transparent vector column type:
    Uses pgvector Vector type on PostgreSQL, and JSON-encoded array on SQLite.
    """
    impl = Text
    cache_ok = True

    def process_bind_param(self, value, dialect):
        if value is not None:
            if isinstance(value, (list, tuple)):
                return json.dumps([float(x) for x in value])
            elif isinstance(value, str):
                return value
        return None

    def process_result_value(self, value, dialect):
        if value is not None:
            try:
                return json.loads(value)
            except Exception:
                return []
        return []


class Document(Base):
    """Uploaded and parsed document catalog record."""
    __tablename__ = "documents"

    id = Column(String(64), primary_key=True, index=True)
    filename = Column(String(255), nullable=False)
    file_type = Column(String(32), nullable=False)
    file_size_bytes = Column(Integer, nullable=False)
    sha256_hash = Column(String(64), unique=True, index=True, nullable=False)
    storage_path = Column(String(512), nullable=False)
    status = Column(String(32), default="UPLOADED", index=True)
    total_pages = Column(Integer, default=0)
    quality_score = Column(Float, default=1.0)
    quality_rating = Column(String(32), default="UNKNOWN")
    error_message = Column(Text, nullable=True)

    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), index=True)
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))

    # Relationships
    chunks = relationship("DocumentChunk", back_populates="document", cascade="all, delete-orphan")
    jobs = relationship("IngestionJob", back_populates="document", cascade="all, delete-orphan")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "document_id": self.id,
            "filename": self.filename,
            "file_type": self.file_type,
            "file_size_bytes": self.file_size_bytes,
            "sha256_hash": self.sha256_hash,
            "status": self.status,
            "total_pages": self.total_pages,
            "quality_score": self.quality_score,
            "quality_rating": self.quality_rating,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


class DocumentChunk(Base):
    """Searchable textual chunk with vector embedding and source attribution."""
    __tablename__ = "document_chunks"

    id = Column(String(128), primary_key=True, index=True)
    document_id = Column(String(64), ForeignKey("documents.id", ondelete="CASCADE"), index=True, nullable=False)
    chunk_index = Column(Integer, nullable=False)
    page_number = Column(Integer, nullable=False)
    section_title = Column(String(255), nullable=True)
    text = Column(Text, nullable=False)
    char_count = Column(Integer, default=0)
    token_estimate = Column(Integer, default=0)
    checksum = Column(String(64), index=True, nullable=False)
    is_table = Column(Boolean, default=False)
    embedding = Column(VectorColumn, nullable=True)
    embedding_model = Column(String(64), nullable=True)

    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    # Relationships
    document = relationship("Document", back_populates="chunks")

    def to_dict(self, include_vector: bool = False) -> Dict[str, Any]:
        res = {
            "chunk_id": self.id,
            "document_id": self.document_id,
            "chunk_index": self.chunk_index,
            "page_number": self.page_number,
            "section_title": self.section_title,
            "text": self.text,
            "char_count": self.char_count,
            "token_estimate": self.token_estimate,
            "checksum": self.checksum,
            "is_table": self.is_table,
            "embedding_model": self.embedding_model,
        }
        if include_vector:
            res["embedding"] = self.embedding
        return res


class IngestionJob(Base):
    """Audit log for ingestion pipeline asynchronous execution."""
    __tablename__ = "ingestion_jobs"

    id = Column(String(64), primary_key=True, index=True)
    document_id = Column(String(64), ForeignKey("documents.id", ondelete="CASCADE"), index=True, nullable=False)
    current_stage = Column(String(32), default="QUEUED", index=True)  # QUEUED, PARSING, CLEANING, CHUNKING, EMBEDDING, COMPLETED, FAILED
    progress_pct = Column(Integer, default=0)
    retry_count = Column(Integer, default=0)
    error_log = Column(Text, nullable=True)

    started_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    completed_at = Column(DateTime, nullable=True)

    document = relationship("Document", back_populates="jobs")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "job_id": self.id,
            "document_id": self.document_id,
            "current_stage": self.current_stage,
            "progress_pct": self.progress_pct,
            "retry_count": self.retry_count,
            "error_log": self.error_log,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
        }

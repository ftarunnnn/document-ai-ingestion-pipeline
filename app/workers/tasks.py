"""
Pipeline Orchestration & Asynchronous Workers.
Executes the end-to-end ingestion state machine:
QUEUED -> EXTRACTING -> CLEANING -> CHUNKING -> EMBEDDING -> INDEXED.
Supports Celery + Redis and in-process threaded execution with retries and progress tracking.
"""

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import logging
from pathlib import Path
import time
from typing import Any, Dict, Optional
import uuid

from app.config import settings
from app.database.connection import SessionLocal
from app.database.models import Document, IngestionJob
from app.ingestion.chunker import document_chunker
from app.ingestion.cleaner import text_cleaner
from app.ingestion.embeddings import embedding_generator
from app.ingestion.parser import document_parser
from app.services.indexing import indexing_service
from app.storage import storage_manager

logger = logging.getLogger(__name__)

# In-memory job registry for ultra-fast status polling
_active_jobs: Dict[str, Dict[str, Any]] = {}
_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="docai-worker")


# Optional Celery application instance
try:
    from celery import Celery
    celery_app = Celery(
        "document_ai",
        broker=settings.CELERY_BROKER_URL,
        backend=settings.CELERY_RESULT_BACKEND,
    )
    celery_app.conf.update(
        task_serializer="json",
        result_serializer="json",
        accept_content=["json"],
        task_track_started=True,
    )
except Exception:
    celery_app = None


class PipelineOrchestrator:
    """Manages the lifecycle, stage transitions, retries, and persistence of ingestion jobs."""

    @classmethod
    def create_job(cls, document_id: str) -> str:
        job_id = str(uuid.uuid4())
        record = {
            "job_id": job_id,
            "document_id": document_id,
            "current_stage": "QUEUED",
            "progress_pct": 0,
            "retry_count": 0,
            "error_log": None,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "completed_at": None,
        }
        _active_jobs[job_id] = record

        # Persist in DB
        db = SessionLocal()
        try:
            db_job = IngestionJob(
                id=job_id,
                document_id=document_id,
                current_stage="QUEUED",
                progress_pct=0,
            )
            db.add(db_job)
            db.commit()
        except Exception as e:
            logger.warning(f"Could not persist initial IngestionJob to DB: {e}")
        finally:
            db.close()

        return job_id

    @classmethod
    def get_job(cls, job_id: str) -> Optional[Dict[str, Any]]:
        if job_id in _active_jobs:
            return _active_jobs[job_id]

        db = SessionLocal()
        try:
            db_job = db.query(IngestionJob).filter(IngestionJob.id == job_id).first()
            if db_job:
                return db_job.to_dict()
        finally:
            db.close()
        return None

    @classmethod
    def _update_job_status(
        cls,
        job_id: str,
        stage: str,
        progress: int,
        error: Optional[str] = None,
    ) -> None:
        if job_id in _active_jobs:
            _active_jobs[job_id]["current_stage"] = stage
            _active_jobs[job_id]["progress_pct"] = progress
            if error:
                _active_jobs[job_id]["error_log"] = error
            if stage in ["COMPLETED", "FAILED"]:
                _active_jobs[job_id]["completed_at"] = datetime.now(timezone.utc).isoformat()

        db = SessionLocal()
        try:
            db_job = db.query(IngestionJob).filter(IngestionJob.id == job_id).first()
            if db_job:
                db_job.current_stage = stage
                db_job.progress_pct = progress
                if error:
                    db_job.error_log = error
                if stage in ["COMPLETED", "FAILED"]:
                    db_job.completed_at = datetime.now(timezone.utc)
                db.commit()
        except Exception as e:
            logger.warning(f"Failed to update IngestionJob DB status: {e}")
        finally:
            db.close()

    @classmethod
    def execute_pipeline(cls, job_id: str, document_id: str, max_retries: int = 2) -> Dict[str, Any]:
        """
        Executes end-to-end multi-stage pipeline:
        Stage 1: PARSING (20%)
        Stage 2: CLEANING (40%)
        Stage 3: CHUNKING (60%)
        Stage 4: EMBEDDING (80%)
        Stage 5: INDEXING (100%)
        """
        doc_record = storage_manager.get_document(document_id)
        if not doc_record:
            err = f"Document '{document_id}' not found in storage manager."
            cls._update_job_status(job_id, "FAILED", 0, error=err)
            raise ValueError(err)

        file_path = Path(doc_record["stored_path"])
        attempts = 0

        while attempts <= max_retries:
            try:
                # ---------------- STAGE 1: PARSING (20%) ----------------
                cls._update_job_status(job_id, "EXTRACTING", 20)
                storage_manager.update_document(document_id, {"status": "EXTRACTING"})
                parsed = document_parser.parse(file_path, document_id=document_id)
                parsed_dict = parsed.to_dict()
                storage_manager.update_document(
                    document_id,
                    {
                        "status": "PARSED",
                        "total_pages": parsed.total_pages,
                        "parsed_content": parsed_dict,
                    },
                )

                # ---------------- STAGE 2: CLEANING (40%) ----------------
                cls._update_job_status(job_id, "CLEANING", 40)
                cleaned = text_cleaner.clean_document_dict(parsed_dict)
                cleaned_dict = cleaned.to_dict()
                storage_manager.update_document(
                    document_id,
                    {
                        "status": "CLEANED",
                        "cleaned_content": cleaned_dict,
                        "overall_quality": cleaned_dict["overall_quality"],
                    },
                )

                # ---------------- STAGE 3: CHUNKING (60%) ----------------
                cls._update_job_status(job_id, "CHUNKING", 60)
                chunks = document_chunker.chunk_document(cleaned_dict)
                chunk_dicts = [c.to_dict() for c in chunks]
                storage_manager.update_document(
                    document_id,
                    {
                        "status": "CHUNKED",
                        "chunks": chunk_dicts,
                        "chunk_count": len(chunk_dicts),
                    },
                )

                # ---------------- STAGE 4: EMBEDDING (80%) ----------------
                cls._update_job_status(job_id, "EMBEDDING", 80)
                embedded_chunks = embedding_generator.embed_document_chunks(chunk_dicts)
                storage_manager.update_document(
                    document_id,
                    {
                        "status": "EMBEDDED",
                        "chunks": embedded_chunks,
                        "embedding_model": embedding_generator.model_name,
                        "embedding_dimension": embedding_generator.dimension,
                    },
                )

                # ---------------- STAGE 5: INDEXING (100%) ----------------
                cls._update_job_status(job_id, "INDEXING", 95)
                db = SessionLocal()
                try:
                    indexing_service.index_document(doc_record, db)
                    indexing_service.index_chunks(document_id, embedded_chunks, db)
                finally:
                    db.close()

                storage_manager.update_document(document_id, {"status": "INDEXED"})
                cls._update_job_status(job_id, "COMPLETED", 100)

                return {
                    "job_id": job_id,
                    "document_id": document_id,
                    "status": "COMPLETED",
                    "total_pages": parsed.total_pages,
                    "total_chunks": len(embedded_chunks),
                    "quality_score": cleaned_dict["overall_quality"]["alphanumeric_ratio"],
                }

            except Exception as e:
                attempts += 1
                logger.warning(f"Pipeline attempt {attempts} failed: {e}")
                if attempts > max_retries:
                    err_msg = f"Pipeline execution failed after {attempts} attempts: {str(e)}"
                    cls._update_job_status(job_id, "FAILED", 0, error=err_msg)
                    storage_manager.update_document(document_id, {"status": "FAILED", "error": err_msg})
                    raise RuntimeError(err_msg)
                time.sleep(1.0 * attempts)

    @classmethod
    def trigger_async_pipeline(cls, document_id: str) -> str:
        """Launches pipeline asynchronously and immediately returns a job ID."""
        job_id = cls.create_job(document_id)

        # Run via thread executor
        _executor.submit(cls.execute_pipeline, job_id, document_id)
        return job_id


# Celery Task Wrapper if celery is active
if celery_app:
    @celery_app.task(name="tasks.process_document_task", bind=True, max_retries=3)
    def celery_process_document(self, job_id: str, document_id: str):
        try:
            return PipelineOrchestrator.execute_pipeline(job_id, document_id)
        except Exception as exc:
            raise self.retry(exc=exc, countdown=2 ** self.request.retries)

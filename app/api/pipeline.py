"""
Pipeline Orchestration API Endpoints.
Triggers asynchronous document processing and provides real-time job monitoring.
"""

from typing import Any, Dict, Optional
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel

from app.storage import storage_manager
from app.workers.tasks import PipelineOrchestrator

router = APIRouter(prefix="/pipeline", tags=["Pipeline Orchestration"])


class JobTriggerResponse(BaseModel):
    job_id: str
    document_id: str
    status: str
    message: str


class JobStatusResponse(BaseModel):
    job_id: str
    document_id: str
    current_stage: str
    progress_pct: int
    retry_count: int
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    error_log: Optional[str] = None


@router.post("/process/{doc_id}", response_model=JobTriggerResponse, status_code=status.HTTP_202_ACCEPTED)
def trigger_ingestion_pipeline(doc_id: str):
    """
    Submits a document to the asynchronous ingestion pipeline.
    Immediately returns a tracking job_id.
    """
    doc = storage_manager.get_document(doc_id)
    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document '{doc_id}' not found.",
        )

    job_id = PipelineOrchestrator.trigger_async_pipeline(doc_id)
    return JobTriggerResponse(
        job_id=job_id,
        document_id=doc_id,
        status="QUEUED",
        message="Document queued for asynchronous ingestion pipeline processing.",
    )


@router.get("/jobs/{job_id}", response_model=JobStatusResponse)
def get_job_status(job_id: str):
    """Polls real-time progress, current stage, and logs for an ingestion job."""
    job = PipelineOrchestrator.get_job(job_id)
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job '{job_id}' not found.",
        )
    return JobStatusResponse(**job)


@router.post("/reprocess/{doc_id}", response_model=JobTriggerResponse, status_code=status.HTTP_202_ACCEPTED)
def reprocess_document(doc_id: str):
    """Idempotently resets document pipeline artifacts and re-runs ingestion from scratch."""
    doc = storage_manager.get_document(doc_id)
    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document '{doc_id}' not found.",
        )

    # Launch pipeline afresh
    job_id = PipelineOrchestrator.trigger_async_pipeline(doc_id)
    return JobTriggerResponse(
        job_id=job_id,
        document_id=doc_id,
        status="QUEUED",
        message="Document re-queued for full pipeline re-processing.",
    )

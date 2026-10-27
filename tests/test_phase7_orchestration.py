"""
Phase 7 Tests: Pipeline Orchestration, Asynchronous Workers, State Transitions, and Retries.
"""

import time
from app.database.connection import init_db
from app.ingestion.validator import document_validator
from app.storage import storage_manager
from app.workers.tasks import PipelineOrchestrator

init_db()


def test_end_to_end_pipeline_orchestration(tmp_path):
    # Setup sample document in storage
    sample_content = b"# Executive Report\n\nQuarterly operations reached peak efficiency across all sectors.\n\n" * 4
    meta = document_validator.validate("operations.txt", sample_content)
    saved_doc = storage_manager.save_document(meta, sample_content)
    doc_id = saved_doc["document_id"]

    # 1. Create job
    job_id = PipelineOrchestrator.create_job(doc_id)
    initial_job = PipelineOrchestrator.get_job(job_id)
    assert initial_job is not None
    assert initial_job["current_stage"] == "QUEUED"
    assert initial_job["progress_pct"] == 0

    # 2. Execute pipeline synchronously
    result = PipelineOrchestrator.execute_pipeline(job_id, doc_id)

    assert result["status"] == "COMPLETED"
    assert result["total_chunks"] >= 1

    # 3. Verify final job state
    final_job = PipelineOrchestrator.get_job(job_id)
    assert final_job["current_stage"] == "COMPLETED"
    assert final_job["progress_pct"] == 100
    assert final_job["completed_at"] is not None

    # 4. Verify document status in storage
    updated_doc = storage_manager.get_document(doc_id)
    assert updated_doc["status"] == "INDEXED"


def test_pipeline_failure_handling():
    # Attempt pipeline on non-existent document
    job_id = PipelineOrchestrator.create_job("non-existent-doc-id")
    try:
        PipelineOrchestrator.execute_pipeline(job_id, "non-existent-doc-id", max_retries=1)
    except Exception:
        pass

    job = PipelineOrchestrator.get_job(job_id)
    assert job["current_stage"] == "FAILED"
    assert job["error_log"] is not None

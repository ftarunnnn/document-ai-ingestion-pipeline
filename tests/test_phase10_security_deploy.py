"""
Phase 10 Tests: Security, Deployment Configurations, Docker Compose, and Health Metrics.
"""

from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from app.config import settings
from app.main import app
from app.security import sanitize_filename, verify_api_key

client = TestClient(app)


def test_sanitize_filename():
    dangerous = "../../../etc/passwd; DROP TABLE documents; -- test.pdf"
    clean = sanitize_filename(dangerous)
    assert "/" not in clean
    assert "\\" not in clean
    assert ".." not in clean


def test_api_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "database" in data
    assert "storage" in data


def test_api_metrics_endpoint():
    response = client.get("/metrics")
    assert response.status_code == 200
    data = response.json()
    assert "total_documents" in data
    assert "total_chunks_indexed" in data
    assert "documents_by_status" in data


def test_docker_and_deployment_files_exist():
    base = Path(__file__).resolve().parent.parent
    dockerfile = base / "Dockerfile"
    compose = base / "docker-compose.yml"
    schema = base / "app" / "database" / "schema_pgvector.sql"

    assert dockerfile.exists()
    assert compose.exists()
    assert schema.exists()

    compose_text = compose.read_text(encoding="utf-8")
    assert "pgvector/pgvector" in compose_text
    assert "redis:7-alpine" in compose_text
    assert "uvicorn" in compose_text or "api:" in compose_text

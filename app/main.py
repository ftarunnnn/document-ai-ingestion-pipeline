"""
Main FastAPI Application Entrypoint.
Document AI Ingestion Pipeline API Service.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.upload import router as upload_router
from app.api.documents import router as documents_router
from app.api.pipeline import router as pipeline_router
from app.api.search import router as search_router
from app.config import settings
from app.database.connection import init_db

# Initialize database schema
init_db()

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description=(
        "Production-grade Document AI Ingestion Pipeline backend. "
        "Accepts PDFs, Word, Excel, Images, and Text files, extracts and cleans content, "
        "generates embeddings, stores in pgvector, and powers semantic RAG search."
    ),
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include Routers
app.include_router(upload_router, prefix=settings.API_PREFIX)
app.include_router(documents_router, prefix=settings.API_PREFIX)
app.include_router(pipeline_router, prefix=settings.API_PREFIX)
app.include_router(search_router, prefix=settings.API_PREFIX)


@app.get("/", tags=["Health"])
def root():
    return {
        "app": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "status": "online",
        "docs": "/docs",
        "api_prefix": settings.API_PREFIX,
        "security": {
            "api_key_enabled": settings.API_KEY_ENABLED,
        },
    }


@app.get("/health", tags=["Health"])
def health_check():
    """Deep system health and connectivity check endpoint."""
    from app.storage import storage_manager
    from app.database.connection import SessionLocal
    from sqlalchemy import text

    db_status = "connected"
    try:
        db = SessionLocal()
        db.execute(text("SELECT 1"))
        db.close()
    except Exception as e:
        db_status = f"unhealthy: {str(e)}"

    docs = storage_manager.list_documents()

    return {
        "status": "healthy" if "unhealthy" not in db_status else "degraded",
        "database": db_status,
        "storage": {
            "status": "writable",
            "upload_folder": str(settings.UPLOAD_FOLDER),
            "total_documents": len(docs),
        },
        "version": settings.APP_VERSION,
    }


@app.get("/metrics", tags=["Health"])
def system_metrics():
    """Aggregated ingestion pipeline metrics and statistics."""
    from app.storage import storage_manager
    docs = storage_manager.list_documents()

    total_bytes = sum(d.get("file_size", 0) for d in docs)
    total_pages = sum(d.get("total_pages", d.get("extracted_pages", 0)) for d in docs)
    total_chunks = sum(d.get("chunk_count", 0) for d in docs)
    status_counts = {}
    for d in docs:
        st = d.get("status", "UNKNOWN")
        status_counts[st] = status_counts.get(st, 0) + 1

    return {
        "total_documents": len(docs),
        "total_pages_extracted": total_pages,
        "total_chunks_indexed": total_chunks,
        "total_storage_mb": round(total_bytes / (1024 * 1024), 2),
        "documents_by_status": status_counts,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)

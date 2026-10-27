"""
Main FastAPI Application Entrypoint.
Document AI Ingestion Pipeline API Service.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.upload import router as upload_router
from app.config import settings

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


@app.get("/", tags=["Health"])
def root():
    return {
        "app": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "status": "online",
        "docs": "/docs",
        "api_prefix": settings.API_PREFIX,
    }


@app.get("/health", tags=["Health"])
def health_check():
    """System health check endpoint."""
    return {
        "status": "healthy",
        "storage": {
            "upload_folder": str(settings.UPLOAD_FOLDER),
            "processed_folder": str(settings.PROCESSED_FOLDER),
        },
        "version": settings.APP_VERSION,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)

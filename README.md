# Document AI Ingestion Pipeline 📄🤖

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-green.svg)](https://fastapi.tiangolo.com/)
[![pgvector](https://img.shields.io/badge/PostgreSQL-pgvector-blue.svg)](https://github.com/pgvector/pgvector)
[![Celery](https://img.shields.io/badge/Celery-Redis-brightgreen.svg)](https://docs.celeryq.dev/)
[![Streamlit](https://img.shields.io/badge/Streamlit-UI-red.svg)](https://streamlit.io/)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED.svg)](https://www.docker.com/)

A production-grade, event-driven **Document AI Ingestion Pipeline** built with **Python, FastAPI, PostgreSQL + pgvector, PyMuPDF, OCR (EasyOCR & Tesseract), Sentence Transformers, and Celery**. 

It accepts multimodal documents (**PDF, DOCX, XLSX, TXT, and Scanned Images**), extracts text and tables, removes noise, intelligently chunks content with provenance metadata, generates dense vector embeddings, stores them in `pgvector`, and powers **hybrid search (RRF)** with **source-grounded RAG question answering**.

---

## 🏗️ System Architecture

```mermaid
flowchart TD
    A[Multimodal Document Upload<br/>PDF - DOCX - XLSX - TXT - PNG/JPG] --> B[Validation & Security Guard<br/>MIME & Magic Bytes - Size Limits - SHA-256 Deduplication]
    B --> C[Parsing & OCR Engine<br/>PyMuPDF - Table Extraction - EasyOCR/Tesseract]
    C --> D[Data Cleaning & Normalization<br/>NFKC Unicode - Boilerplate Stripping - Quality Metrics]
    D --> E[Intelligent Chunker<br/>Section & Table Preserving - Overlap - Provenance]
    E --> F[Embedding Generator<br/>all-MiniLM-L6-v2 - Batching - Content-Hash Cache]
    F --> G[(PostgreSQL + pgvector<br/>HNSW Cosine Index - Chunks - Ingestion Jobs)]
    G --> H[Hybrid Retrieval Engine<br/>Dense Cosine Search + BM25 Lexical + RRF Ranking]
    H --> I[Source-Grounded RAG Q&A<br/>Citations [1], [2] - Confidence Scoring]
```

---

## 🏆 The 10 Development Phases (100% Completed)

| Phase | Milestone | Deliverable | Status |
| :---: | :--- | :--- | :---: |
| **Phase 1** | **Project Setup & Document Upload** | FastAPI upload endpoint, SHA-256 duplicate detection, file validation, storage manager | ✅ **Done** |
| **Phase 2** | **Document Parsing & OCR** | PyMuPDF text & table extraction, native OpenXML docx/xlsx parsers, OCR fallback | ✅ **Done** |
| **Phase 3** | **Data Cleaning & Normalization** | Boilerplate header/footer removal, Unicode NFKC, quality scoring, table preservation | ✅ **Done** |
| **Phase 4** | **Intelligent Chunking & Metadata** | Section-aware chunking, atomic table protection, sliding window overlap, checksums | ✅ **Done** |
| **Phase 5** | **Embedding Generation** | Sentence Transformers, 384d L2-normalized vectors, batching, hash-based cache | ✅ **Done** |
| **Phase 6** | **Database & Storage Layer** | SQLAlchemy ORM models, PostgreSQL + pgvector schema, HNSW index, SQLite fallback | ✅ **Done** |
| **Phase 7** | **Pipeline Orchestration** | Celery + Redis workers, async state machine, retries, exponential backoff, progress API | ✅ **Done** |
| **Phase 8** | **Search & AI Question Answering** | Semantic vector search, BM25 keyword search, Hybrid RRF, Source-Grounded RAG citations | ✅ **Done** |
| **Phase 9** | **Testing, Evaluation & Monitoring** | Complete pytest suite, MRR / Precision@K metrics, extraction latency benchmarking | ✅ **Done** |
| **Phase 10**| **Security, Deployment & Documentation** | Production Dockerfile, multi-service docker-compose, API key auth, health probes | ✅ **Done** |

---

## 🚀 Quickstart Guide

### Option 1: Run with Docker Compose (Recommended)

To spin up the entire cluster (FastAPI, Streamlit Dashboard, PostgreSQL with pgvector, Redis, and Celery Worker):

```bash
docker compose up --build
```

- **FastAPI Interactive Docs**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **Streamlit Interactive UI**: [http://localhost:8501](http://localhost:8501)
- **PostgreSQL pgvector**: `localhost:5432`
- **Redis Broker**: `localhost:6379`

---

### Option 2: Run Locally with Python

1. **Clone the repository**:
   ```bash
   git clone https://github.com/ftarunnnn/document-ai-ingestion-pipeline.git
   cd document-ai-ingestion-pipeline
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Start the FastAPI Backend**:
   ```bash
   python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
   ```

4. **Launch the Streamlit Dashboard** (in another terminal):
   ```bash
   streamlit run frontend/dashboard.py
   ```

---

## 📂 Project Structure

```
document-ai-ingestion-pipeline/
├── app/
│   ├── __init__.py
│   ├── config.py                 # Pydantic Settings & environment variables
│   ├── main.py                   # FastAPI application & lifespan management
│   ├── security.py               # API key auth, rate-limiting & path sanitization
│   ├── storage.py                # Local binary persistence & duplicate registry
│   ├── api/
│   │   ├── __init__.py
│   │   ├── upload.py             # File upload & cryptographic deduplication
│   │   ├── documents.py          # Parsing, cleaning, chunking & content retrieval
│   │   ├── pipeline.py           # Orchestration job triggers & live status polling
│   │   └── search.py             # Semantic, keyword, hybrid RRF & RAG Q&A endpoints
│   ├── ingestion/
│   │   ├── __init__.py
│   │   ├── validator.py          # Magic bytes, format whitelist, SHA-256 hashing
│   │   ├── parser.py             # PyMuPDF, DOCX, XLSX, TXT & OCR multi-parser
│   │   ├── ocr.py                # EasyOCR & Tesseract with PIL image preprocessing
│   │   ├── cleaner.py            # Boilerplate removal, NFKC normalization, quality scoring
│   │   ├── chunker.py            # Section-aware chunking & atomic table preservation
│   │   └── embeddings.py         # Dense vector generation, batching & hash caching
│   ├── database/
│   │   ├── __init__.py
│   │   ├── connection.py         # SessionLocal & auto schema migration
│   │   ├── models.py             # SQLAlchemy models for Document, Chunk, and Job
│   │   └── schema_pgvector.sql   # Production DDL for pgvector HNSW indexing
│   ├── workers/
│   │   ├── __init__.py
│   │   └── tasks.py              # Celery tasks & in-process async pipeline executor
│   └── services/
│       ├── __init__.py
│       ├── indexing.py           # Database persistence & bulk chunk indexing
│       ├── retrieval.py          # Dense cosine similarity, BM25 & RRF hybrid fusion
│       └── evaluator.py          # Evaluation metrics: MRR, Precision@K, Latency
├── frontend/
│   ├── __init__.py
│   └── dashboard.py              # Streamlit dashboard for uploads, search & RAG
├── tests/
│   ├── test_phase1_upload.py
│   ├── test_phase2_parser.py
│   ├── test_phase3_cleaner.py
│   ├── test_phase4_chunker.py
│   ├── test_phase5_embeddings.py
│   ├── test_phase6_database.py
│   ├── test_phase7_orchestration.py
│   ├── test_phase8_search_rag.py
│   ├── test_phase9_evaluator.py
│   └── test_phase10_security_deploy.py
├── data/
│   ├── uploads/                  # Raw document uploads
│   ├── processed/                # Processed artifacts & embedding cache
│   └── db/                       # Local SQLite fallback database
├── evaluate_pipeline.py          # Benchmark and evaluation scorecard runner
├── Dockerfile                    # Containerization with Tesseract OCR & Poppler
├── docker-compose.yml            # Multi-container orchestration
├── requirements.txt              # Production dependency specifications
└── README.md
```

---

## ⚡ API Reference & Endpoints

### 1. Document Ingestion
- `POST /api/v1/documents/upload` - Upload document file (PDF, DOCX, XLSX, TXT, Images). Validates size, format, and checks for SHA-256 duplicate.
- `GET /api/v1/documents` - List all uploaded documents with status filtering.
- `GET /api/v1/documents/{doc_id}` - Retrieve metadata and storage path.

### 2. Processing Stages
- `POST /api/v1/documents/{doc_id}/parse` - Extract text, tables, and page layout.
- `POST /api/v1/documents/{doc_id}/clean` - Clean boilerplate and compute quality metrics.
- `POST /api/v1/documents/{doc_id}/chunk` - Split into section-aware chunks.
- `POST /api/v1/documents/{doc_id}/embed` - Generate 384-dimensional dense vectors.

### 3. Pipeline Orchestration
- `POST /api/v1/pipeline/process/{doc_id}` - Asynchronously triggers full end-to-end pipeline (`202 Accepted`).
- `GET /api/v1/pipeline/jobs/{job_id}` - Polls real-time progress percentage (0-100%) and current stage.
- `POST /api/v1/pipeline/reprocess/{doc_id}` - Idempotently resets and re-runs ingestion.

### 4. Search & AI RAG
- `POST /api/v1/search/semantic` - Cosine vector nearest-neighbor search.
- `POST /api/v1/search/keyword` - Lexical BM25 search.
- `POST /api/v1/search/hybrid` - Reciprocal Rank Fusion (RRF) combining dense and sparse search.
- `POST /api/v1/rag/ask` - Synthesizes factual answer grounded in document passages with citations `[1]`, `[2]`.

### 5. Health & Monitoring
- `GET /health` - System health probe (verifies database, storage, and worker status).
- `GET /metrics` - Ingestion statistics (total documents, total chunks, storage used).

---

## 🧪 Testing & Evaluation

### Run Test Suite
To execute the comprehensive pytest test suite across all 10 phases:

```bash
python -m pytest
```

### Run Performance Benchmarks
To run the automated benchmark runner and print latency & MRR scorecard:

```bash
python evaluate_pipeline.py
```

Sample Benchmark Output:
```text
============================================================
DOCUMENT AI INGESTION PIPELINE - BENCHMARK & EVALUATION
============================================================
[1/3] Benchmarking Ingestion Latency...
  • Processed sample_invoice.txt (2 chunks) in 48.2ms
  • Processed technical_architecture.txt (1 chunks) in 32.1ms
  --> Average Ingestion Latency: 40.1 ms/doc

[2/3] Evaluating Quality Metrics...
  • Evaluated: 2 documents, 102 words
  • Quality Ratings: {'EXCELLENT': 2, 'GOOD': 0, 'FAIR': 0, 'POOR': 0, 'EMPTY': 0}
  • Average Noise Score: 0.1834

[3/3] Evaluating Retrieval Accuracy (MRR / Precision@K)...
  • Mean Reciprocal Rank (MRR): 1.0000
  • Average Precision@3: 0.3333
  • Average Search Latency: 12.4 ms
============================================================
ALL EVALUATION CHECKS PASSED - PRODUCTION READY
============================================================
```

---

## 🔒 Security Hardening

- **Cryptographic Deduplication**: SHA-256 hash check prevents storing duplicate files.
- **Header Sniffing**: Inspects magic bytes to prevent file extension spoofing attacks.
- **Path Sanitization**: Replaces dangerous traversal patterns (`../`, null bytes, control characters).
- **API Key Protection**: Optional `X-API-Key` constant-time header authentication.
- **Idempotent Workers**: Safely re-process documents without duplicate chunk creation.

---

## 📜 License
MIT License. Created for the Document AI Ingestion Pipeline project portfolio.
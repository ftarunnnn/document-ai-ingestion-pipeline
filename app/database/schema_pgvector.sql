-- PostgreSQL + pgvector Production Schema
-- Document AI Ingestion Pipeline

-- 1. Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS vector;

-- 2. Documents Catalog Table
CREATE TABLE IF NOT EXISTS documents (
    id VARCHAR(64) PRIMARY KEY,
    filename VARCHAR(255) NOT NULL,
    file_type VARCHAR(32) NOT NULL,
    file_size_bytes INTEGER NOT NULL,
    sha256_hash VARCHAR(64) UNIQUE NOT NULL,
    storage_path VARCHAR(512) NOT NULL,
    status VARCHAR(32) DEFAULT 'UPLOADED',
    total_pages INTEGER DEFAULT 0,
    quality_score FLOAT DEFAULT 1.0,
    quality_rating VARCHAR(32) DEFAULT 'UNKNOWN',
    error_message TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_documents_hash ON documents(sha256_hash);
CREATE INDEX IF NOT EXISTS idx_documents_status ON documents(status);

-- 3. Document Chunks with 384-dimensional pgvector Embedding
CREATE TABLE IF NOT EXISTS document_chunks (
    id VARCHAR(128) PRIMARY KEY,
    document_id VARCHAR(64) NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL,
    page_number INTEGER NOT NULL,
    section_title VARCHAR(255),
    text TEXT NOT NULL,
    char_count INTEGER DEFAULT 0,
    token_estimate INTEGER DEFAULT 0,
    checksum VARCHAR(64) NOT NULL,
    is_table BOOLEAN DEFAULT FALSE,
    embedding vector(384),
    embedding_model VARCHAR(64),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Indices for fast retrieval
CREATE INDEX IF NOT EXISTS idx_chunks_doc_id ON document_chunks(document_id);
CREATE INDEX IF NOT EXISTS idx_chunks_checksum ON document_chunks(checksum);

-- HNSW Vector Index for sub-millisecond approximate nearest neighbor (ANN) search
CREATE INDEX IF NOT EXISTS idx_chunks_hnsw_cosine 
ON document_chunks 
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

-- Full-Text Search index for hybrid keyword queries
CREATE INDEX IF NOT EXISTS idx_chunks_fts 
ON document_chunks 
USING gin (to_tsvector('english', text));

-- 4. Ingestion Jobs Audit Table
CREATE TABLE IF NOT EXISTS ingestion_jobs (
    id VARCHAR(64) PRIMARY KEY,
    document_id VARCHAR(64) NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    current_stage VARCHAR(32) DEFAULT 'QUEUED',
    progress_pct INTEGER DEFAULT 0,
    retry_count INTEGER DEFAULT 0,
    error_log TEXT,
    started_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    completed_at TIMESTAMP WITH TIME ZONE
);

CREATE INDEX IF NOT EXISTS idx_jobs_status ON ingestion_jobs(current_stage);

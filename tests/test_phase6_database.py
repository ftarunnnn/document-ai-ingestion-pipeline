"""
Phase 6 Tests: Database Models, Persistence, pgvector Compatibility, and Chunk Indexing.
"""

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.database.models import Base, Document, DocumentChunk
from app.services.indexing import IndexingService


def test_database_creation_and_indexing(tmp_path):
    # Test with isolated SQLite DB
    db_file = tmp_path / "test_pipeline.db"
    test_engine = create_engine(f"sqlite:///{db_file}")
    Base.metadata.create_all(bind=test_engine)
    TestingSession = sessionmaker(bind=test_engine)
    db = TestingSession()

    try:
        service = IndexingService()

        # 1. Index document
        doc_meta = {
            "document_id": "test-doc-db-1",
            "filename": "financial_report.pdf",
            "file_size": 104200,
            "extension": ".pdf",
            "sha256_hash": "a1b2c3d4e5f67890123456789abcdef0123456789abcdef0123456789abcdef0",
            "stored_path": "/fake/path/doc.pdf",
            "status": "UPLOADED",
            "extracted_pages": 5,
        }
        doc = service.index_document(doc_meta, db)
        assert doc.id == "test-doc-db-1"
        assert doc.filename == "financial_report.pdf"

        # 2. Check duplicate hash query
        dup = service.check_duplicate(doc_meta["sha256_hash"], db)
        assert dup is not None
        assert dup.id == "test-doc-db-1"

        # 3. Index embedded chunks with vectors
        fake_vector = [0.05] * 384
        embedded_chunks = [
            {
                "chunk_id": "test-doc-db-1_c0",
                "chunk_index": 0,
                "page_number": 1,
                "section_title": "Executive Summary",
                "text": "Quarterly financial figures surpassed initial guidance.",
                "char_count": 55,
                "token_estimate": 13,
                "checksum": "checksum1",
                "is_table": False,
                "embedding": fake_vector,
                "embedding_model": "all-MiniLM-L6-v2",
            },
            {
                "chunk_id": "test-doc-db-1_c1",
                "chunk_index": 1,
                "page_number": 2,
                "section_title": "Revenue Breakdown",
                "text": "| Region | Revenue |\n| North America | $50M |",
                "char_count": 48,
                "token_estimate": 12,
                "checksum": "checksum2",
                "is_table": True,
                "embedding": fake_vector,
                "embedding_model": "all-MiniLM-L6-v2",
            },
        ]

        count = service.index_chunks("test-doc-db-1", embedded_chunks, db)
        assert count == 2

        # 4. Verify chunks retrieved
        chunks = service.get_document_chunks("test-doc-db-1", db)
        assert len(chunks) == 2
        assert chunks[0].section_title == "Executive Summary"
        assert len(chunks[0].embedding) == 384
        assert chunks[1].is_table is True

        # 5. Verify document status updated to INDEXED
        updated_doc = db.query(Document).filter(Document.id == "test-doc-db-1").first()
        assert updated_doc.status == "INDEXED"

        # 6. Verify Cascade deletion
        db.delete(updated_doc)
        db.commit()
        remaining_chunks = db.query(DocumentChunk).filter(DocumentChunk.document_id == "test-doc-db-1").all()
        assert len(remaining_chunks) == 0

    finally:
        db.close()

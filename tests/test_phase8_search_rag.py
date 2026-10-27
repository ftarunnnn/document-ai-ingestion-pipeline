"""
Phase 8 Tests: Semantic Search, Keyword Search, Hybrid RRF, and Grounded RAG Citations.
"""

from app.database.connection import SessionLocal, init_db
from app.database.models import Document, DocumentChunk
from app.ingestion.embeddings import embedding_generator
from app.services.retrieval import RetrievalService


def test_retrieval_and_rag_pipeline(tmp_path):
    init_db()
    db = SessionLocal()
    try:
        # Populate test document and chunk
        doc_id = "test-rag-doc-1"
        db.query(DocumentChunk).filter(DocumentChunk.document_id == doc_id).delete()
        db.query(Document).filter(Document.id == doc_id).delete()

        doc = Document(
            id=doc_id,
            filename="annual_contract.pdf",
            file_type="pdf",
            file_size_bytes=45000,
            sha256_hash="hash_rag_test_12345",
            storage_path="/tmp/annual_contract.pdf",
            status="INDEXED",
            total_pages=1,
        )
        db.add(doc)

        c1_text = "The vendor SLA guarantees 99.95% system uptime and sub-second latency for all API requests."
        c1_vec = embedding_generator.generate_embedding(c1_text)

        chunk1 = DocumentChunk(
            id=f"{doc_id}_c0",
            document_id=doc_id,
            chunk_index=0,
            page_number=1,
            section_title="Service Level Agreement",
            text=c1_text,
            char_count=len(c1_text),
            token_estimate=25,
            checksum="chk1",
            is_table=False,
            embedding=c1_vec,
            embedding_model=embedding_generator.model_name,
        )
        db.add(chunk1)
        db.commit()

        service = RetrievalService()

        # 1. Semantic search test
        sem_res = service.semantic_search("What is the system uptime SLA?", top_k=1, document_id=doc_id)
        assert len(sem_res) == 1
        assert "99.95%" in sem_res[0].text
        assert sem_res[0].page_number == 1
        assert "annual_contract.pdf" in sem_res[0].citation

        # 2. Keyword search test
        key_res = service.keyword_search("latency requests", top_k=1, document_id=doc_id)
        assert len(key_res) == 1
        assert "sub-second latency" in key_res[0].text

        # 3. Hybrid RRF search test
        hyb_res = service.hybrid_search("uptime and latency", top_k=1, document_id=doc_id)
        assert len(hyb_res) == 1
        assert hyb_res[0].retrieval_mode == "hybrid_rrf"

        # 4. RAG Question Answering test
        rag_res = service.generate_rag_answer("What SLA does the vendor guarantee?", top_k=1, document_id=doc_id)
        assert "annual_contract.pdf" in rag_res["answer"] or "99.95%" in rag_res["answer"]
        assert len(rag_res["citations"]) == 1
        assert rag_res["citations"][0]["page_number"] == 1
        assert rag_res["confidence"] > 0.0

    finally:
        db.close()

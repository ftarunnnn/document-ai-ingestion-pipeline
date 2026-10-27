"""
Phase 4 Tests: Intelligent Section-Aware Chunking, Table Preservation, and Metadata.
"""

from app.ingestion.chunker import IntelligentChunker


def test_chunking_with_heading_and_page_provenance():
    chunker = IntelligentChunker(chunk_size=200, chunk_overlap=30)
    sample_text = (
        "## Architecture Overview\n\n"
        "The Document AI Ingestion pipeline uses an event-driven ingestion workflow.\n\n"
        "It supports PDFs, Word files, Excel spreadsheets, and scanned images.\n\n"
        "## Security Specifications\n\n"
        "Every file is validated against strict magic byte signatures."
    )

    chunks = chunker.chunk_page(sample_text, document_id="doc-test-1", page_number=2)
    assert len(chunks) >= 2

    # Check provenance
    assert chunks[0].document_id == "doc-test-1"
    assert chunks[0].page_number == 2
    assert chunks[0].section_title is not None
    assert len(chunks[0].checksum) == 64
    assert chunks[0].token_estimate > 0


def test_table_preservation_as_atomic_chunk():
    chunker = IntelligentChunker(chunk_size=150, chunk_overlap=20)
    text_with_table = (
        "Introduction paragraph describing the quarterly revenue figures.\n\n"
        "| Quarter | Revenue | Profit |\n"
        "| --- | --- | --- |\n"
        "| Q1 | $1.2M | $300K |\n"
        "| Q2 | $1.5M | $400K |\n"
        "| Q3 | $1.8M | $550K |\n\n"
        "Concluding remarks about future outlook."
    )

    chunks = chunker.chunk_page(text_with_table, document_id="doc-tbl-1", page_number=1)

    table_chunks = [c for c in chunks if c.is_table]
    assert len(table_chunks) == 1
    assert "| Quarter | Revenue | Profit |" in table_chunks[0].text
    assert "| Q3 | $1.8M | $550K |" in table_chunks[0].text


def test_sliding_window_overlap():
    chunker = IntelligentChunker(chunk_size=100, chunk_overlap=25)
    long_text = "This is a sentence that goes on to test whether sliding window continuity preserves overlap. " * 3

    chunks = chunker.chunk_page(long_text, document_id="doc-overlap", page_number=1)
    assert len(chunks) > 1

    # Overlap should ensure the end of chunk 0 appears near the start of chunk 1
    end_of_c0 = chunks[0].text[-20:]
    assert end_of_c0 in chunks[1].text or chunks[1].text[:30] in chunks[0].text

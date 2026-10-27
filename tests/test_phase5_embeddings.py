"""
Phase 5 Tests: Vector Embedding Generation, Batching, Caching, and Normalization.
"""

import math
import numpy as np
from app.ingestion.embeddings import EmbeddingGenerator


def test_embedding_dimension_and_normalization(tmp_path):
    generator = EmbeddingGenerator(dimension=384, cache_dir=tmp_path)
    text = "Document AI Ingestion Pipeline vector embedding generation test."

    vec = generator.generate_embedding(text)
    assert len(vec) == 384

    # Check L2 norm is 1.0 (unit vector)
    norm = np.linalg.norm(vec)
    assert math.isclose(norm, 1.0, abs_tol=1e-3)


def test_embedding_cache_reuse(tmp_path):
    generator = EmbeddingGenerator(dimension=384, cache_dir=tmp_path)
    text = "Deterministic content for hash-based caching test."

    vec1 = generator.generate_embedding(text)
    # Call again with identical text
    vec2 = generator.generate_embedding(text)

    assert vec1 == vec2
    # Verify vector was recorded in internal cache dict
    assert len(generator._cache) >= 1


def test_batch_embedding_generation(tmp_path):
    generator = EmbeddingGenerator(dimension=384, cache_dir=tmp_path)
    texts = [
        "First technical paragraph about OCR.",
        "Second paragraph about pgvector search.",
        "Third paragraph about document normalization.",
    ]

    vectors = generator.generate_batch(texts, batch_size=2)
    assert len(vectors) == 3
    for v in vectors:
        assert len(v) == 384
        norm = np.linalg.norm(v)
        assert math.isclose(norm, 1.0, abs_tol=1e-3)


def test_embed_document_chunks(tmp_path):
    generator = EmbeddingGenerator(dimension=384, cache_dir=tmp_path)
    chunks = [
        {"chunk_id": "c1", "text": "Extracting text from PDF.", "page_number": 1},
        {"chunk_id": "c2", "text": "Storing vectors in database.", "page_number": 1},
    ]

    embedded = generator.embed_document_chunks(chunks)
    assert len(embedded) == 2
    assert "embedding" in embedded[0]
    assert len(embedded[0]["embedding"]) == 384
    assert embedded[0]["embedding_model"] == generator.model_name

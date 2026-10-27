"""
Search & RAG Question Answering API Endpoints.
Provides semantic, keyword, hybrid retrieval, and source-grounded Q&A.
"""

from typing import Any, Dict, List, Optional
from fastapi import APIRouter, HTTPException, Query, status
from pydantic import BaseModel, Field

from app.services.retrieval import retrieval_service

router = APIRouter(tags=["Search & AI Retrieval"])


class SearchQuery(BaseModel):
    query: str = Field(..., description="Search query string")
    top_k: int = Field(5, ge=1, le=50, description="Maximum passages to return")
    document_id: Optional[str] = Field(None, description="Optional document filter")


class SearchResultItem(BaseModel):
    chunk_id: str
    document_id: str
    filename: str
    page_number: int
    section_title: Optional[str]
    text: str
    score: float
    retrieval_mode: str
    is_table: bool
    citation: str


class SearchResponse(BaseModel):
    query: str
    mode: str
    total_results: int
    results: List[SearchResultItem]


class RAGQuery(BaseModel):
    question: str = Field(..., description="User question to answer using ingested documents")
    top_k: int = Field(3, ge=1, le=10, description="Number of context passages to retrieve")
    document_id: Optional[str] = Field(None, description="Optional filter by document ID")


class RAGResponse(BaseModel):
    question: str
    answer: str
    confidence: float
    citations: List[Dict[str, Any]]
    retrieved_passages_count: int


@router.post("/search/semantic", response_model=SearchResponse)
def semantic_search_endpoint(payload: SearchQuery):
    """Performs semantic vector nearest-neighbor search using cosine similarity."""
    results = retrieval_service.semantic_search(
        query=payload.query,
        top_k=payload.top_k,
        document_id=payload.document_id,
    )
    return SearchResponse(
        query=payload.query,
        mode="semantic",
        total_results=len(results),
        results=[r.to_dict() for r in results],
    )


@router.post("/search/keyword", response_model=SearchResponse)
def keyword_search_endpoint(payload: SearchQuery):
    """Performs lexical BM25-style keyword search."""
    results = retrieval_service.keyword_search(
        query=payload.query,
        top_k=payload.top_k,
        document_id=payload.document_id,
    )
    return SearchResponse(
        query=payload.query,
        mode="keyword",
        total_results=len(results),
        results=[r.to_dict() for r in results],
    )


@router.post("/search/hybrid", response_model=SearchResponse)
def hybrid_search_endpoint(payload: SearchQuery):
    """
    Performs hybrid retrieval combining semantic vectors and keyword search
    using Reciprocal Rank Fusion (RRF).
    """
    results = retrieval_service.hybrid_search(
        query=payload.query,
        top_k=payload.top_k,
        document_id=payload.document_id,
    )
    return SearchResponse(
        query=payload.query,
        mode="hybrid_rrf",
        total_results=len(results),
        results=[r.to_dict() for r in results],
    )


@router.post("/rag/ask", response_model=RAGResponse)
def rag_ask_endpoint(payload: RAGQuery):
    """
    Synthesizes a factually grounded answer based on retrieved document passages,
    including inline footnote citations.
    """
    answer_payload = retrieval_service.generate_rag_answer(
        question=payload.question,
        top_k=payload.top_k,
        document_id=payload.document_id,
    )
    return RAGResponse(**answer_payload)

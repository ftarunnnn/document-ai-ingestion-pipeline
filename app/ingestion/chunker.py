"""
Intelligent Document Chunking Engine.
Splits cleaned documents into traceable semantic units with rich provenance metadata.
Supports:
- Section- and paragraph-aware boundary detection
- Markdown table preservation (atomic table protection)
- Configurable sliding window with token/char overlap
- Deterministic chunk indexing and SHA-256 checksums for embedding cache reuse
"""

import hashlib
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from app.config import settings

# Regex patterns for structural boundary detection
HEADING_PATTERN = re.compile(r"^(#{1,6}\s+.*|[A-Z0-9\s]{4,40}:|Section\s+\d+.*)$", re.MULTILINE)
TABLE_BLOCK_PATTERN = re.compile(r"(?:^\|.+\|\r?\n)+", re.MULTILINE)


@dataclass
class DocumentChunk:
    """Represents a discrete searchable passage with source attribution."""
    chunk_id: str
    document_id: str
    chunk_index: int
    text: str
    page_number: int
    section_title: Optional[str]
    char_count: int
    token_estimate: int
    checksum: str
    is_table: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "chunk_index": self.chunk_index,
            "text": self.text,
            "page_number": self.page_number,
            "section_title": self.section_title,
            "char_count": self.char_count,
            "token_estimate": self.token_estimate,
            "checksum": self.checksum,
            "is_table": self.is_table,
            "metadata": self.metadata,
        }


class IntelligentChunker:
    """Performs semantic section-aware and sliding-window chunking."""

    def __init__(
        self,
        chunk_size: int = settings.CHUNK_SIZE,
        chunk_overlap: int = settings.CHUNK_OVERLAP,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    @staticmethod
    def compute_checksum(text: str) -> str:
        """Computes SHA-256 of chunk text for embedding cache matching."""
        return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()

    @staticmethod
    def estimate_tokens(text: str) -> int:
        """Approximates token count (avg ~4 chars per token in English)."""
        return max(1, len(text) // 4)

    def _split_into_paragraphs_and_tables(self, text: str) -> List[Tuple[str, bool, Optional[str]]]:
        """
        Splits page text into logical elements (paragraphs, tables, headings).
        Returns tuples of (content, is_table, heading_context).
        """
        elements: List[Tuple[str, bool, Optional[str]]] = []
        current_heading: Optional[str] = None

        # Split on double line breaks while keeping table blocks intact
        blocks = re.split(r"\n{2,}", text)
        for block in blocks:
            trimmed = block.strip()
            if not trimmed:
                continue

            # Check if this block is a heading
            if HEADING_PATTERN.match(trimmed.split("\n")[0]):
                current_heading = trimmed.split("\n")[0].lstrip("#").strip()

            # Check if block is a markdown table
            is_table = trimmed.startswith("|") and ("---" in trimmed)
            elements.append((trimmed, is_table, current_heading))

        return elements

    def _subdivide_long_text(self, text: str) -> List[str]:
        """Subdivides a long block exceeding chunk_size into overlapping slices."""
        if len(text) <= self.chunk_size:
            return [text]

        # Try splitting by sentences first
        sentences = re.split(r"(?<=[.!?])\s+", text)
        sub_chunks = []
        buf = []
        buf_len = 0

        for sent in sentences:
            if buf_len + len(sent) > self.chunk_size and buf:
                slice_text = " ".join(buf).strip()
                sub_chunks.append(slice_text)
                # Overlap: keep last sentence or characters
                if self.chunk_overlap > 0 and len(slice_text) > self.chunk_overlap:
                    buf = [buf[-1]] if len(buf[-1]) <= self.chunk_overlap else [slice_text[-self.chunk_overlap:]]
                    buf_len = sum(len(s) for s in buf)
                else:
                    buf = []
                    buf_len = 0
            buf.append(sent)
            buf_len += len(sent)

        if buf:
            sub_chunks.append(" ".join(buf).strip())

        # If any sub-chunk is still oversized (e.g. no punctuation), slice by character window
        final_slices = []
        for sc in sub_chunks:
            if len(sc) > self.chunk_size:
                step = max(1, self.chunk_size - self.chunk_overlap)
                for i in range(0, len(sc), step):
                    final_slices.append(sc[i : i + self.chunk_size])
            else:
                final_slices.append(sc)

        return final_slices

    def chunk_page(
        self,
        page_text: str,
        document_id: str,
        page_number: int,
        start_index: int = 0,
    ) -> List[DocumentChunk]:
        """
        Chunks text from a single page preserving section hierarchy and table structure.
        """
        if not page_text.strip():
            return []

        elements = self._split_into_paragraphs_and_tables(page_text)
        chunks: List[DocumentChunk] = []
        current_buffer: List[str] = []
        current_len = 0
        current_heading: Optional[str] = None
        chunk_idx = start_index

        def flush_buffer(heading_ctx: Optional[str]):
            nonlocal current_buffer, current_len, chunk_idx
            if not current_buffer:
                return

            text_block = "\n\n".join(current_buffer).strip()
            if text_block:
                chunk_id = f"{document_id}_p{page_number}_c{chunk_idx}"
                checksum = self.compute_checksum(text_block)
                chunks.append(
                    DocumentChunk(
                        chunk_id=chunk_id,
                        document_id=document_id,
                        chunk_index=chunk_idx,
                        text=text_block,
                        page_number=page_number,
                        section_title=heading_ctx,
                        char_count=len(text_block),
                        token_estimate=self.estimate_tokens(text_block),
                        checksum=checksum,
                        is_table=False,
                    )
                )
                chunk_idx += 1

            # Keep overlap for sliding continuity
            if self.chunk_overlap > 0 and len(text_block) > self.chunk_overlap:
                overlap_text = text_block[-self.chunk_overlap :]
                current_buffer = [overlap_text]
                current_len = len(overlap_text)
            else:
                current_buffer = []
                current_len = 0

        for content, is_table, heading in elements:
            if heading and not current_heading:
                current_heading = heading

            # If element is an atomic table
            if is_table:
                if current_buffer:
                    flush_buffer(current_heading)

                chunk_id = f"{document_id}_p{page_number}_c{chunk_idx}"
                chunks.append(
                    DocumentChunk(
                        chunk_id=chunk_id,
                        document_id=document_id,
                        chunk_index=chunk_idx,
                        text=content,
                        page_number=page_number,
                        section_title=current_heading or "Table",
                        char_count=len(content),
                        token_estimate=self.estimate_tokens(content),
                        checksum=self.compute_checksum(content),
                        is_table=True,
                    )
                )
                chunk_idx += 1
                continue

            # Subdivide content if larger than chunk_size
            sub_parts = self._subdivide_long_text(content)
            for part in sub_parts:
                if current_len + len(part) > self.chunk_size and current_buffer:
                    flush_buffer(current_heading)
                current_buffer.append(part)
                current_len += len(part)

            if heading:
                current_heading = heading

        # Flush final remaining buffer
        if current_buffer:
            flush_buffer(current_heading)

        return chunks

    def chunk_document(
        self,
        cleaned_doc_dict: Dict[str, Any],
    ) -> List[DocumentChunk]:
        """
        Iterates over all cleaned pages of a document and generates indexed chunks.
        """
        all_chunks: List[DocumentChunk] = []
        doc_id = cleaned_doc_dict.get("document_id", "doc")
        pages = cleaned_doc_dict.get("pages", [])

        chunk_counter = 0
        for p in pages:
            p_num = p.get("page_number", 1)
            p_text = p.get("cleaned_text", "")
            if not p_text.strip():
                continue

            page_chunks = self.chunk_page(
                page_text=p_text,
                document_id=doc_id,
                page_number=p_num,
                start_index=chunk_counter,
            )
            all_chunks.extend(page_chunks)
            chunk_counter += len(page_chunks)

        return all_chunks


document_chunker = IntelligentChunker()

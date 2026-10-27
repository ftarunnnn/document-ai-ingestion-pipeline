"""
Data Cleaning & Text Normalization Engine.
Cleans raw extracted text, strips recurring headers/footers, normalizes Unicode,
preserves Markdown tables, handles empty pages, and computes document quality metrics.
"""

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

# Common regex patterns
CONTROL_CHARS_PATTERN = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")
PAGE_NUMBER_PATTERN = re.compile(
    r"^\s*(?:page\s*\d+(?:\s*(?:of|\/)\s*\d+)?|\d+\s*(?:of|\/)\s*\d+|\-\s*\d+\s*\-)\s*$",
    re.IGNORECASE,
)
HYPHENATED_LINE_BREAK = re.compile(r"(\w+)-\n(\w+)")
MULTIPLE_NEWLINES = re.compile(r"\n{3,}")
EXCESSIVE_SPACES = re.compile(r"[ \t]{2,}")
TABLE_LINE_PATTERN = re.compile(r"^\s*\|.*\|\s*$")


@dataclass
class QualityMetrics:
    """Document extraction quality metrics."""
    total_characters: int
    total_words: int
    line_count: int
    alphanumeric_ratio: float
    noise_score: float
    is_empty: bool
    quality_rating: str  # EXCELLENT, GOOD, FAIR, POOR, EMPTY

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_characters": self.total_characters,
            "total_words": self.total_words,
            "line_count": self.line_count,
            "alphanumeric_ratio": round(self.alphanumeric_ratio, 4),
            "noise_score": round(self.noise_score, 4),
            "is_empty": self.is_empty,
            "quality_rating": self.quality_rating,
        }


@dataclass
class CleanedPage:
    """Cleaned representation of a document page."""
    page_number: int
    cleaned_text: str
    original_text: str
    tables: List[List[List[str]]] = field(default_factory=list)
    quality: QualityMetrics = field(default_factory=lambda: QualityMetrics(0, 0, 0, 0.0, 0.0, True, "EMPTY"))
    warnings: List[str] = field(default_factory=list)


@dataclass
class CleanedDocument:
    """Complete cleaned document container."""
    document_id: str
    filename: str
    total_pages: int
    pages: List[CleanedPage]
    overall_quality: QualityMetrics
    removed_headers_footers: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "document_id": self.document_id,
            "filename": self.filename,
            "total_pages": self.total_pages,
            "overall_quality": self.overall_quality.to_dict(),
            "removed_headers_footers": self.removed_headers_footers,
            "pages": [
                {
                    "page_number": p.page_number,
                    "cleaned_text": p.cleaned_text,
                    "tables": p.tables,
                    "quality": p.quality.to_dict(),
                    "warnings": p.warnings,
                }
                for p in self.pages
            ],
        }


class TextCleaner:
    """Text cleaning, noise reduction, and formatting normalization."""

    def normalize_unicode(self, text: str) -> str:
        """Applies NFKC normalization and removes non-printable control characters."""
        normalized = unicodedata.normalize("NFKC", text)
        return CONTROL_CHARS_PATTERN.sub("", normalized)

    def fix_hyphenated_words(self, text: str) -> str:
        """Fixes words split across lines by a hyphen (e.g. 'docu-\nment' -> 'document')."""
        return HYPHENATED_LINE_BREAK.sub(r"\1\2", text)

    def normalize_whitespace(self, text: str) -> str:
        """Cleans excessive spaces while preserving table formatting and paragraph breaks."""
        lines = text.split("\n")
        cleaned_lines = []
        for line in lines:
            if TABLE_LINE_PATTERN.match(line):
                # Preserve table line intact
                cleaned_lines.append(line.strip())
            else:
                # Collapse excessive horizontal spaces
                cleaned_lines.append(EXCESSIVE_SPACES.sub(" ", line).strip())

        rejoined = "\n".join(cleaned_lines)
        # Collapse 3 or more blank lines down to 2
        return MULTIPLE_NEWLINES.sub("\n\n", rejoined).strip()

    def detect_boilerplate(self, pages_text: List[str]) -> Set[str]:
        """
        Detects recurring header/footer candidates present across multiple pages.
        """
        if len(pages_text) <= 1:
            return set()

        line_occurrences: Dict[str, int] = {}
        for p_text in pages_text:
            lines = [l.strip() for l in p_text.split("\n") if len(l.strip()) > 3]
            # Check top 3 lines (headers) and bottom 3 lines (footers)
            candidates = set(lines[:3] + lines[-3:])
            for cand in candidates:
                if not TABLE_LINE_PATTERN.match(cand):
                    line_occurrences[cand] = line_occurrences.get(cand, 0) + 1

        # If a line appears on > 60% of pages, mark as repeated boilerplate
        threshold = max(2, int(len(pages_text) * 0.6))
        boilerplate = {line for line, count in line_occurrences.items() if count >= threshold}
        return boilerplate

    def calculate_quality(self, text: str) -> QualityMetrics:
        """Computes objective quality scores on the cleaned text."""
        total_chars = len(text)
        if total_chars == 0:
            return QualityMetrics(0, 0, 0, 0.0, 1.0, True, "EMPTY")

        words = text.split()
        total_words = len(words)
        lines = [l for l in text.split("\n") if l.strip()]
        line_count = len(lines)

        alnum_chars = sum(1 for c in text if c.isalnum())
        alnum_ratio = alnum_chars / total_chars
        noise_score = 1.0 - alnum_ratio

        if alnum_ratio >= 0.75:
            rating = "EXCELLENT"
        elif alnum_ratio >= 0.55:
            rating = "GOOD"
        elif alnum_ratio >= 0.35:
            rating = "FAIR"
        else:
            rating = "POOR"

        return QualityMetrics(
            total_characters=total_chars,
            total_words=total_words,
            line_count=line_count,
            alphanumeric_ratio=alnum_ratio,
            noise_score=noise_score,
            is_empty=False,
            quality_rating=rating,
        )

    def clean_page(
        self,
        raw_text: str,
        page_number: int,
        boilerplate: Set[str],
        tables: Optional[List[Any]] = None,
    ) -> CleanedPage:
        """Cleans an individual page's text content."""
        warnings = []
        if not raw_text.strip():
            warnings.append(f"Page {page_number} is empty.")
            return CleanedPage(
                page_number=page_number,
                cleaned_text="",
                original_text=raw_text,
                tables=tables or [],
                quality=QualityMetrics(0, 0, 0, 0.0, 1.0, True, "EMPTY"),
                warnings=warnings,
            )

        # 1. Unicode Normalization
        text = self.normalize_unicode(raw_text)

        # 2. Fix hyphenation
        text = self.fix_hyphenated_words(text)

        # 3. Strip page numbers and boilerplate lines
        filtered_lines = []
        for line in text.split("\n"):
            stripped = line.strip()
            if PAGE_NUMBER_PATTERN.match(stripped):
                continue
            if stripped in boilerplate:
                continue
            filtered_lines.append(line)

        # 4. Whitespace Normalization
        cleaned_text = self.normalize_whitespace("\n".join(filtered_lines))

        # 5. Quality scoring
        quality = self.calculate_quality(cleaned_text)
        if quality.quality_rating in ["POOR", "FAIR"]:
            warnings.append(
                f"Page {page_number} has low alphanumeric ratio ({quality.alphanumeric_ratio:.1%}). Possible scan noise."
            )

        return CleanedPage(
            page_number=page_number,
            cleaned_text=cleaned_text,
            original_text=raw_text,
            tables=tables or [],
            quality=quality,
            warnings=warnings,
        )

    def clean_document_dict(self, parsed_doc_dict: Dict[str, Any]) -> CleanedDocument:
        """Cleans entire parsed document record from dictionary format."""
        raw_pages = parsed_doc_dict.get("pages", [])
        page_texts = [p.get("text", "") for p in raw_pages]

        # Detect headers/footers
        boilerplate = self.detect_boilerplate(page_texts)

        cleaned_pages: List[CleanedPage] = []
        for p in raw_pages:
            p_num = p.get("page_number", 1)
            p_raw = p.get("text", "")
            p_tables = p.get("tables", [])
            c_page = self.clean_page(p_raw, p_num, boilerplate, p_tables)
            cleaned_pages.append(c_page)

        # Combine text for overall document metric
        combined_text = "\n\n".join([p.cleaned_text for p in cleaned_pages if p.cleaned_text])
        overall_quality = self.calculate_quality(combined_text)

        return CleanedDocument(
            document_id=parsed_doc_dict.get("document_id", ""),
            filename=parsed_doc_dict.get("filename", ""),
            total_pages=len(cleaned_pages),
            pages=cleaned_pages,
            overall_quality=overall_quality,
            removed_headers_footers=sorted(list(boilerplate)),
        )


text_cleaner = TextCleaner()

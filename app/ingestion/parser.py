"""
Multi-Format Document Parsing Engine.
Extracts text, structured tables, page numbers, and layouts from:
- PDF files (PyMuPDF / fitz, table extraction, OCR fallback for scanned pages)
- Word documents (python-docx with native XML fallback)
- Excel spreadsheets (openpyxl/pandas with native XML fallback)
- Plain text files (.txt with multi-encoding auto-detection)
- Scanned documents and images (PNG, JPG, TIFF, etc. via OCR)
"""

import io
import os
import re
import xml.etree.ElementTree as ET
import zipfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.ingestion.ocr import ocr_extractor

# Try importing primary document parsing libraries
try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False

try:
    import docx
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False

try:
    import openpyxl
    OPENPYXL_AVAILABLE = True
except ImportError:
    OPENPYXL_AVAILABLE = False


@dataclass
class ParsedTable:
    """Represents a structured table extracted from a page."""
    rows: List[List[str]]
    title: Optional[str] = None

    def to_markdown(self) -> str:
        """Converts table rows into a clean Markdown table string."""
        if not self.rows:
            return ""
        header = self.rows[0]
        sanitized_header = [str(c).replace("\n", " ").strip() or " " for c in header]
        md_lines = ["| " + " | ".join(sanitized_header) + " |"]
        md_lines.append("| " + " | ".join(["---"] * len(sanitized_header)) + " |")

        for row in self.rows[1:]:
            padded = row + [""] * (len(sanitized_header) - len(row))
            sanitized_row = [str(c).replace("\n", " ").strip() or " " for c in padded[:len(sanitized_header)]]
            md_lines.append("| " + " | ".join(sanitized_row) + " |")

        return "\n".join(md_lines)


@dataclass
class ParsedPage:
    """Extracted text and structure for a single document page."""
    page_number: int
    text: str
    tables: List[ParsedTable] = field(default_factory=list)
    char_count: int = 0
    word_count: int = 0
    is_scanned: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.char_count:
            self.char_count = len(self.text)
        if not self.word_count:
            self.word_count = len(self.text.split())


@dataclass
class ParsedDocument:
    """Complete structured representation of an extracted document."""
    document_id: str
    filename: str
    file_type: str
    total_pages: int
    pages: List[ParsedPage] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    extraction_method: str = "direct"

    def full_text(self) -> str:
        """Concatenates all pages with page marker comments."""
        parts = []
        for p in self.pages:
            parts.append(f"<!-- Page {p.page_number} -->\n{p.text}")
            for t in p.tables:
                parts.append(f"\n{t.to_markdown()}\n")
        return "\n\n".join(parts)

    def to_dict(self) -> Dict[str, Any]:
        """Serializes document to JSON-compatible dictionary."""
        return {
            "document_id": self.document_id,
            "filename": self.filename,
            "file_type": self.file_type,
            "total_pages": self.total_pages,
            "errors": self.errors,
            "warnings": self.warnings,
            "extraction_method": self.extraction_method,
            "pages": [
                {
                    "page_number": p.page_number,
                    "text": p.text,
                    "tables": [t.rows for t in p.tables],
                    "char_count": p.char_count,
                    "word_count": p.word_count,
                    "is_scanned": p.is_scanned,
                    "metadata": p.metadata,
                }
                for p in self.pages
            ],
        }


class DocumentParser:
    """Unified entry point for multi-format document parsing."""

    def parse(self, file_path: str | Path, document_id: str = "") -> ParsedDocument:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Document file not found at: {path}")

        ext = path.suffix.lower()
        doc_id = document_id or path.stem

        if ext == ".pdf":
            return self._parse_pdf(path, doc_id)
        elif ext in [".docx", ".doc"]:
            return self._parse_docx(path, doc_id)
        elif ext in [".xlsx", ".xls"]:
            return self._parse_excel(path, doc_id)
        elif ext == ".txt":
            return self._parse_text(path, doc_id)
        elif ext in [".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".webp"]:
            return self._parse_image(path, doc_id)
        else:
            raise ValueError(f"No parser available for extension '{ext}'")

    # 1. PDF Parser (PyMuPDF + OCR fallback)
    def _parse_pdf(self, path: Path, doc_id: str) -> ParsedDocument:
        errors = []
        warnings = []
        pages = []
        method = "pymupdf"

        if PYMUPDF_AVAILABLE:
            try:
                doc = fitz.open(str(path))
                total_pages = len(doc)

                for page_idx in range(total_pages):
                    page = doc[page_idx]
                    page_num = page_idx + 1

                    # Extract native text
                    raw_text = page.get_text("text").strip()
                    tables = []
                    is_scanned = False

                    # Try extracting tables using PyMuPDF table finder
                    try:
                        tabs = page.find_tables()
                        if tabs and tabs.tables:
                            for tab in tabs.tables:
                                table_data = tab.extract()
                                if table_data:
                                    tables.append(ParsedTable(rows=table_data))
                    except Exception:
                        pass

                    # If page has minimal text, it's likely a scan: fallback to OCR
                    if len(raw_text) < 30:
                        try:
                            pix = page.get_pixmap(dpi=150)
                            img_bytes = pix.tobytes("png")
                            ocr_res = ocr_extractor.extract_from_image(img_bytes)
                            if ocr_res["text"].strip():
                                raw_text = ocr_res["text"].strip()
                                is_scanned = True
                                warnings.append(
                                    f"Page {page_num} was scanned/raster: extracted text via {ocr_res['engine']} OCR."
                                )
                        except Exception as ocr_err:
                            errors.append(f"OCR fallback failed on page {page_num}: {str(ocr_err)}")

                    pages.append(
                        ParsedPage(
                            page_number=page_num,
                            text=raw_text,
                            tables=tables,
                            is_scanned=is_scanned,
                        )
                    )
                doc.close()

                return ParsedDocument(
                    document_id=doc_id,
                    filename=path.name,
                    file_type="pdf",
                    total_pages=total_pages,
                    pages=pages,
                    errors=errors,
                    warnings=warnings,
                    extraction_method=method,
                )
            except Exception as e:
                errors.append(f"PyMuPDF error: {str(e)}")

        # Fallback if PyMuPDF failed
        warnings.append("Using raw PDF text extraction fallback.")
        content = path.read_bytes()
        text_matches = re.findall(rb"\((.*?)\)Tj", content)
        extracted = " ".join([m.decode("latin-1", errors="ignore") for m in text_matches])
        pages.append(ParsedPage(page_number=1, text=extracted or "[Unextractable PDF content]"))
        return ParsedDocument(
            document_id=doc_id,
            filename=path.name,
            file_type="pdf",
            total_pages=1,
            pages=pages,
            errors=errors,
            warnings=warnings,
            extraction_method="fallback_raw",
        )

    # 2. DOCX Parser (python-docx + Native XML fallback)
    def _parse_docx(self, path: Path, doc_id: str) -> ParsedDocument:
        pages = []
        errors = []
        warnings = []
        tables = []
        full_text_blocks = []

        if DOCX_AVAILABLE:
            try:
                doc = docx.Document(str(path))
                # Extract paragraphs and headings
                for para in doc.paragraphs:
                    text = para.text.strip()
                    if text:
                        if para.style and para.style.name.startswith("Heading"):
                            level = para.style.name.replace("Heading", "").strip() or "1"
                            full_text_blocks.append(f"{'#' * int(level if level.isdigit() else 1)} {text}")
                        else:
                            full_text_blocks.append(text)

                # Extract tables
                for t in doc.tables:
                    rows = []
                    for row in t.rows:
                        row_vals = [cell.text.strip() for cell in row.cells]
                        rows.append(row_vals)
                    if rows:
                        tables.append(ParsedTable(rows=rows))

                text_combined = "\n\n".join(full_text_blocks)
                pages.append(ParsedPage(page_number=1, text=text_combined, tables=tables))
                return ParsedDocument(
                    document_id=doc_id,
                    filename=path.name,
                    file_type="docx",
                    total_pages=1,
                    pages=pages,
                    extraction_method="python-docx",
                )
            except Exception as e:
                warnings.append(f"python-docx parsing error ({e}), switching to native XML parser.")

        # Native OpenXML fallback (zero-dependency zipfile/ElementTree parser)
        try:
            with zipfile.ZipFile(str(path), "r") as z:
                xml_content = z.read("word/document.xml")
                root = ET.fromstring(xml_content)
                ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}

                # Paragraphs
                for p in root.findall(".//w:p", ns):
                    texts = [node.text for node in p.findall(".//w:t", ns) if node.text]
                    if texts:
                        full_text_blocks.append("".join(texts).strip())

                # Tables
                for tbl in root.findall(".//w:tbl", ns):
                    t_rows = []
                    for tr in tbl.findall(".//w:tr", ns):
                        row_cells = []
                        for tc in tr.findall(".//w:tc", ns):
                            cell_text = "".join(node.text for node in tc.findall(".//w:t", ns) if node.text)
                            row_cells.append(cell_text.strip())
                        if row_cells:
                            t_rows.append(row_cells)
                    if t_rows:
                        tables.append(ParsedTable(rows=t_rows))

                pages.append(
                    ParsedPage(page_number=1, text="\n\n".join(full_text_blocks), tables=tables)
                )
                return ParsedDocument(
                    document_id=doc_id,
                    filename=path.name,
                    file_type="docx",
                    total_pages=1,
                    pages=pages,
                    warnings=warnings,
                    extraction_method="native_openxml",
                )
        except Exception as e:
            errors.append(f"DOCX native parser failure: {str(e)}")
            pages.append(ParsedPage(page_number=1, text="[Failed to parse Word document]"))
            return ParsedDocument(
                document_id=doc_id,
                filename=path.name,
                file_type="docx",
                total_pages=1,
                pages=pages,
                errors=errors,
                extraction_method="failed",
            )

    # 3. Excel Parser (openpyxl + Native XML Fallback)
    def _parse_excel(self, path: Path, doc_id: str) -> ParsedDocument:
        pages = []
        errors = []
        warnings = []

        if OPENPYXL_AVAILABLE:
            try:
                wb = openpyxl.load_workbook(str(path), data_only=True)
                for idx, sheetname in enumerate(wb.sheetnames):
                    ws = wb[sheetname]
                    rows_data = []
                    for row in ws.iter_rows(values_only=True):
                        # Filter out completely empty rows
                        if any(v is not None for v in row):
                            row_str = [str(c) if c is not None else "" for c in row]
                            rows_data.append(row_str)

                    if rows_data:
                        tbl = ParsedTable(rows=rows_data, title=f"Sheet: {sheetname}")
                        text_summary = f"### Sheet: {sheetname}\n\n{tbl.to_markdown()}"
                        pages.append(
                            ParsedPage(
                                page_number=idx + 1,
                                text=text_summary,
                                tables=[tbl],
                                metadata={"sheet_name": sheetname, "row_count": len(rows_data)},
                            )
                        )
                wb.close()
                return ParsedDocument(
                    document_id=doc_id,
                    filename=path.name,
                    file_type="xlsx",
                    total_pages=max(len(pages), 1),
                    pages=pages or [ParsedPage(page_number=1, text="[Empty Excel Sheet]")],
                    extraction_method="openpyxl",
                )
            except Exception as e:
                warnings.append(f"openpyxl failed ({e}), using native spreadsheet parser.")

        # Native OpenXML Spreadsheet fallback (reads sharedStrings and sheet xml)
        try:
            with zipfile.ZipFile(str(path), "r") as z:
                # Load shared strings
                shared_strings = []
                if "xl/sharedStrings.xml" in z.namelist():
                    s_tree = ET.fromstring(z.read("xl/sharedStrings.xml"))
                    for si in s_tree.findall(".//{http://schemas.openxmlformats.org/spreadsheetml/2006/main}t"):
                        shared_strings.append(si.text or "")

                sheet_files = sorted([f for f in z.namelist() if f.startswith("xl/worksheets/sheet")])
                for idx, sheet_file in enumerate(sheet_files):
                    sheet_tree = ET.fromstring(z.read(sheet_file))
                    ns = {"s": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
                    sheet_rows = []
                    for row in sheet_tree.findall(".//s:row", ns):
                        curr_row = []
                        for c in row.findall(".//s:c", ns):
                            v_elem = c.find("s:v", ns)
                            val = v_elem.text if v_elem is not None else ""
                            if c.get("t") == "s" and val.isdigit() and int(val) < len(shared_strings):
                                val = shared_strings[int(val)]
                            curr_row.append(val)
                        if any(curr_row):
                            sheet_rows.append(curr_row)

                    if sheet_rows:
                        tbl = ParsedTable(rows=sheet_rows)
                        pages.append(
                            ParsedPage(
                                page_number=idx + 1,
                                text=f"### Sheet {idx + 1}\n\n{tbl.to_markdown()}",
                                tables=[tbl],
                            )
                        )

                return ParsedDocument(
                    document_id=doc_id,
                    filename=path.name,
                    file_type="xlsx",
                    total_pages=max(len(pages), 1),
                    pages=pages or [ParsedPage(page_number=1, text="[Empty Sheet]")],
                    warnings=warnings,
                    extraction_method="native_openxml_sheet",
                )
        except Exception as e:
            errors.append(f"Excel native parser failure: {str(e)}")
            pages.append(ParsedPage(page_number=1, text="[Failed to parse Excel file]"))
            return ParsedDocument(
                document_id=doc_id,
                filename=path.name,
                file_type="xlsx",
                total_pages=1,
                pages=pages,
                errors=errors,
                extraction_method="failed",
            )

    # 4. Text Parser (.txt)
    def _parse_text(self, path: Path, doc_id: str) -> ParsedDocument:
        content_bytes = path.read_bytes()
        text = ""
        # Auto-detect encoding
        for enc in ["utf-8", "utf-16", "cp1252", "latin-1"]:
            try:
                text = content_bytes.decode(enc)
                break
            except UnicodeDecodeError:
                continue

        # Split into pages by form feed (\x0c) or group every ~50 lines
        raw_pages = text.split("\x0c") if "\x0c" in text else [text]
        pages = []
        for i, page_str in enumerate(raw_pages, start=1):
            clean_str = page_str.strip()
            pages.append(ParsedPage(page_number=i, text=clean_str))

        return ParsedDocument(
            document_id=doc_id,
            filename=path.name,
            file_type="txt",
            total_pages=len(pages),
            pages=pages,
            extraction_method="text_decoder",
        )

    # 5. Image Parser (Images via OCR)
    def _parse_image(self, path: Path, doc_id: str) -> ParsedDocument:
        ocr_result = ocr_extractor.extract_from_image(str(path))
        page = ParsedPage(
            page_number=1,
            text=ocr_result["text"],
            is_scanned=True,
            metadata={
                "ocr_confidence": ocr_result["confidence"],
                "ocr_engine": ocr_result["engine"],
                "image_width": ocr_result["width"],
                "image_height": ocr_result["height"],
            },
        )
        return ParsedDocument(
            document_id=doc_id,
            filename=path.name,
            file_type=path.suffix.lstrip("."),
            total_pages=1,
            pages=[page],
            extraction_method=f"ocr_{ocr_result['engine']}",
        )


# Singleton document parser
document_parser = DocumentParser()

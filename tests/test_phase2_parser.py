"""
Phase 2 Tests: Document Parsing (PDF, Word, Excel, Text) and OCR Fallback.
"""

from pathlib import Path
import fitz  # PyMuPDF
from PIL import Image, ImageDraw
import pytest

from app.ingestion.ocr import ocr_extractor
from app.ingestion.parser import DocumentParser, ParsedTable


def test_table_to_markdown():
    table = ParsedTable(
        rows=[
            ["Product", "Q1 Sales", "Q2 Sales"],
            ["Enterprise DocAI", "$120k", "$180k"],
            ["Lite Parser", "$45k", "$60k"],
        ]
    )
    md = table.to_markdown()
    assert "| Product | Q1 Sales | Q2 Sales |" in md
    assert "| Enterprise DocAI | $120k | $180k |" in md


def test_parse_text_file(tmp_path):
    text_file = tmp_path / "sample.txt"
    text_file.write_text("Header Section\n\nThis is paragraph one.\n\nThis is paragraph two.", encoding="utf-8")

    parser = DocumentParser()
    doc = parser.parse(text_file, document_id="doc-txt-1")

    assert doc.file_type == "txt"
    assert doc.total_pages == 1
    assert "paragraph one" in doc.pages[0].text
    assert doc.pages[0].word_count > 0


def test_parse_pdf_with_pymupdf(tmp_path):
    # Dynamically generate a valid PDF using PyMuPDF
    pdf_path = tmp_path / "test_report.pdf"
    doc_writer = fitz.open()

    # Page 1
    page1 = doc_writer.new_page()
    page1.insert_text((50, 50), "Document AI Ingestion Pipeline - Confidential", fontsize=14)
    page1.insert_text((50, 100), "Executive Summary: Multi-format parsing in progress.", fontsize=11)

    # Page 2
    page2 = doc_writer.new_page()
    page2.insert_text((50, 50), "Page 2 Financial Overview and Metrics", fontsize=12)

    doc_writer.save(str(pdf_path))
    doc_writer.close()

    parser = DocumentParser()
    parsed_pdf = parser.parse(pdf_path, document_id="pdf-doc-1")

    assert parsed_pdf.total_pages == 2
    assert "Document AI Ingestion Pipeline" in parsed_pdf.pages[0].text
    assert "Financial Overview" in parsed_pdf.pages[1].text
    assert parsed_pdf.extraction_method == "pymupdf"


def test_ocr_image_extraction(tmp_path):
    # Create an image with visual text or verify OCR handler doesn't crash
    img_path = tmp_path / "scanned_doc.png"
    img = Image.new("RGB", (300, 100), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.text((20, 40), "HELLO AI INGESTION", fill=(0, 0, 0))
    img.save(str(img_path))

    parser = DocumentParser()
    res = parser.parse(img_path, document_id="img-1")

    assert res.total_pages == 1
    assert res.pages[0].is_scanned is True
    assert "ocr" in res.extraction_method.lower()

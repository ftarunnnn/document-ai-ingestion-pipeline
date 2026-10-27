"""
Phase 3 Tests: Data Cleaning, Boilerplate Removal, Normalization, and Quality Metrics.
"""

from app.ingestion.cleaner import TextCleaner


def test_unicode_normalization_and_control_chars():
    cleaner = TextCleaner()
    # NFKC should normalize fullwidth characters, and control characters should be stripped
    messy_text = "DocAI\x00 Ingestion\x08 Pipeline \uff11\uff12\uff13"
    cleaned = cleaner.normalize_unicode(messy_text)
    assert "\x00" not in cleaned
    assert "\x08" not in cleaned
    assert "123" in cleaned


def test_hyphenated_line_break_fix():
    cleaner = TextCleaner()
    text = "The docu-\nment ingestion pipe-\nline processes contracts."
    fixed = cleaner.fix_hyphenated_words(text)
    assert "document" in fixed
    assert "pipeline" in fixed


def test_whitespace_normalization_preserving_tables():
    cleaner = TextCleaner()
    text = (
        "Heading 1   \n\n\n\n"
        "Paragraph with    multiple     spaces.\n\n"
        "| Column A | Column B |\n"
        "| --- | --- |\n"
        "| Cell 1 | Cell 2 |\n\n\n"
        "Next paragraph."
    )
    normalized = cleaner.normalize_whitespace(text)
    assert "\n\n\n" not in normalized
    assert "Paragraph with multiple spaces." in normalized
    assert "| Column A | Column B |" in normalized


def test_boilerplate_detection_and_removal():
    cleaner = TextCleaner()
    page1 = "COMPANY CONFIDENTIAL - INTERNAL USE ONLY\n\nFirst page content.\n\nPage 1 of 3"
    page2 = "COMPANY CONFIDENTIAL - INTERNAL USE ONLY\n\nSecond page content.\n\nPage 2 of 3"
    page3 = "COMPANY CONFIDENTIAL - INTERNAL USE ONLY\n\nThird page content.\n\nPage 3 of 3"

    boilerplate = cleaner.detect_boilerplate([page1, page2, page3])
    assert "COMPANY CONFIDENTIAL - INTERNAL USE ONLY" in boilerplate

    cleaned_p1 = cleaner.clean_page(page1, 1, boilerplate)
    assert "COMPANY CONFIDENTIAL" not in cleaned_p1.cleaned_text
    assert "Page 1 of 3" not in cleaned_p1.cleaned_text
    assert "First page content." in cleaned_p1.cleaned_text


def test_quality_metrics_calculation():
    cleaner = TextCleaner()

    # High quality clean text
    clean_text = "This is a clean, structured document containing high quality technical text."
    q_clean = cleaner.calculate_quality(clean_text)
    assert q_clean.is_empty is False
    assert q_clean.quality_rating in ["EXCELLENT", "GOOD"]
    assert q_clean.noise_score < 0.3

    # Empty text
    q_empty = cleaner.calculate_quality("")
    assert q_empty.is_empty is True
    assert q_empty.quality_rating == "EMPTY"

    # Degraded noisy text
    noisy_text = "!!@#$ %^&* !@#$ %^&* ???"
    q_noisy = cleaner.calculate_quality(noisy_text)
    assert q_noisy.quality_rating == "POOR"
    assert q_noisy.noise_score > 0.8

"""
OCR Extraction Engine.
Extracts text from scanned PDF pages, raster images, and document screenshots.
Supports EasyOCR, Tesseract, and image preprocessing with PIL/OpenCV.
"""

import io
import logging
import os
import shutil
from typing import Dict, List, Optional, Tuple
from PIL import Image, ImageEnhance, ImageFilter

from app.config import settings

logger = logging.getLogger(__name__)


class OCRExtractor:
    """Multi-backend OCR processor with image cleanup and confidence estimation."""

    def __init__(self):
        self._easyocr_reader = None
        self._tesseract_available: Optional[bool] = None

    def _is_tesseract_installed(self) -> bool:
        if self._tesseract_available is None:
            cmd = settings.TESSERACT_CMD
            self._tesseract_available = shutil.which(cmd) is not None
        return self._tesseract_available

    def _get_easyocr_reader(self):
        if self._easyocr_reader is None:
            # Check if models already exist locally before initializing to prevent hanging downloads
            model_dir = os.path.expanduser("~/.EasyOCR/model")
            has_local_weights = os.path.exists(model_dir) and len(os.listdir(model_dir)) > 0
            if not has_local_weights:
                # Do not block runtime with synchronous huge model downloads unless explicitly primed
                self._easyocr_reader = False
                return None

            try:
                import easyocr
                self._easyocr_reader = easyocr.Reader(["en"], gpu=False, verbose=False, download_enabled=False)
            except Exception as e:
                logger.warning(f"EasyOCR reader initialization deferred: {e}")
                self._easyocr_reader = False
        return self._easyocr_reader if self._easyocr_reader is not False else None

    @staticmethod
    def preprocess_image(image: Image.Image) -> Image.Image:
        """
        Preprocesses image for optimal OCR extraction:
        1. Convert to Grayscale
        2. Enhance contrast
        3. Apply subtle sharpening
        """
        if image.mode != "L":
            image = image.convert("L")

        # Contrast enhancement
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(1.8)

        # Sharpness enhancement
        enhancer = ImageEnhance.Sharpness(image)
        image = enhancer.enhance(1.4)

        return image

    def extract_from_image(
        self,
        image_input: Image.Image | bytes | str,
        lang: str = "eng",
    ) -> Dict[str, any]:
        """
        Extracts text from an image object, image bytes, or file path.
        Returns extracted text, confidence score, and detection metadata.
        """
        if isinstance(image_input, bytes):
            img = Image.open(io.BytesIO(image_input))
        elif isinstance(image_input, str):
            img = Image.open(image_input)
        else:
            img = image_input

        processed_img = self.preprocess_image(img)
        extracted_text = ""
        avg_confidence = 0.0
        engine_used = "none"

        # 1. Try PyTesseract first if binary is present
        if self._is_tesseract_installed():
            try:
                import pytesseract
                data = pytesseract.image_to_data(processed_img, output_type=pytesseract.Output.DICT)
                confidences = [int(c) for c in data.get("conf", []) if str(c).lstrip("-").isdigit() and int(c) >= 0]
                extracted_text = pytesseract.image_to_string(processed_img, lang=lang).strip()
                avg_confidence = sum(confidences) / max(len(confidences), 1)
                engine_used = "tesseract"
            except Exception as e:
                logger.debug(f"Tesseract extraction fallback: {e}")

        # 2. Try EasyOCR if Tesseract gave no text or was unavailable
        if not extracted_text:
            reader = self._get_easyocr_reader()
            if reader:
                try:
                    # Convert PIL to byte array for EasyOCR
                    byte_arr = io.BytesIO()
                    processed_img.save(byte_arr, format="PNG")
                    results = reader.readtext(byte_arr.getvalue())
                    lines = [res[1] for res in results if res[1].strip()]
                    confidences = [res[2] for res in results]
                    extracted_text = "\n".join(lines).strip()
                    avg_confidence = (sum(confidences) / max(len(confidences), 1)) * 100
                    engine_used = "easyocr"
                except Exception as e:
                    logger.debug(f"EasyOCR extraction fallback: {e}")

        # 3. Fallback heuristic if neither is configured
        if not extracted_text:
            engine_used = "fallback"
            extracted_text = "[OCR: Image contains no detectable textual content or OCR engine not configured]"
            avg_confidence = 0.0

        return {
            "text": extracted_text,
            "confidence": round(avg_confidence, 2),
            "engine": engine_used,
            "width": img.width,
            "height": img.height,
        }


# Singleton instance
ocr_extractor = OCRExtractor()

"""Arabic OCR via vision-capable LLMs."""

from .ocr import OCRResult, PageResult, ocr_file
from .providers import get_provider

__all__ = ["OCRResult", "PageResult", "ocr_file", "get_provider"]
__version__ = "0.1.0"

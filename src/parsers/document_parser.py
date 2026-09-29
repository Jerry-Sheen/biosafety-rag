"""
Unified Master Document Parser.
Coordinates Docling (with OCR), HWP Parser, and Fallback Parsers.
Guarantees resilient processing across PDF, Excel, Word, CSV, HWP, and HWPX.
"""
from pathlib import Path
from typing import Dict, Any, Optional
from src.parsers.docling_parser import DoclingParser, HAS_DOCLING
from src.parsers.hwp_parser import HWPParser
from src.parsers.fallback_parser import FallbackParser
from src.security.logger import get_secure_logger

logger = get_secure_logger("document_parser")

SUPPORTED_EXTENSIONS = {
    ".pdf", ".docx", ".doc", ".xlsx", ".xls", ".csv",
    ".hwp", ".hwpx", ".txt", ".md"
}


class DocumentParser:
    """Master document parser with intelligent parser selection and fault tolerance."""

    def __init__(self, prefer_docling: bool = True, enable_ocr: bool = True):
        self.prefer_docling = prefer_docling and HAS_DOCLING
        self.enable_ocr = enable_ocr
        if self.prefer_docling:
            try:
                self.docling_parser = DoclingParser(enable_ocr=self.enable_ocr)
            except Exception as e:
                logger.warning(f"Failed to initialize Docling: {e}. Defaulting to fallback parser.")
                self.docling_parser = None
                self.prefer_docling = False
        else:
            self.docling_parser = None

    def parse_file(self, file_path: Path) -> Dict[str, Any]:
        """
        Parses an uploaded file using the most appropriate parser.
        Returns:
            {
                "success": bool,
                "text": str,
                "metadata": dict,
                "tables": list,
                "error": Optional[str]
            }
        """
        file_path = Path(file_path)
        ext = file_path.suffix.lower()

        if ext not in SUPPORTED_EXTENSIONS:
            return {
                "success": False,
                "text": "",
                "metadata": {"source": file_path.name},
                "tables": [],
                "error": f"Unsupported file extension: {ext}. Supported formats: {', '.join(SUPPORTED_EXTENSIONS)}"
            }

        logger.info(f"Dispatching parser for {file_path.name} (ext: {ext})")

        # 1. HWP / HWPX
        if ext in (".hwp", ".hwpx"):
            try:
                result = HWPParser.parse(file_path)
                result["success"] = True
                result["error"] = None
                return result
            except Exception as e:
                logger.error(f"HWP Parser failed on {file_path.name}: {e}")
                return {
                    "success": False,
                    "text": "",
                    "metadata": {"source": file_path.name, "format": ext},
                    "tables": [],
                    "error": f"HWP parsing error: {str(e)}"
                }

        # 2. Docling parser for PDF, Word, Excel (if preferred and available)
        if self.prefer_docling and self.docling_parser and ext in (".pdf", ".docx", ".xlsx"):
            try:
                logger.info(f"Attempting Docling parsing for {file_path.name}")
                result = self.docling_parser.parse(file_path)
                result["success"] = True
                result["error"] = None
                return result
            except Exception as e:
                logger.warning(f"Docling parsing failed on {file_path.name}: {e}. Retrying with fallback parser.")

        # 3. Fallback Parser (PDF, DOCX, XLSX, CSV, TXT, MD)
        try:
            result = FallbackParser.parse(file_path)
            result["success"] = True
            result["error"] = None
            return result
        except Exception as e:
            logger.error(f"Fallback parsing failed on {file_path.name}: {e}")
            return {
                "success": False,
                "text": "",
                "metadata": {"source": file_path.name, "format": ext},
                "tables": [],
                "error": f"Parsing failed: {str(e)}"
            }

"""
Docling-based document parser.
Uses IBM Docling for deep layout analysis, table extraction, and OCR on scanned/image-based documents.
Supports PDF, DOCX, PPTX, XLSX, HTML, and images.
"""
from pathlib import Path
from typing import Dict, Any, List, Optional
from src.security.logger import get_secure_logger

logger = get_secure_logger("docling_parser")

try:
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.datamodel.base_models import InputFormat
    HAS_DOCLING = True
except ImportError:
    HAS_DOCLING = False
    logger.warning("Docling package is not available. Please install 'docling' for advanced layout/OCR parsing.")


class DoclingParser:
    """Document parser leveraging IBM Docling with OCR capabilities."""

    def __init__(self, enable_ocr: bool = True):
        self.enable_ocr = enable_ocr
        self._converter: Optional[Any] = None

    def _get_converter(self):
        """Initializes DocumentConverter lazily with OCR configuration."""
        if not HAS_DOCLING:
            raise ImportError(
                "Docling is not installed in the environment. Install it via 'pip install docling'."
            )

        if self._converter is None:
            logger.info(f"Initializing Docling DocumentConverter (OCR Enabled: {self.enable_ocr})")
            
            # Configure pipeline options for PDF with OCR
            pipeline_options = PdfPipelineOptions()
            pipeline_options.do_ocr = self.enable_ocr
            pipeline_options.do_table_structure = True
            
            format_options = {
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
            }
            
            self._converter = DocumentConverter(format_options=format_options)
            
        return self._converter

    def parse(self, file_path: Path) -> Dict[str, Any]:
        """
        Parses a document with Docling.
        Returns:
            {
                "text": str (Markdown representation),
                "metadata": dict,
                "tables": list of markdown formatted tables,
                "parser": "docling"
            }
        """
        file_path = Path(file_path)
        logger.info(f"Running Docling parser on: {file_path.name}")

        converter = self._get_converter()
        conv_result = converter.convert(str(file_path))
        doc = conv_result.document

        # Export structured markdown
        markdown_text = doc.export_to_markdown()

        # Extract structured tables
        tables = []
        try:
            for table in getattr(doc, "tables", []):
                tables.append(table.export_to_markdown())
        except Exception as e:
            logger.warning(f"Could not export table directly: {e}")

        metadata = {
            "source": file_path.name,
            "parser": "docling",
            "file_size": file_path.stat().st_size,
            "ocr_applied": self.enable_ocr and file_path.suffix.lower() == ".pdf"
        }

        # Page count if available
        if hasattr(doc, "pages") and doc.pages:
            metadata["page_count"] = len(doc.pages)

        return {
            "text": markdown_text,
            "metadata": metadata,
            "tables": tables,
            "raw_doc": doc
        }

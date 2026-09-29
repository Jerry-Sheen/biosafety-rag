"""
HWP and HWPX document parser for Korean office documents.
Supports:
- .hwpx: XML-based Hangul documents (unzipped XML stream extraction)
- .hwp: OLE binary Hangul documents (OLE stream decompression & UTF-16LE decoding)
"""
import zipfile
import zlib
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, Any, List
from src.security.logger import get_secure_logger

logger = get_secure_logger("hwp_parser")

try:
    import olefile
    HAS_OLEFILE = True
except ImportError:
    HAS_OLEFILE = False
    logger.warning("olefile is not installed. Standard .hwp binary parsing will be limited.")


class HWPParser:
    """Extracts text, metadata, and tables from .hwp and .hwpx files."""

    @classmethod
    def parse(cls, file_path: Path) -> Dict[str, Any]:
        """
        Parses an HWP or HWPX file and returns structured output.
        Returns: {
            "text": str,
            "metadata": dict,
            "tables": list,
            "format": "hwpx" or "hwp"
        }
        """
        file_path = Path(file_path)
        ext = file_path.suffix.lower()

        if ext == ".hwpx":
            return cls._parse_hwpx(file_path)
        elif ext == ".hwp":
            return cls._parse_hwp(file_path)
        else:
            raise ValueError(f"Unsupported HWP format: {ext}")

    @classmethod
    def _parse_hwpx(cls, file_path: Path) -> Dict[str, Any]:
        """Extracts text from HWPX (Zip + XML)."""
        logger.info(f"Parsing HWPX document: {file_path.name}")
        full_text_parts: List[str] = []
        tables: List[str] = []

        try:
            with zipfile.ZipFile(file_path, "r") as zf:
                section_files = [name for name in zf.namelist() if name.startswith("Contents/section") and name.endswith(".xml")]
                section_files.sort()

                for sec_name in section_files:
                    xml_data = zf.read(sec_name)
                    root = ET.fromstring(xml_data)

                    # Extract text elements (typically under tags ending in 't')
                    sec_texts = []
                    for elem in root.iter():
                        tag_name = elem.tag.split("}")[-1] if "}" in elem.tag else elem.tag
                        if tag_name == "t" and elem.text:
                            sec_texts.append(elem.text)

                    if sec_texts:
                        full_text_parts.append("\n".join(sec_texts))

            text = "\n\n".join(full_text_parts)
            return {
                "text": text,
                "metadata": {
                    "source": file_path.name,
                    "format": "hwpx",
                    "file_size": file_path.stat().st_size
                },
                "tables": tables
            }
        except Exception as e:
            logger.error(f"Error parsing HWPX {file_path.name}: {e}")
            raise

    @classmethod
    def _parse_hwp(cls, file_path: Path) -> Dict[str, Any]:
        """Extracts text from binary HWP (OLE storage)."""
        logger.info(f"Parsing HWP binary document: {file_path.name}")
        if not HAS_OLEFILE:
            raise RuntimeError("olefile library is required to parse .hwp files. Please install olefile.")

        extracted_text_parts: List[str] = []

        try:
            ole = olefile.OleFileIO(str(file_path))
            dirs = ole.listdir()

            # Find all BodyText/Section streams
            section_streams = [
                d for d in dirs if len(d) >= 2 and d[0] == "BodyText" and d[1].startswith("Section")
            ]
            section_streams.sort(key=lambda x: x[1])

            for stream_path in section_streams:
                stream_name = "/".join(stream_path)
                data = ole.openstream(stream_path).read()

                # HWP section streams are compressed with Deflate without zlib wrapper (raw deflate)
                try:
                    decompressed = zlib.decompress(data, -15)
                except Exception:
                    try:
                        decompressed = zlib.decompress(data)
                    except Exception:
                        decompressed = data

                # Decode decompressed stream (HWP characters are encoded in UTF-16LE)
                # Filter printable text
                text_chars = []
                idx = 0
                while idx < len(decompressed) - 1:
                    code = int.from_bytes(decompressed[idx:idx+2], byteorder="little")
                    idx += 2
                    # Common Korean/ASCII characters and whitespace
                    if code in (10, 13, 9) or (32 <= code <= 126) or (0xAC00 <= code <= 0xD7A3):
                        text_chars.append(chr(code))
                    elif code == 0:
                        continue
                    elif 0x3000 <= code <= 0x303F or 0x2000 <= code <= 0x206F:
                        text_chars.append(chr(code))

                section_text = "".join(text_chars).strip()
                if section_text:
                    extracted_text_parts.append(section_text)

            ole.close()
            combined_text = "\n\n".join(extracted_text_parts)
            return {
                "text": combined_text,
                "metadata": {
                    "source": file_path.name,
                    "format": "hwp",
                    "sections_count": len(section_streams),
                    "file_size": file_path.stat().st_size
                },
                "tables": []
            }
        except Exception as e:
            logger.error(f"Error parsing HWP binary file {file_path.name}: {e}")
            raise

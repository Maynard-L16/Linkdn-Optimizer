"""Step 0 - Text extraction from uploaded files.

WHAT : Gets plain text out of an uploaded PDF (e.g. LinkedIn "Save to PDF"
       export or a resume) or TXT file.
WHY  : Every later NLP step works on plain text.
INPUT: file name + raw bytes (as Streamlit's uploader provides them).
OUTPUT: a plain-text string.

Limitation: scanned PDFs are images and contain no text layer; they would need
OCR, which is out of scope. The user is asked to paste the text instead.
"""

from __future__ import annotations

import io
from pathlib import Path

from pypdf import PdfReader
from pypdf.errors import PdfReadError

SUPPORTED_SUFFIXES = (".pdf", ".txt")


def extract_text_from_pdf(data: bytes) -> str:
    try:
        reader = PdfReader(io.BytesIO(data))
        if reader.is_encrypted:
            reader.decrypt("")  # many PDFs are "encrypted" with an empty password
        pages = [page.extract_text() or "" for page in reader.pages]
    except (PdfReadError, ValueError, KeyError) as exc:
        raise ValueError(f"Could not read this PDF ({exc}). Try another file or paste the text.") from exc
    text = "\n".join(pages).strip()
    if not text:
        raise ValueError("No selectable text found in this PDF (it may be a scanned image). Please paste the text instead.")
    return text


def extract_text_from_txt(data: bytes) -> str:
    for encoding in ("utf-8-sig", "cp1252", "latin-1"):  # cp1252 = typical Windows Notepad encoding
        try:
            return data.decode(encoding).strip()
        except UnicodeDecodeError:
            continue
    raise ValueError("Could not decode this text file.")


def extract_text(file_name: str, data: bytes) -> str:
    suffix = Path(file_name).suffix.lower()
    if suffix == ".pdf":
        return extract_text_from_pdf(data)
    if suffix == ".txt":
        return extract_text_from_txt(data)
    raise ValueError(f"Unsupported file type '{suffix}'. Upload one of: {', '.join(SUPPORTED_SUFFIXES)}.")

"""Header-based chunking for policy PDFs.

Minimal heuristic chunker: walks the extracted text line by line, treats
short "header-looking" lines (numbered sections like "2.1 Retry Limits",
or short ALL-CAPS lines like "PAYMENT POLICY") as section boundaries, and
groups everything until the next header into one chunk (header + body).
Text with no detectable headers falls back to a single chunk so ingestion
never silently produces zero chunks for a plain document.
"""

import re

# Matches either:
#  - numbered headings: "1", "1.2", "2.1.3 Retry Limits", etc.
#  - short, mostly-uppercase lines that read like a section title
_HEADER_PATTERN = re.compile(
    r"^(?:\d+(?:\.\d+)*[\.\)]?\s+\S.{0,80}|[A-Z][A-Z0-9\s\-\/&,]{3,80})$"
)

MAX_HEADER_LEN = 90


def _is_header(line: str) -> bool:
    line = line.strip()
    if not line or len(line) > MAX_HEADER_LEN:
        return False
    # Lines ending in typical sentence punctuation are prose, not headers.
    if line.endswith((".", ",", ";")) and not re.match(r"^\d+(\.\d+)*[\.\)]?\s", line):
        return False
    return bool(_HEADER_PATTERN.match(line))


def chunk_text_by_headers(text: str) -> list[str]:
    """Split `text` into a list of chunk strings, one per detected section."""
    lines = [l.strip() for l in text.splitlines()]

    chunks: list[str] = []
    current_header: str | None = None
    current_body: list[str] = []

    def flush():
        body = "\n".join(current_body).strip()
        if not current_header and not body:
            return
        content = f"{current_header}\n{body}".strip() if current_header else body
        if content:
            chunks.append(content)

    for line in lines:
        if not line:
            continue
        if _is_header(line):
            flush()
            current_header = line
            current_body = []
        else:
            current_body.append(line)
    flush()

    # Fallback: no headers detected anywhere (e.g. a single-paragraph doc).
    if not chunks and text.strip():
        chunks = [text.strip()]

    return chunks

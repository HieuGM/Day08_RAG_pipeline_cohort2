"""
Task 3 — Convert toàn bộ file trong data/landing/ thành Markdown.

Sử dụng MarkItDown của Microsoft:
    https://github.com/microsoft/markitdown

Cài đặt:
    pip install markitdown

Hướng dẫn:
    1. Scan toàn bộ file trong data/landing/ (PDF, DOCX, JSON)
    2. Convert sang Markdown
    3. Lưu vào data/standardized/ giữ nguyên cấu trúc thư mục
"""

import json
import re
import unicodedata
from datetime import datetime
from pathlib import Path
from typing import Any

LANDING_DIR = Path(__file__).parent.parent / "data" / "landing"
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "standardized"
SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".doc", ".json", ".html", ".htm", ".txt", ".md"}


def _load_markitdown():
    try:
        from markitdown import MarkItDown

        return MarkItDown()
    except Exception:
        return None


def _metadata_header(title: str, source_file: Path, extra: dict[str, Any] | None = None) -> str:
    lines = [
        f"# {title or source_file.stem}",
        "",
        f"**Source file:** {source_file.as_posix()}",
        f"**Converted:** {datetime.now().astimezone().isoformat(timespec='seconds')}",
    ]
    for key, value in (extra or {}).items():
        if value:
            label = key.replace("_", " ").title()
            lines.append(f"**{label}:** {value}")
    lines.extend(["", "---", ""])
    return "\n".join(lines)


def _read_text_file(filepath: Path) -> str:
    for encoding in ("utf-8", "utf-8-sig", "cp1258", "latin-1"):
        try:
            return filepath.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    return filepath.read_text(encoding="utf-8", errors="ignore")


def _convert_with_markitdown(filepath: Path) -> str:
    md = _load_markitdown()
    if not md:
        return ""
    result = md.convert(str(filepath))
    return getattr(result, "text_content", "") or ""


def _convert_pdf_fallback(filepath: Path) -> str:
    try:
        from pypdf import PdfReader

        reader = PdfReader(str(filepath))
        return "\n\n".join(page.extract_text() or "" for page in reader.pages).strip()
    except Exception:
        pass

    try:
        from PyPDF2 import PdfReader

        reader = PdfReader(str(filepath))
        return "\n\n".join(page.extract_text() or "" for page in reader.pages).strip()
    except Exception:
        pass

    try:
        import fitz

        with fitz.open(str(filepath)) as doc:
            return "\n\n".join(page.get_text() for page in doc).strip()
    except Exception:
        return ""


def _convert_docx_fallback(filepath: Path) -> str:
    try:
        import docx

        document = docx.Document(str(filepath))
        return "\n\n".join(p.text for p in document.paragraphs if p.text.strip()).strip()
    except Exception:
        return ""


def _convert_legacy_doc_fallback(filepath: Path) -> str:
    """Extract readable UTF-16LE text from older binary .doc files."""
    data = filepath.read_bytes()
    text = data.decode("utf-16le", errors="ignore")
    for needle in ("CHÍNH PHỦ", "CỘNG HÒA", "NGHỊ ĐỊNH", "LUẬT"):
        index = text.find(needle)
        if index >= 0:
            text = text[index:]
            break

    stop_candidates = [
        index
        for needle in ("Normal", "Heading 1", "Times New Roman", "Root Entry", "WordDocument", "SummaryInformation")
        for index in [text.find(needle)]
        if index > 0
    ]
    if stop_candidates:
        text = text[: min(stop_candidates)]

    text = (
        text.replace("\r", "\n")
        .replace("\x0b", "\n")
        .replace("\x07", "\n")
        .replace("\x08", " ")
        .replace("\x00", "")
    )
    text = re.sub(r"[\x01-\x06\x09\x0e-\x1f]+", " ", text)

    cleaned_lines: list[str] = []
    for raw_line in text.splitlines():
        line = re.sub(r"\s+", " ", raw_line).strip()
        if len(line) < 2:
            continue
        useful_chars = sum(_is_legal_text_char(ch) for ch in line)
        if useful_chars / max(len(line), 1) < 0.75:
            continue
        cleaned_lines.append(line)

    cleaned = "\n\n".join(cleaned_lines).strip()
    return cleaned if len(cleaned) > 200 else ""


def _is_legal_text_char(ch: str) -> bool:
    if ch.isdigit() or ch.isspace() or ch in ".,;:!?()[]/-–+%\"'“”‘’…_□":
        return True
    return "LATIN" in unicodedata.name(ch, "")


def _html_to_markdown(html: str, title: str = "") -> tuple[str, str]:
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript", "iframe", "svg", "form"]):
        tag.decompose()
    page_title = title or (soup.find("h1").get_text(" ", strip=True) if soup.find("h1") else "")
    page_title = page_title or (soup.title.get_text(" ", strip=True) if soup.title else "")

    lines: list[str] = []
    for node in soup.find_all(["h1", "h2", "h3", "p", "li"], recursive=True):
        text = " ".join(node.get_text(" ", strip=True).split())
        if not text:
            continue
        if node.name == "h1":
            lines.extend([f"# {text}", ""])
        elif node.name == "h2":
            lines.extend([f"## {text}", ""])
        elif node.name == "h3":
            lines.extend([f"### {text}", ""])
        elif node.name == "li":
            lines.append(f"- {text}")
        else:
            lines.extend([text, ""])

    return page_title, "\n".join(lines).strip()


def _convert_json_article(filepath: Path) -> str:
    data = json.loads(_read_text_file(filepath))
    title = data.get("title") or filepath.stem
    content = (
        data.get("content_markdown")
        or data.get("markdown")
        or data.get("content")
        or data.get("text")
        or ""
    )
    if isinstance(content, list):
        content = "\n\n".join(str(item) for item in content)
    header = _metadata_header(
        title,
        filepath.relative_to(LANDING_DIR),
        {
            "url": data.get("url", ""),
            "source_domain": data.get("source_domain", ""),
            "published_at": data.get("published_at", ""),
            "date_crawled": data.get("date_crawled", ""),
        },
    )
    return header + str(content).strip() + "\n"


def convert_file(filepath: Path) -> str:
    """Convert one landing file to markdown text."""
    suffix = filepath.suffix.lower()

    if suffix == ".json":
        return _convert_json_article(filepath)
    if suffix in {".txt", ".md"}:
        title = filepath.stem
        return _metadata_header(title, filepath.relative_to(LANDING_DIR)) + _read_text_file(filepath).strip() + "\n"
    if suffix in {".html", ".htm"}:
        html = _read_text_file(filepath)
        title, body = _html_to_markdown(html)
        return _metadata_header(title or filepath.stem, filepath.relative_to(LANDING_DIR)) + body + "\n"

    text = _convert_with_markitdown(filepath)
    if not text and suffix == ".pdf":
        text = _convert_pdf_fallback(filepath)
    if not text and suffix in {".docx", ".doc"}:
        text = _convert_docx_fallback(filepath)
    if not text and suffix == ".doc":
        text = _convert_legacy_doc_fallback(filepath)
    if not text and suffix == ".pdf":
        fallback_doc = filepath.with_suffix(".congbao.doc")
        if fallback_doc.exists():
            text = _convert_legacy_doc_fallback(fallback_doc)
    if not text:
        raise RuntimeError(f"Không convert được {filepath}")

    return _metadata_header(filepath.stem, filepath.relative_to(LANDING_DIR)) + text.strip() + "\n"


def _output_path_for(filepath: Path) -> Path:
    relative = filepath.relative_to(LANDING_DIR)
    return (OUTPUT_DIR / relative).with_suffix(".md")


def convert_legal_docs():
    """Convert PDF/DOCX files trong data/landing/legal/ sang markdown."""
    legal_dir = LANDING_DIR / "legal"
    output_dir = OUTPUT_DIR / "legal"
    output_dir.mkdir(parents=True, exist_ok=True)

    for filepath in legal_dir.iterdir():
        if filepath.suffix.lower() in (".pdf", ".docx", ".doc"):
            print(f"Converting: {filepath.name}")
            output_path = output_dir / f"{filepath.stem}.md"
            output_path.write_text(convert_file(filepath), encoding="utf-8")
            print(f"  [OK] Saved: {output_path}")


def convert_news_articles():
    """Convert JSON crawled articles trong data/landing/news/ sang markdown."""
    news_dir = LANDING_DIR / "news"
    output_dir = OUTPUT_DIR / "news"
    output_dir.mkdir(parents=True, exist_ok=True)

    for filepath in news_dir.iterdir():
        if filepath.suffix.lower() in (".json", ".html", ".htm", ".txt", ".md"):
            print(f"Converting: {filepath.name}")
            output_path = output_dir / f"{filepath.stem}.md"
            output_path.write_text(convert_file(filepath), encoding="utf-8")
            print(f"  [OK] Saved: {output_path}")


def convert_all_files():
    """Generic converter that preserves every supported subdirectory."""
    converted = 0
    skipped = 0
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for filepath in LANDING_DIR.rglob("*"):
        if not filepath.is_file() or filepath.name.startswith("."):
            continue
        if filepath.suffix.lower() not in SUPPORTED_EXTENSIONS:
            skipped += 1
            continue
        output_path = _output_path_for(filepath)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Converting: {filepath.relative_to(LANDING_DIR)}")
        output_path.write_text(convert_file(filepath), encoding="utf-8")
        converted += 1
        print(f"  [OK] Saved: {output_path.relative_to(OUTPUT_DIR.parent)}")

    return converted, skipped


def convert_all():
    """Convert toàn bộ files."""
    print("=" * 50)
    print("Task 3: Convert to Markdown (MarkItDown)")
    print("=" * 50)

    converted, skipped = convert_all_files()

    print(f"\n[OK] Done! Converted {converted} files, skipped {skipped}. Output tai: {OUTPUT_DIR}")


if __name__ == "__main__":
    convert_all()

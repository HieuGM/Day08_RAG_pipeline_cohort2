"""
Task 2 — Crawl bài báo về nghệ sĩ liên quan tới ma tuý.

Hướng dẫn:
    1. Crawl tối thiểu 5 bài báo từ các trang tin tức Việt Nam.
    2. Sử dụng Crawl4AI hoặc thư viện crawling tương tự.
    3. Lưu output vào data/landing/news/
    4. Mỗi bài lưu 1 file JSON với metadata (url, title, date_crawled, content).

Cài đặt:
    pip install crawl4ai
"""

import asyncio
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

DATA_DIR = Path(__file__).parent.parent / "data" / "landing" / "news"


def setup_directory():
    """Tạo thư mục data/landing/news/ nếu chưa có."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)


ARTICLE_URLS = [
    "https://tuoitre.vn/bat-nguoi-mau-an-tay-ca-si-chi-dan-co-tien-truc-phuong-do-lien-quan-ma-tuy-20241114114826655.htm",
    "https://vnexpress.net/nguoi-mau-andrea-aybar-bi-tinh-nghi-lien-quan-ma-tuy-4814289.html",
    "https://vnexpress.net/dien-vien-le-hang-bi-dieu-tra-mua-ban-ma-tuy-4597048.html",
    "https://vnexpress.net/dien-vien-hai-bi-tam-giu-vi-lien-quan-ma-tuy-4475240.html",
    "https://ngoisao.vnexpress.net/nam-than-lai-nga-nhikolai-dinh-bi-bat-4762594.html",
]


def _slugify(text: str, max_length: int = 80) -> str:
    """Create stable ASCII-ish file names from Vietnamese titles or URLs."""
    normalized = text.lower()
    replacements = {
        "đ": "d",
        "à": "a", "á": "a", "ả": "a", "ã": "a", "ạ": "a",
        "ă": "a", "ằ": "a", "ắ": "a", "ẳ": "a", "ẵ": "a", "ặ": "a",
        "â": "a", "ầ": "a", "ấ": "a", "ẩ": "a", "ẫ": "a", "ậ": "a",
        "è": "e", "é": "e", "ẻ": "e", "ẽ": "e", "ẹ": "e",
        "ê": "e", "ề": "e", "ế": "e", "ể": "e", "ễ": "e", "ệ": "e",
        "ì": "i", "í": "i", "ỉ": "i", "ĩ": "i", "ị": "i",
        "ò": "o", "ó": "o", "ỏ": "o", "õ": "o", "ọ": "o",
        "ô": "o", "ồ": "o", "ố": "o", "ổ": "o", "ỗ": "o", "ộ": "o",
        "ơ": "o", "ờ": "o", "ớ": "o", "ở": "o", "ỡ": "o", "ợ": "o",
        "ù": "u", "ú": "u", "ủ": "u", "ũ": "u", "ụ": "u",
        "ư": "u", "ừ": "u", "ứ": "u", "ử": "u", "ữ": "u", "ự": "u",
        "ỳ": "y", "ý": "y", "ỷ": "y", "ỹ": "y", "ỵ": "y",
    }
    for source, target in replacements.items():
        normalized = normalized.replace(source, target)
    normalized = re.sub(r"[^a-z0-9]+", "-", normalized).strip("-")
    return (normalized[:max_length].strip("-") or "article")


def _metadata_value(soup: Any, *names: str) -> str:
    for name in names:
        tag = soup.find("meta", attrs={"property": name}) or soup.find("meta", attrs={"name": name})
        if tag and tag.get("content"):
            return tag["content"].strip()
    return ""


def _extract_markdown_from_html(html: str, url: str) -> tuple[str, str, str]:
    """Best-effort article extraction for Vietnamese news pages."""
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript", "iframe", "svg", "form"]):
        tag.decompose()

    title = (
        _metadata_value(soup, "og:title", "twitter:title")
        or (soup.find("h1").get_text(" ", strip=True) if soup.find("h1") else "")
        or (soup.title.get_text(" ", strip=True) if soup.title else "")
        or url
    )
    published_at = _metadata_value(
        soup,
        "article:published_time",
        "pubdate",
        "publishdate",
        "date",
        "dc.date.issued",
    )
    if not published_at:
        time_tag = soup.find("time")
        if time_tag:
            published_at = (time_tag.get("datetime") or time_tag.get_text(" ", strip=True)).strip()

    selectors = [
        "article",
        ".fck_detail",
        ".content-detail",
        ".detail-content",
        ".article__body",
        ".article-content",
        ".maincontent",
        ".content-news-detail",
        ".detail__content",
    ]
    candidates = []
    for selector in selectors:
        candidates.extend(soup.select(selector))
    if not candidates:
        candidates = [soup.body or soup]
    body = max(candidates, key=lambda node: len(node.get_text(" ", strip=True)))

    lines: list[str] = [f"# {title}", ""]
    description = _metadata_value(soup, "og:description", "description")
    if description:
        lines.extend([f"> {description}", ""])

    seen: set[str] = set()
    for node in body.find_all(["h2", "h3", "p", "li"], recursive=True):
        text = re.sub(r"\s+", " ", node.get_text(" ", strip=True)).strip()
        if len(text) < 20 or text in seen:
            continue
        seen.add(text)
        if node.name == "h2":
            lines.extend([f"## {text}", ""])
        elif node.name == "h3":
            lines.extend([f"### {text}", ""])
        elif node.name == "li":
            lines.append(f"- {text}")
        else:
            lines.extend([text, ""])

    markdown = "\n".join(lines).strip()
    return title, published_at, markdown


def _fetch_article_with_requests(url: str) -> dict:
    import requests

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/125.0 Safari/537.36"
        )
    }
    response = requests.get(url, headers=headers, timeout=30)
    response.raise_for_status()
    title, published_at, content_markdown = _extract_markdown_from_html(response.text, url)
    return {
        "title": title,
        "published_at": published_at,
        "content_markdown": content_markdown,
        "fetcher": "requests+beautifulsoup",
    }


async def crawl_article(url: str) -> dict:
    """
    Crawl một bài báo và trả về dict chứa metadata + content.

    Returns:
        {
            "url": str,
            "title": str,
            "date_crawled": str (ISO format),
            "content_markdown": str
        }
    """
    date_crawled = datetime.now().astimezone().isoformat(timespec="seconds")
    source_domain = urlparse(url).netloc

    try:
        from crawl4ai import AsyncWebCrawler

        async with AsyncWebCrawler() as crawler:
            result = await crawler.arun(url=url)
        metadata = getattr(result, "metadata", {}) or {}
        content_markdown = getattr(result, "markdown", "") or ""
        if hasattr(content_markdown, "raw_markdown"):
            content_markdown = content_markdown.raw_markdown
        title = metadata.get("title") or _extract_markdown_title(content_markdown) or url
        article = {
            "title": title,
            "published_at": metadata.get("published_time") or metadata.get("date") or "",
            "content_markdown": content_markdown,
            "fetcher": "crawl4ai",
        }
    except Exception as exc:
        article = _fetch_article_with_requests(url)
        article["crawl4ai_error"] = str(exc)

    return {
        "url": url,
        "source_domain": source_domain,
        "title": article["title"],
        "published_at": article.get("published_at", ""),
        "date_crawled": date_crawled,
        "content_markdown": article["content_markdown"],
        "fetcher": article.get("fetcher", "unknown"),
        "crawl4ai_error": article.get("crawl4ai_error", ""),
    }


def _extract_markdown_title(markdown: str) -> str:
    for line in markdown.splitlines():
        line = line.strip()
        if line.startswith("#"):
            return line.lstrip("#").strip()
    return ""


async def crawl_all():
    """Crawl toàn bộ bài báo trong ARTICLE_URLS."""
    setup_directory()

    for i, url in enumerate(ARTICLE_URLS, 1):
        print(f"[{i}/{len(ARTICLE_URLS)}] Crawling: {url}")
        article = await crawl_article(url)

        # Lưu file JSON
        slug = _slugify(article.get("title") or urlparse(url).path)
        filename = f"article_{i:02d}_{slug}.json"
        filepath = DATA_DIR / filename
        filepath.write_text(json.dumps(article, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  [OK] Saved: {filepath}")


if __name__ == "__main__":
    asyncio.run(crawl_all())

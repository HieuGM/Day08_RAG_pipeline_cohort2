"""
Task 8 - PageIndex vectorless retrieval.

Primary path:
    - Upload PDF documents to PageIndex cloud.
    - Query PageIndex retrieval/chat APIs using stored doc_ids.

Offline path:
    - Build a local PageIndex-style structural tree from standardized markdown.
    - Search sections by heading/path/content without embeddings or vector DB.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import requests
from dotenv import load_dotenv


# =============================================================================
# Configuration
# =============================================================================

load_dotenv()

ROOT_DIR = Path(__file__).parent.parent
LANDING_DIR = ROOT_DIR / "data" / "landing"
STANDARDIZED_DIR = ROOT_DIR / "data" / "standardized"
INDEX_DIR = ROOT_DIR / "data" / "index"

PAGEINDEX_API_BASE = os.getenv("PAGEINDEX_API_BASE", "https://api.pageindex.ai").rstrip("/")
PAGEINDEX_API_KEY = os.getenv("PAGEINDEX_API_KEY", "")
PAGEINDEX_MANIFEST_PATH = INDEX_DIR / "pageindex_manifest.json"
PAGEINDEX_LOCAL_INDEX_PATH = INDEX_DIR / "pageindex_local_index.json"
LOCAL_INDEX_VERSION = 2

REQUEST_TIMEOUT = float(os.getenv("PAGEINDEX_REQUEST_TIMEOUT", "60"))
UPLOAD_POLL_INTERVAL = float(os.getenv("PAGEINDEX_UPLOAD_POLL_INTERVAL", "8"))
UPLOAD_TIMEOUT = float(os.getenv("PAGEINDEX_UPLOAD_TIMEOUT", "900"))
RETRIEVAL_POLL_INTERVAL = float(os.getenv("PAGEINDEX_RETRIEVAL_POLL_INTERVAL", "3"))
RETRIEVAL_TIMEOUT = float(os.getenv("PAGEINDEX_RETRIEVAL_TIMEOUT", "120"))
MAX_DOCS_PER_QUERY = int(os.getenv("PAGEINDEX_MAX_DOCS_PER_QUERY", "8"))
MAX_LOCAL_SECTION_CHARS = int(os.getenv("PAGEINDEX_LOCAL_SECTION_CHARS", "3500"))
MIN_LOCAL_TOP_SCORE = float(os.getenv("PAGEINDEX_MIN_LOCAL_TOP_SCORE", "0.14"))


# =============================================================================
# Upload / indexing
# =============================================================================

def upload_documents(
    wait: bool = False,
    include_local_index: bool = True,
) -> list[dict]:
    """
    Upload landing PDFs to PageIndex and persist doc_ids in a local manifest.

    PageIndex cloud currently processes PDFs for document upload. Standardized
    markdown is also indexed locally for an offline vectorless fallback.
    """
    load_dotenv()
    api_key = os.getenv("PAGEINDEX_API_KEY", PAGEINDEX_API_KEY)
    manifest = _load_manifest()
    uploaded: list[dict] = []

    if include_local_index:
        local_index = _build_local_index()
        _save_local_index(local_index)

    if not api_key:
        return [
            {
                "mode": "local_fallback",
                "status": "ready",
                "message": "PAGEINDEX_API_KEY is missing; built local structural index only.",
                "documents": len(_load_local_index().get("documents", [])),
            }
        ]

    existing_by_name = _list_remote_documents_by_name(api_key)
    for pdf_path in _iter_uploadable_pdfs():
        fingerprint = _file_fingerprint(pdf_path)
        record = manifest.get("api_documents", {}).get(fingerprint)
        if record and record.get("doc_id"):
            if wait:
                record = _refresh_document_status(api_key, record, wait=True)
                manifest["api_documents"][fingerprint] = record
            uploaded.append(record)
            continue

        remote_match = existing_by_name.get(pdf_path.name)
        if remote_match:
            record = _manifest_record_from_remote(pdf_path, fingerprint, remote_match)
            if wait:
                record = _refresh_document_status(api_key, record, wait=True)
            manifest.setdefault("api_documents", {})[fingerprint] = record
            uploaded.append(record)
            continue

        result = _submit_document(api_key, pdf_path)
        doc_id = result.get("doc_id") or result.get("id")
        record = {
            "doc_id": doc_id,
            "status": result.get("status", "submitted"),
            "file_name": pdf_path.name,
            "path": _relative_path(pdf_path),
            "sha256": fingerprint,
            "uploaded_at": _now_iso(),
        }
        if wait and doc_id:
            record = _refresh_document_status(api_key, record, wait=True)
        manifest.setdefault("api_documents", {})[fingerprint] = record
        uploaded.append(record)

    manifest["updated_at"] = _now_iso()
    _save_manifest(manifest)
    return uploaded


# =============================================================================
# Search
# =============================================================================

def pageindex_search(query: str, top_k: int = 5) -> list[dict]:
    """
    Vectorless retrieval using PageIndex.

    Returns:
        List of {'content': str, 'score': float, 'metadata': dict, 'source': 'pageindex'}.
    """
    query = (query or "").strip()
    if not query or top_k <= 0:
        return []

    load_dotenv()
    api_key = os.getenv("PAGEINDEX_API_KEY", PAGEINDEX_API_KEY)
    if api_key and not _external_api_disabled_for_tests():
        results = _pageindex_cloud_search(api_key, query, top_k)
        if results:
            return results[:top_k]

    return _local_vectorless_search(query, top_k)


def _pageindex_cloud_search(api_key: str, query: str, top_k: int) -> list[dict]:
    doc_ids = _get_query_doc_ids(api_key)
    if not doc_ids:
        return []

    results: list[dict] = []

    for doc_rank, doc_id in enumerate(doc_ids[:MAX_DOCS_PER_QUERY], start=1):
        try:
            doc_results = _query_retrieval_api(api_key, doc_id, query, top_k)
        except Exception as exc:
            print(f"[WARN] PageIndex retrieval failed for {doc_id}: {exc}")
            doc_results = []

        for item_rank, item in enumerate(doc_results, start=1):
            item["score"] = float(item.get("score", 0.0) or _rank_score(doc_rank, item_rank))
            results.append(item)

    results.sort(key=lambda item: item["score"], reverse=True)
    if results:
        return results[:top_k]

    if doc_ids and os.getenv("PAGEINDEX_USE_CHAT_FALLBACK", "1") == "1":
        try:
            chat_result = _query_chat_api(api_key, doc_ids, query)
            if chat_result:
                return [chat_result]
        except Exception as exc:
            print(f"[WARN] PageIndex chat fallback failed: {exc}")

    return []


def _get_query_doc_ids(api_key: str) -> list[str]:
    manifest = _load_manifest()
    records = list(manifest.get("api_documents", {}).values())
    doc_ids = [
        str(record["doc_id"])
        for record in records
        if record.get("doc_id")
        and (
            record.get("status") == "completed"
            or record.get("retrieval_ready") is True
        )
    ]
    if doc_ids:
        return doc_ids

    if records and os.getenv("PAGEINDEX_REFRESH_MANIFEST", "0") == "1":
        refreshed_ids: list[str] = []
        for fingerprint, record in list(manifest.get("api_documents", {}).items()):
            refreshed = _refresh_document_status(api_key, record, wait=False)
            manifest["api_documents"][fingerprint] = refreshed
            if refreshed.get("status") == "completed" or refreshed.get("retrieval_ready") is True:
                refreshed_ids.append(str(refreshed["doc_id"]))
        manifest["updated_at"] = _now_iso()
        _save_manifest(manifest)
        if refreshed_ids:
            return refreshed_ids

    if os.getenv("PAGEINDEX_AUTO_UPLOAD", "0") == "1":
        records = upload_documents(wait=False, include_local_index=False)
        return [
            str(record["doc_id"])
            for record in records
            if record.get("doc_id")
            and (
                record.get("status") == "completed"
                or record.get("retrieval_ready") is True
            )
        ]

    if os.getenv("PAGEINDEX_DISCOVER_REMOTE", "0") != "1":
        return []

    return _list_completed_remote_doc_ids(api_key)


# =============================================================================
# PageIndex API / SDK wrappers
# =============================================================================

def _submit_document(api_key: str, pdf_path: Path) -> dict:
    client = _get_sdk_client(api_key)
    if client is not None:
        return client.submit_document(str(pdf_path))

    with pdf_path.open("rb") as file_obj:
        response = requests.post(
            f"{PAGEINDEX_API_BASE}/doc/",
            headers={"api_key": api_key},
            files={"file": (pdf_path.name, file_obj, "application/pdf")},
            timeout=REQUEST_TIMEOUT,
        )
    response.raise_for_status()
    return response.json()


def _get_tree(api_key: str, doc_id: str, summary: bool = False) -> dict:
    client = _get_sdk_client(api_key)
    if client is not None:
        try:
            return client.get_tree(doc_id, node_summary=summary)
        except TypeError:
            return client.get_tree(doc_id)

    response = requests.get(
        f"{PAGEINDEX_API_BASE}/doc/{doc_id}/",
        headers={"api_key": api_key},
        params={"type": "tree", "summary": str(summary).lower()},
        timeout=REQUEST_TIMEOUT,
    )
    response.raise_for_status()
    return response.json()


def _list_documents(api_key: str, limit: int = 100, offset: int = 0) -> dict:
    client = _get_sdk_client(api_key)
    if client is not None and hasattr(client, "list_documents"):
        return client.list_documents(limit=limit, offset=offset)

    response = requests.get(
        f"{PAGEINDEX_API_BASE}/docs",
        headers={"api_key": api_key},
        params={"limit": limit, "offset": offset},
        timeout=REQUEST_TIMEOUT,
    )
    response.raise_for_status()
    return response.json()


def _query_retrieval_api(api_key: str, doc_id: str, query: str, top_k: int) -> list[dict]:
    tree = _get_tree(api_key, doc_id, summary=False)
    if tree.get("status") not in {None, "completed"} or tree.get("retrieval_ready") is False:
        return []

    response = requests.post(
        f"{PAGEINDEX_API_BASE}/retrieval/",
        headers={"api_key": api_key, "Content-Type": "application/json"},
        json={"doc_id": doc_id, "query": query, "thinking": True},
        timeout=REQUEST_TIMEOUT,
    )
    response.raise_for_status()
    retrieval_id = response.json().get("retrieval_id")
    if not retrieval_id:
        return []

    deadline = time.monotonic() + RETRIEVAL_TIMEOUT
    while time.monotonic() < deadline:
        poll = requests.get(
            f"{PAGEINDEX_API_BASE}/retrieval/{retrieval_id}/",
            headers={"api_key": api_key},
            timeout=REQUEST_TIMEOUT,
        )
        poll.raise_for_status()
        data = poll.json()
        status = data.get("status")
        if status == "completed":
            return _format_retrieval_results(data, doc_id, top_k)
        if status == "failed":
            return []
        time.sleep(RETRIEVAL_POLL_INTERVAL)

    return []


def _query_chat_api(api_key: str, doc_ids: list[str], query: str) -> Optional[dict]:
    payload: dict[str, Any] = {
        "messages": [{"role": "user", "content": query}],
        "stream": False,
        "temperature": 0,
        "enable_citations": True,
    }
    if doc_ids:
        payload["doc_id"] = doc_ids[:MAX_DOCS_PER_QUERY]

    client = _get_sdk_client(api_key)
    if client is not None and hasattr(client, "chat_completions"):
        response = client.chat_completions(**payload)
    else:
        api_response = requests.post(
            f"{PAGEINDEX_API_BASE}/chat/completions",
            headers={"api_key": api_key, "Content-Type": "application/json"},
            json=payload,
            timeout=max(REQUEST_TIMEOUT, 120),
        )
        api_response.raise_for_status()
        response = api_response.json()

    content = (
        response.get("choices", [{}])[0]
        .get("message", {})
        .get("content", "")
    )
    if not content:
        return None

    return {
        "content": content,
        "score": 1.0,
        "metadata": {
            "provider": "pageindex_chat",
            "doc_ids": doc_ids[:MAX_DOCS_PER_QUERY],
            "usage": response.get("usage", {}),
        },
        "source": "pageindex",
    }


def _get_sdk_client(api_key: str):
    try:
        from pageindex import PageIndexClient
    except ImportError:
        return None
    return PageIndexClient(api_key=api_key)


def _format_retrieval_results(data: dict, doc_id: str, top_k: int) -> list[dict]:
    results: list[dict] = []
    for node_rank, node in enumerate(data.get("retrieved_nodes", []), start=1):
        node_title = str(node.get("title", "") or "")
        node_id = str(node.get("node_id", "") or "")
        relevant_contents = node.get("relevant_contents") or []
        for content_rank, content_item in enumerate(relevant_contents, start=1):
            content = (
                content_item.get("relevant_content")
                or content_item.get("text")
                or content_item.get("markdown")
                or ""
            )
            if not content:
                continue
            score = _rank_score(node_rank, content_rank)
            results.append(
                {
                    "content": str(content),
                    "score": score,
                    "metadata": {
                        "provider": "pageindex_retrieval",
                        "doc_id": doc_id,
                        "retrieval_id": data.get("retrieval_id"),
                        "query": data.get("query"),
                        "node_id": node_id,
                        "title": node_title,
                        "page_index": content_item.get("page_index"),
                    },
                    "source": "pageindex",
                }
            )

    results.sort(key=lambda item: item["score"], reverse=True)
    return results[:top_k]


# =============================================================================
# Local PageIndex-style structural fallback
# =============================================================================

def _local_vectorless_search(query: str, top_k: int) -> list[dict]:
    index = _load_or_build_local_index()
    expanded_query = _expand_query(query)
    query_tokens = _tokenize(expanded_query)
    scored: list[dict] = []
    seen_results: set[str] = set()

    for section in index.get("sections", []):
        score = _score_section(query, expanded_query, query_tokens, section)
        if score <= 0:
            continue
        result_key = _section_result_key(section)
        if result_key in seen_results:
            continue
        seen_results.add(result_key)
        scored.append(
            {
                "content": section["content"],
                "score": float(score),
                "metadata": {
                    "provider": "pageindex_local_tree",
                    "document": section.get("document"),
                    "source_path": section.get("source_path"),
                    "doc_type": section.get("doc_type"),
                    "title": section.get("title"),
                    "section_path": section.get("section_path"),
                    "line_start": section.get("line_start"),
                    "line_end": section.get("line_end"),
                },
                "source": "pageindex",
            }
        )

    scored.sort(key=lambda item: item["score"], reverse=True)
    if scored and scored[0]["score"] < MIN_LOCAL_TOP_SCORE:
        return []
    return scored[:top_k]


def _load_or_build_local_index() -> dict:
    current_fingerprints = _standardized_fingerprints()
    if PAGEINDEX_LOCAL_INDEX_PATH.exists():
        try:
            index = json.loads(PAGEINDEX_LOCAL_INDEX_PATH.read_text(encoding="utf-8"))
            if (
                index.get("version") == LOCAL_INDEX_VERSION
                and index.get("fingerprints") == current_fingerprints
            ):
                return index
        except (json.JSONDecodeError, OSError):
            pass

    index = _build_local_index()
    _save_local_index(index)
    return index


def _build_local_index() -> dict:
    documents: list[dict] = []
    sections: list[dict] = []
    seen_hashes: set[str] = set()

    for md_path in _iter_standardized_markdown():
        content = md_path.read_text(encoding="utf-8", errors="ignore")
        content_hash = hashlib.sha256(_normalize_for_dedupe(content).encode("utf-8")).hexdigest()
        if content_hash in seen_hashes:
            continue
        seen_hashes.add(content_hash)

        doc = {
            "document": md_path.stem,
            "source_path": _relative_path(md_path),
            "doc_type": md_path.parent.name,
            "sha256": content_hash,
        }
        documents.append(doc)
        sections.extend(_extract_structural_sections(md_path, content, doc))

    return {
        "version": LOCAL_INDEX_VERSION,
        "built_at": _now_iso(),
        "fingerprints": _standardized_fingerprints(),
        "documents": documents,
        "sections": sections,
    }


def _extract_structural_sections(md_path: Path, content: str, doc: dict) -> list[dict]:
    lines = content.splitlines()
    headings = _find_headings(lines)
    if not headings:
        headings = [(1, 0, md_path.stem)]

    sections: list[dict] = []
    for idx, (level, start_line, title) in enumerate(headings):
        end_line = headings[idx + 1][1] if idx + 1 < len(headings) else len(lines)
        body_lines = lines[start_line:end_line]
        body = "\n".join(body_lines).strip()
        if not body:
            continue

        section_path = _section_path(headings, idx)
        sections.append(
            {
                "document": doc["document"],
                "source_path": doc["source_path"],
                "doc_type": doc["doc_type"],
                "title": title,
                "section_path": section_path,
                "level": level,
                "line_start": start_line + 1,
                "line_end": end_line,
                "content": _compact_section(body),
            }
        )
    return sections


def _find_headings(lines: list[str]) -> list[tuple[int, int, str]]:
    headings: list[tuple[int, int, str]] = []
    for idx, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            continue

        md_match = re.match(r"^(#{1,6})\s+(.+)$", stripped)
        if md_match:
            headings.append((len(md_match.group(1)), idx, md_match.group(2).strip()))
            continue

        legal_match = re.match(
            r"^(Chương\s+[IVXLCDM\d]+|Mục\s+\d+|Điều\s+\d+[a-zA-Z]?\.)\s*(.*)$",
            stripped,
            flags=re.IGNORECASE,
        )
        if legal_match:
            title = stripped
            if stripped.lower().startswith(("chương", "mục")):
                next_title = _next_non_empty_line(lines, idx + 1)
                if next_title:
                    title = f"{stripped} - {next_title}"
            level = 2 if stripped.lower().startswith("chương") else 3
            headings.append((level, idx, title))

    if headings and headings[0][1] != 0:
        headings.insert((0), (1, 0, "Document metadata and introduction"))
    return headings


def _section_path(headings: list[tuple[int, int, str]], idx: int) -> str:
    current_level = headings[idx][0]
    parts = [headings[idx][2]]
    min_level = current_level
    for prev_level, _, prev_title in reversed(headings[:idx]):
        if prev_level < min_level:
            parts.append(prev_title)
            min_level = prev_level
    return " > ".join(reversed(parts))


def _next_non_empty_line(lines: list[str], start: int) -> str:
    for line in lines[start:start + 4]:
        stripped = line.strip()
        if stripped:
            return stripped
    return ""


def _score_section(query: str, expanded_query: str, query_tokens: list[str], section: dict) -> float:
    title_text = " ".join(
        str(section.get(key, "") or "")
        for key in ("title", "section_path", "document", "doc_type")
    )
    content = str(section.get("content", "") or "")
    title_tokens = _tokenize(title_text)
    content_tokens = _tokenize(content)

    title_overlap = _overlap_score(query_tokens, title_tokens)
    content_overlap = _overlap_score(query_tokens, content_tokens)
    phrase = max(
        _phrase_score(query, f"{title_text}\n{content}"),
        0.75 * _phrase_score(expanded_query, f"{title_text}\n{content}"),
    )
    legal = _legal_reference_score(query, title_text, content)
    definition = _definition_score(query, title_text, content)
    doc_type_boost = 0.08 if section.get("doc_type") == "legal" and _looks_legal_query(query) else 0.0
    short_heading_penalty = 0.12 if len(_tokenize(content)) < 8 and not definition else 0.0

    score = (
        0.38 * title_overlap
        + 0.34 * content_overlap
        + 0.14 * phrase
        + 0.10 * legal
        + 0.16 * definition
        + doc_type_boost
        - short_heading_penalty
    )
    return max(0.0, score)


def _compact_section(text: str) -> str:
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if len(text) <= MAX_LOCAL_SECTION_CHARS:
        return text
    head = MAX_LOCAL_SECTION_CHARS // 2
    tail = MAX_LOCAL_SECTION_CHARS - head
    return f"{text[:head]}\n...\n{text[-tail:]}"


# =============================================================================
# Manifest / file helpers
# =============================================================================

def _iter_uploadable_pdfs() -> list[Path]:
    if not LANDING_DIR.exists():
        return []
    return sorted(path for path in LANDING_DIR.rglob("*.pdf") if path.is_file())


def _iter_standardized_markdown() -> list[Path]:
    if not STANDARDIZED_DIR.exists():
        return []
    return sorted(path for path in STANDARDIZED_DIR.rglob("*.md") if path.is_file())


def _load_manifest() -> dict:
    if not PAGEINDEX_MANIFEST_PATH.exists():
        return {"version": 1, "api_documents": {}, "updated_at": None}
    try:
        manifest = json.loads(PAGEINDEX_MANIFEST_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        manifest = {"version": 1, "api_documents": {}, "updated_at": None}
    manifest.setdefault("api_documents", {})
    return manifest


def _save_manifest(manifest: dict) -> None:
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    PAGEINDEX_MANIFEST_PATH.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _load_local_index() -> dict:
    if not PAGEINDEX_LOCAL_INDEX_PATH.exists():
        return {"documents": [], "sections": []}
    try:
        return json.loads(PAGEINDEX_LOCAL_INDEX_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {"documents": [], "sections": []}


def _save_local_index(index: dict) -> None:
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    PAGEINDEX_LOCAL_INDEX_PATH.write_text(
        json.dumps(index, ensure_ascii=False),
        encoding="utf-8",
    )


def _standardized_fingerprints() -> dict[str, str]:
    return {
        _relative_path(path): _file_fingerprint(path)
        for path in _iter_standardized_markdown()
    }


def _file_fingerprint(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT_DIR)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def _normalize_for_dedupe(text: str) -> str:
    without_metadata = re.sub(
        r"(?ms)^# .+?\n\n(?:\*\*.*?\*\*:.*?\n)+\n?---\s*",
        "",
        text,
        count=1,
    )
    return re.sub(r"\s+", " ", without_metadata).strip()


def _section_result_key(section: dict) -> str:
    content = _normalize_for_dedupe(str(section.get("content", "") or ""))
    title = _normalize_text(str(section.get("title", "") or ""))
    compact = re.sub(r"\s+", " ", content[:1200]).strip()
    return hashlib.sha256(f"{title}\n{compact}".encode("utf-8")).hexdigest()


def _refresh_document_status(api_key: str, record: dict, wait: bool) -> dict:
    doc_id = record.get("doc_id")
    if not doc_id:
        return record

    deadline = time.monotonic() + UPLOAD_TIMEOUT
    while True:
        tree = _get_tree(api_key, str(doc_id), summary=False)
        record["status"] = tree.get("status", record.get("status"))
        record["retrieval_ready"] = tree.get("retrieval_ready")
        record["checked_at"] = _now_iso()
        if not wait or record.get("status") in {"completed", "failed"}:
            return record
        if time.monotonic() >= deadline:
            return record
        time.sleep(UPLOAD_POLL_INTERVAL)


def _list_remote_documents_by_name(api_key: str) -> dict[str, dict]:
    try:
        data = _list_documents(api_key, limit=100, offset=0)
    except Exception as exc:
        print(f"[WARN] Cannot list PageIndex documents: {exc}")
        return {}
    return {
        str(doc.get("name") or doc.get("file_name") or ""): doc
        for doc in data.get("documents", [])
        if doc.get("name") or doc.get("file_name")
    }


def _list_completed_remote_doc_ids(api_key: str) -> list[str]:
    try:
        data = _list_documents(api_key, limit=100, offset=0)
    except Exception as exc:
        print(f"[WARN] Cannot list PageIndex documents: {exc}")
        return []
    return [
        str(doc.get("id"))
        for doc in data.get("documents", [])
        if doc.get("id") and doc.get("status") == "completed"
    ]


def _manifest_record_from_remote(pdf_path: Path, fingerprint: str, remote_doc: dict) -> dict:
    return {
        "doc_id": remote_doc.get("id") or remote_doc.get("doc_id"),
        "status": remote_doc.get("status", "unknown"),
        "file_name": pdf_path.name,
        "path": _relative_path(pdf_path),
        "sha256": fingerprint,
        "remote_name": remote_doc.get("name"),
        "page_num": remote_doc.get("pageNum"),
        "created_at": remote_doc.get("createdAt"),
        "matched_existing_remote": True,
        "updated_at": _now_iso(),
    }


# =============================================================================
# Scoring helpers
# =============================================================================

def _tokenize(text: str) -> list[str]:
    normalized = _normalize_text(text)
    tokens = re.findall(r"[a-z0-9]+", normalized)
    return [token for token in tokens if len(token) > 1 or token.isdigit()]


def _expand_query(query: str) -> str:
    normalized = _normalize_text(query)
    additions: list[str] = []

    if "hinh phat" in normalized:
        additions.extend(["xu phat", "phat tien", "che tai", "trach nhiem hinh su"])
    if "xu phat" in normalized or "muc phat" in normalized:
        additions.extend(["phat tien", "vi pham hanh chinh", "che tai"])
    if "su dung" in normalized and "ma tuy" in normalized:
        additions.extend([
            "su dung trai phep chat ma tuy",
            "nguoi su dung trai phep chat ma tuy",
            "to chuc su dung trai phep chat ma tuy",
        ])
    if "cai nghien" in normalized:
        additions.extend(["co so cai nghien", "quan ly sau cai nghien", "ho tro cai nghien"])
    if "danh muc" in normalized or "chat cam" in normalized:
        additions.extend(["danh muc chat ma tuy", "chat gay nghien", "chat huong than", "tien chat"])
    if _looks_definition_query(query):
        additions.extend([
            "giai thich tu ngu",
            "dinh nghia",
            "duoc hieu nhu sau",
            "chat ma tuy la chat gay nghien chat huong than",
        ])

    if not additions:
        return query
    return f"{query} {' '.join(additions)}"


def _normalize_text(text: str) -> str:
    lowered = (text or "").lower()
    stripped = "".join(
        char
        for char in unicodedata.normalize("NFD", lowered)
        if unicodedata.category(char) != "Mn"
    )
    return stripped.replace("đ", "d")


def _overlap_score(query_tokens: list[str], document_tokens: list[str]) -> float:
    if not query_tokens or not document_tokens:
        return 0.0
    query_set = set(query_tokens)
    document_set = set(document_tokens)
    recall = len(query_set & document_set) / max(1, len(query_set))
    precision = len(query_set & document_set) / max(1, min(len(document_set), 80))
    if recall + precision == 0:
        return 0.0
    return 2 * recall * precision / (recall + precision)


def _phrase_score(query: str, text: str) -> float:
    normalized_query = _normalize_text(query)
    normalized_text = _normalize_text(text)
    if not normalized_query or not normalized_text:
        return 0.0
    if normalized_query in normalized_text:
        return 1.0

    query_tokens = [token for token in _tokenize(query) if len(token) > 2]
    if len(query_tokens) < 2:
        return 0.0
    bigrams = [" ".join(query_tokens[idx:idx + 2]) for idx in range(len(query_tokens) - 1)]
    matches = sum(1 for bigram in bigrams if bigram in normalized_text)
    return matches / max(1, len(bigrams))


def _legal_reference_score(query: str, title: str, content: str) -> float:
    normalized_query = _normalize_text(query)
    normalized_text = _normalize_text(f"{title}\n{content}")
    score = 0.0

    article_matches = re.findall(r"(?:dieu|đieu)\s*(\d+[a-z]?)", normalized_query)
    for article in article_matches:
        if re.search(rf"\bdieu\s+{re.escape(article)}\b", normalized_text):
            score += 0.5

    for term in ("hinh phat", "muc phat", "xu phat", "cai nghien", "chat ma tuy", "to chuc su dung"):
        if term in normalized_query and term in normalized_text:
            score += 0.15

    return min(1.0, score)


def _definition_score(query: str, title: str, content: str) -> float:
    if not _looks_definition_query(query):
        return 0.0

    normalized_query = _normalize_text(query)
    normalized_text = _normalize_text(f"{title}\n{content}")
    score = 0.0

    if "giai thich tu ngu" in normalized_text:
        score += 0.35
    if "duoc hieu nhu sau" in normalized_text:
        score += 0.25
    if "chat ma tuy la" in normalized_text and "ma tuy" in normalized_query:
        score += 0.65
    if "chat gay nghien la" in normalized_text and "gay nghien" in normalized_query:
        score += 0.45
    if "chat huong than la" in normalized_text and "huong than" in normalized_query:
        score += 0.45

    return min(1.0, score)


def _looks_definition_query(query: str) -> bool:
    normalized = _normalize_text(query)
    return any(
        phrase in normalized
        for phrase in (
            "la gi",
            "dinh nghia",
            "duoc hieu",
            "khai niem",
            "nghia la",
        )
    )


def _looks_legal_query(query: str) -> bool:
    normalized = _normalize_text(query)
    return any(
        term in normalized
        for term in (
            "dieu",
            "luat",
            "nghi dinh",
            "hinh phat",
            "muc phat",
            "xu phat",
            "quy dinh",
            "toi",
        )
    )


def _rank_score(outer_rank: int, inner_rank: int) -> float:
    return 1.0 / (outer_rank + (inner_rank - 1) * 0.25)


def _external_api_disabled_for_tests() -> bool:
    return bool(os.getenv("PYTEST_CURRENT_TEST")) and os.getenv("PAGEINDEX_ALLOW_API_IN_TESTS") != "1"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    load_dotenv()
    if not os.getenv("PAGEINDEX_API_KEY"):
        print("PAGEINDEX_API_KEY is missing; using local structural fallback.")
    else:
        print("Uploading PageIndex documents...")
        records = upload_documents(wait=False)
        print(f"Prepared {len(records)} PageIndex records.")

    print("\nTest query:")
    for result in pageindex_search("hình phạt sử dụng ma túy", top_k=3):
        print(f"[{result['score']:.3f}] {result['content'][:120]}...")

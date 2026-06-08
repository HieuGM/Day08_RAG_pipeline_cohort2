"""
Task 4 - Legal-aware semantic chunking + OpenAI embeddings + Qdrant indexing.

Design:
    1. Load markdown documents from data/standardized/.
    2. Remove duplicate fallback files such as *.congbao.md when the main file exists.
    3. Pre-split legal documents on structural markers (Chuong, Dieu, Mau so, Phu luc).
    4. Use SemanticChunker-style OpenAI semantic breakpoints inside long
       sections when run_pipeline() is called with OPENAI_API_KEY available.
    5. Enforce a hard character cap with recursive splitting so chunks remain
       useful for citation and compatible with the test suite.
    6. Embed chunks with OpenAI text-embedding-3-small and index them in a
       persistent local Qdrant collection.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from pathlib import Path
from typing import Any

STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"
INDEX_DIR = Path(__file__).parent.parent / "data" / "index"
QDRANT_PATH = INDEX_DIR / "qdrant"
CHUNKS_JSONL = INDEX_DIR / "task4_chunks.jsonl"
EMBEDDED_JSONL = INDEX_DIR / "task4_embedded_chunks.jsonl"


# =============================================================================
# CONFIGURATION
# =============================================================================

# Semantic-hybrid chunking keeps legal boundaries first, then uses embeddings to
# find topic shifts inside long sections. The hard cap preserves compact evidence
# chunks for citation and keeps downstream retrieval predictable.
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 160
CHUNKING_METHOD = "semantic_hybrid"
SEMANTIC_BREAKPOINT_TYPE = "percentile"
SEMANTIC_BREAKPOINT_AMOUNT = 92
SEMANTIC_MIN_CHARS = int(CHUNK_SIZE * 1.35)
SEMANTIC_SENTENCE_WINDOW = 3
SEMANTIC_EMBED_BATCH_SIZE = 12

# OpenAI docs: text-embedding-3-small defaults to 1536 dimensions.
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM = 1536
EMBED_BATCH_SIZE = 24
OPENAI_MAX_RETRIES = 8
OPENAI_RETRY_FLOOR_SECONDS = 1.25

# Qdrant local mode gives a real persistent vector index without requiring a
# Docker service during class demos. The same payload schema can be reused for
# server/cloud Qdrant later.
VECTOR_STORE = "qdrant"
COLLECTION_NAME = "drug_law_docs_openai_small"


def load_documents() -> list[dict]:
    """
    Read markdown files from data/standardized/.

    Returns:
        List of {'content': str, 'metadata': {'source': str, 'type': str, ...}}
    """
    documents: list[dict] = []
    seen_hashes: set[str] = set()

    for md_file in sorted(STANDARDIZED_DIR.rglob("*.md")):
        if md_file.name.startswith(".") or _is_duplicate_fallback(md_file):
            continue

        raw_content = md_file.read_text(encoding="utf-8").strip()
        if not raw_content:
            continue

        body = _strip_conversion_header(raw_content)
        content_hash = _content_hash(body)
        if content_hash in seen_hashes:
            continue
        seen_hashes.add(content_hash)

        relative_path = md_file.relative_to(STANDARDIZED_DIR)
        doc_type = "legal" if "legal" in relative_path.parts else "news"
        title = _extract_title(raw_content) or md_file.stem

        documents.append(
            {
                "content": body,
                "metadata": {
                    "source": md_file.name,
                    "source_path": relative_path.as_posix(),
                    "type": doc_type,
                    "title": title,
                    "url": _extract_header_field(raw_content, "Url"),
                    "published_at": _extract_header_field(raw_content, "Published At"),
                    "content_hash": content_hash,
                },
            }
        )

    return documents


def chunk_documents(documents: list[dict], use_semantic: bool = False) -> list[dict]:
    """
    Chunk documents with legal-aware structural splitting and optional semantic
    chunking for long sections.

    Returns:
        List of {'content': str, 'metadata': dict}
    """
    chunks: list[dict] = []
    semantic_splitter = _build_semantic_splitter() if use_semantic else None

    for doc in documents:
        metadata = doc.get("metadata", {})
        sections = _split_structural_sections(doc["content"], metadata)
        doc_chunk_index = 0

        for section_index, section in enumerate(sections):
            section_text = section["text"].strip()
            if not section_text:
                continue

            semantic_parts = _semantic_split(section_text, semantic_splitter)
            for semantic_part_index, semantic_part in enumerate(semantic_parts):
                for capped_part in _hard_cap_split(semantic_part):
                    chunk_text = capped_part.strip()
                    if len(chunk_text) < 40:
                        continue

                    chunk_metadata = {
                        **metadata,
                        "section": section["section"],
                        "section_index": section_index,
                        "semantic_part_index": semantic_part_index,
                        "chunk_index": doc_chunk_index,
                        "parent_id": _stable_id(
                            metadata.get("source_path", metadata.get("source", "")),
                            section["section"],
                        ),
                    }
                    chunk_metadata["chunk_id"] = _stable_id(
                        chunk_metadata.get("source_path", ""),
                        str(doc_chunk_index),
                        chunk_text[:120],
                    )
                    chunks.append({"content": chunk_text, "metadata": chunk_metadata})
                    doc_chunk_index += 1

    return chunks


def embed_chunks(chunks: list[dict]) -> list[dict]:
    """
    Embed chunks with OpenAI text-embedding-3-small.

    Returns:
        Each chunk dict with an added 'embedding': list[float].
    """
    if not chunks:
        return []

    _load_dotenv_if_available()
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY is missing. Create .env from .env.example or set the "
            "environment variable before running Task 4 embedding/indexing."
        )

    from openai import OpenAI

    client = OpenAI()
    embedded_chunks: list[dict] = []

    for batch_start in range(0, len(chunks), EMBED_BATCH_SIZE):
        batch = chunks[batch_start : batch_start + EMBED_BATCH_SIZE]
        response = _call_openai_with_retries(
            lambda: client.embeddings.create(
                model=EMBEDDING_MODEL,
                input=[_embedding_input(chunk) for chunk in batch],
                encoding_format="float",
            ),
            operation=f"embedding batch {batch_start // EMBED_BATCH_SIZE + 1}",
        )
        vectors = [item.embedding for item in response.data]

        for chunk, vector in zip(batch, vectors):
            item = {
                "content": chunk["content"],
                "metadata": dict(chunk.get("metadata", {})),
                "embedding": vector,
            }
            embedded_chunks.append(item)

    _write_jsonl(EMBEDDED_JSONL, embedded_chunks)
    return embedded_chunks


def index_to_vectorstore(chunks: list[dict]):
    """
    Store embedded chunks in a persistent local Qdrant collection.
    """
    if not chunks:
        return None
    if "embedding" not in chunks[0]:
        raise ValueError("Chunks must be embedded before indexing.")

    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, PointStruct, VectorParams

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    client = QdrantClient(path=str(QDRANT_PATH))

    if client.collection_exists(COLLECTION_NAME):
        client.delete_collection(COLLECTION_NAME)

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
    )

    points = []
    for point_id, chunk in enumerate(chunks):
        metadata = dict(chunk.get("metadata", {}))
        points.append(
            PointStruct(
                id=point_id,
                vector=chunk["embedding"],
                payload={
                    "content": chunk["content"],
                    "source": metadata.get("source", ""),
                    "source_path": metadata.get("source_path", ""),
                    "doc_type": metadata.get("type", ""),
                    "title": metadata.get("title", ""),
                    "section": metadata.get("section", ""),
                    "url": metadata.get("url", ""),
                    "chunk_index": metadata.get("chunk_index", 0),
                    "metadata": metadata,
                },
            )
        )

    client.upsert(collection_name=COLLECTION_NAME, points=points)
    return client


def run_pipeline():
    """Run load -> chunk -> embed -> index."""
    print("=" * 50)
    print("Task 4: Chunking & Indexing")
    print(f"  Chunking: {CHUNKING_METHOD} (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
    print(f"  Semantic threshold: {SEMANTIC_BREAKPOINT_TYPE}={SEMANTIC_BREAKPOINT_AMOUNT}")
    print(f"  Embedding: {EMBEDDING_MODEL} (dim={EMBEDDING_DIM})")
    print(f"  Vector Store: {VECTOR_STORE} local path={QDRANT_PATH}")
    print("=" * 50)

    docs = load_documents()
    print(f"\n[OK] Loaded {len(docs)} documents")

    chunks = chunk_documents(docs, use_semantic=True)
    _write_jsonl(CHUNKS_JSONL, chunks)
    print(f"[OK] Created {len(chunks)} chunks")
    print(f"[OK] Wrote chunk manifest: {CHUNKS_JSONL}")

    embedded = embed_chunks(chunks)
    print(f"[OK] Embedded {len(embedded)} chunks")

    index_to_vectorstore(embedded)
    print(f"[OK] Indexed to Qdrant collection: {COLLECTION_NAME}")


def _is_duplicate_fallback(md_file: Path) -> bool:
    """Skip generated fallback copies when their main converted file exists."""
    if not md_file.name.endswith(".congbao.md"):
        return False
    main_name = md_file.name.replace(".congbao.md", ".md")
    return (md_file.parent / main_name).exists()


def _strip_conversion_header(content: str) -> str:
    lines = content.splitlines()
    for index, line in enumerate(lines[:12]):
        if line.strip() == "---":
            return "\n".join(lines[index + 1 :]).strip()
    return content.strip()


def _extract_title(content: str) -> str:
    for line in content.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            return stripped.lstrip("#").strip()
    return ""


def _extract_header_field(content: str, field_name: str) -> str:
    pattern = re.compile(rf"^\*\*{re.escape(field_name)}:\*\*\s*(.+?)\s*$", re.IGNORECASE)
    for line in content.splitlines()[:20]:
        match = pattern.match(line.strip())
        if match:
            return match.group(1).strip()
    return ""


def _content_hash(text: str) -> str:
    normalized = re.sub(r"\s+", " ", text).strip().lower()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _split_structural_sections(text: str, metadata: dict) -> list[dict]:
    doc_type = metadata.get("type", "")
    if doc_type == "legal":
        return _split_legal_sections(text, metadata)
    return _split_news_sections(text, metadata)


LEGAL_SECTION_RE = re.compile(
    r"^(Chương\s+[IVXLCDM0-9]+|Điều\s+\d+[a-zA-Z]?\.|Mẫu số\s+\d+[A-Z]?|Phụ lục\b|DANH MỤC\b)",
    re.IGNORECASE,
)


def _split_legal_sections(text: str, metadata: dict) -> list[dict]:
    sections: list[dict] = []
    current_lines: list[str] = []
    current_section = metadata.get("title", "legal-document")

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if LEGAL_SECTION_RE.match(line) and current_lines:
            sections.append({"section": current_section, "text": "\n".join(current_lines).strip()})
            current_lines = []
            current_section = line[:140]
        elif LEGAL_SECTION_RE.match(line):
            current_section = line[:140]
        current_lines.append(raw_line)

    if current_lines:
        sections.append({"section": current_section, "text": "\n".join(current_lines).strip()})

    return _merge_tiny_sections(sections)


def _split_news_sections(text: str, metadata: dict) -> list[dict]:
    sections: list[dict] = []
    current_lines: list[str] = []
    current_section = metadata.get("title", "news-article")

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if line.startswith("## ") and current_lines:
            sections.append({"section": current_section, "text": "\n".join(current_lines).strip()})
            current_lines = []
            current_section = line.lstrip("#").strip()[:140]
        elif line.startswith("# "):
            current_section = line.lstrip("#").strip()[:140]
        current_lines.append(raw_line)

    if current_lines:
        sections.append({"section": current_section, "text": "\n".join(current_lines).strip()})

    return _merge_tiny_sections(sections)


def _merge_tiny_sections(sections: list[dict]) -> list[dict]:
    merged: list[dict] = []
    for section in sections:
        if merged and len(section["text"]) < 220:
            merged[-1]["text"] = f'{merged[-1]["text"]}\n\n{section["text"]}'.strip()
            merged[-1]["section"] = f'{merged[-1]["section"]} / {section["section"]}'[:180]
        else:
            merged.append(section)
    return merged


def _build_semantic_splitter():
    _load_dotenv_if_available()
    if not os.getenv("OPENAI_API_KEY"):
        return None

    try:
        from openai import OpenAI

        return _OpenAISemanticSplitter(OpenAI())
    except Exception as exc:
        print(f"[WARN] Semantic splitter unavailable, falling back to recursive splitting: {exc}")
        return None


class _OpenAISemanticSplitter:
    """SemanticChunker-style splitter with explicit OpenAI retry control."""

    def __init__(self, client):
        self.client = client

    def split_text(self, text: str) -> list[str]:
        units = _semantic_units(text)
        if len(units) < 4:
            return [text]

        windows = []
        half_window = max((SEMANTIC_SENTENCE_WINDOW - 1) // 2, 0)
        for index in range(len(units)):
            start = max(0, index - half_window)
            end = min(len(units), index + half_window + 1)
            windows.append(" ".join(units[start:end]))

        embeddings = self._embed_windows(windows)
        distances = [
            1.0 - _cosine_similarity(embeddings[index], embeddings[index + 1])
            for index in range(len(embeddings) - 1)
        ]
        if not distances:
            return [text]

        threshold = _percentile(distances, SEMANTIC_BREAKPOINT_AMOUNT)
        split_after_indices = {
            index
            for index, distance in enumerate(distances)
            if distance >= threshold and distance > 0
        }

        parts: list[str] = []
        current_units: list[str] = []
        for index, unit in enumerate(units):
            current_units.append(unit)
            if index in split_after_indices:
                parts.append(" ".join(current_units).strip())
                current_units = []
        if current_units:
            parts.append(" ".join(current_units).strip())

        return _merge_tiny_text_parts([part for part in parts if part])

    def _embed_windows(self, windows: list[str]) -> list[list[float]]:
        embeddings: list[list[float]] = []
        for batch_start in range(0, len(windows), SEMANTIC_EMBED_BATCH_SIZE):
            batch = windows[batch_start : batch_start + SEMANTIC_EMBED_BATCH_SIZE]
            response = _call_openai_with_retries(
                lambda: self.client.embeddings.create(
                    model=EMBEDDING_MODEL,
                    input=batch,
                    encoding_format="float",
                ),
                operation=f"semantic chunking window batch {batch_start // SEMANTIC_EMBED_BATCH_SIZE + 1}",
            )
            embeddings.extend(item.embedding for item in response.data)
        return embeddings


def _semantic_split(text: str, semantic_splitter) -> list[str]:
    if semantic_splitter is None or len(text) < SEMANTIC_MIN_CHARS:
        return [text]
    for attempt in range(OPENAI_MAX_RETRIES + 1):
        try:
            parts = [part.strip() for part in semantic_splitter.split_text(text) if part.strip()]
            return parts or [text]
        except Exception as exc:
            if not _is_openai_rate_limit_error(exc) or attempt >= OPENAI_MAX_RETRIES:
                print(f"[WARN] Semantic split failed, falling back to structural section: {exc}")
                return [text]
            _sleep_before_retry(exc, attempt, "semantic chunking")
    return [text]


def _hard_cap_split(text: str) -> list[str]:
    if len(text) <= CHUNK_SIZE:
        return [text]

    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=[
                "\nĐiều ",
                "\nChương ",
                "\nMẫu số ",
                "\n\n",
                "\n",
                ". ",
                "; ",
                ", ",
                " ",
                "",
            ],
        )
        return splitter.split_text(text)
    except Exception:
        return _manual_sliding_split(text)


def _manual_sliding_split(text: str) -> list[str]:
    chunks: list[str] = []
    start = 0
    step = max(CHUNK_SIZE - CHUNK_OVERLAP, 1)
    while start < len(text):
        end = min(start + CHUNK_SIZE, len(text))
        chunks.append(text[start:end])
        if end == len(text):
            break
        start += step
    return chunks


def _semantic_units(text: str) -> list[str]:
    rough_units = re.split(r"(?<=[.!?])\s+|\n{2,}", text)
    units: list[str] = []

    for rough_unit in rough_units:
        unit = re.sub(r"\s+", " ", rough_unit).strip()
        if not unit:
            continue
        if len(unit) <= 700:
            units.append(unit)
            continue

        # Long legal paragraphs often use semicolons and numbered clauses.
        sub_units = re.split(r"(?<=;)\s+|(?<=:)\s+|(?=\b\d+\.\s+)|(?=\b[a-zđ]\)\s+)", unit)
        for sub_unit in sub_units:
            sub_unit = re.sub(r"\s+", " ", sub_unit).strip()
            if sub_unit:
                units.append(sub_unit)

    return units


def _merge_tiny_text_parts(parts: list[str]) -> list[str]:
    merged: list[str] = []
    for part in parts:
        if merged and len(part) < 180:
            merged[-1] = f"{merged[-1]} {part}".strip()
        else:
            merged.append(part)
    return merged


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    rank = (len(sorted_values) - 1) * percentile / 100.0
    lower = int(rank)
    upper = min(lower + 1, len(sorted_values) - 1)
    fraction = rank - lower
    return sorted_values[lower] * (1 - fraction) + sorted_values[upper] * fraction


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    dot_product = 0.0
    left_norm = 0.0
    right_norm = 0.0
    for left_value, right_value in zip(left, right):
        dot_product += left_value * right_value
        left_norm += left_value * left_value
        right_norm += right_value * right_value
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return dot_product / ((left_norm ** 0.5) * (right_norm ** 0.5))


def _embedding_input(chunk: dict) -> str:
    metadata = chunk.get("metadata", {})
    return "\n".join(
        [
            f"Document: {metadata.get('title', '')}",
            f"Type: {metadata.get('type', '')}",
            f"Section: {metadata.get('section', '')}",
            f"Source: {metadata.get('source', '')}",
            "Content:",
            chunk["content"],
        ]
    )


def _call_openai_with_retries(callable_obj, operation: str):
    for attempt in range(OPENAI_MAX_RETRIES + 1):
        try:
            return callable_obj()
        except Exception as exc:
            if not _is_openai_rate_limit_error(exc) or attempt >= OPENAI_MAX_RETRIES:
                raise
            _sleep_before_retry(exc, attempt, operation)

    raise RuntimeError(f"OpenAI call failed after retries: {operation}")


def _is_openai_rate_limit_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return (
        "rate_limit" in message
        or "rate limit" in message
        or "429" in message
        or "tokens per min" in message
        or "tpm" in message
    )


def _sleep_before_retry(exc: Exception, attempt: int, operation: str) -> None:
    wait_seconds = _retry_after_seconds(str(exc))
    exponential_seconds = OPENAI_RETRY_FLOOR_SECONDS * (2 ** attempt)
    wait_seconds = max(wait_seconds, min(exponential_seconds, 20.0))
    print(f"[WAIT] OpenAI rate limit during {operation}; retrying in {wait_seconds:.2f}s")
    time.sleep(wait_seconds)


def _retry_after_seconds(message: str) -> float:
    millisecond_match = re.search(r"try again in\s+([0-9.]+)\s*ms", message, re.IGNORECASE)
    if millisecond_match:
        return max(float(millisecond_match.group(1)) / 1000.0, OPENAI_RETRY_FLOOR_SECONDS)

    second_match = re.search(r"try again in\s+([0-9.]+)\s*s", message, re.IGNORECASE)
    if second_match:
        return max(float(second_match.group(1)), OPENAI_RETRY_FLOOR_SECONDS)

    return OPENAI_RETRY_FLOOR_SECONDS


def _stable_id(*parts: str) -> str:
    joined = "||".join(part or "" for part in parts)
    return hashlib.sha1(joined.encode("utf-8")).hexdigest()


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _load_dotenv_if_available() -> None:
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except Exception:
        return


if __name__ == "__main__":
    run_pipeline()

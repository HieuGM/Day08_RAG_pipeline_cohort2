"""
Task 5 - Semantic search over the Task 4 OpenAI/Qdrant index.

Primary path:
    query -> OpenAI text-embedding-3-small -> Qdrant cosine search

Fallback:
    If local Qdrant is unavailable or locked, scan
    data/index/task4_embedded_chunks.jsonl and compute cosine similarity.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import unicodedata
from pathlib import Path

try:
    from .task4_chunking_indexing import (
        COLLECTION_NAME,
        EMBEDDED_JSONL,
        EMBEDDING_MODEL,
        QDRANT_PATH,
        _call_openai_with_retries,
        _cosine_similarity,
        _load_dotenv_if_available,
    )
except ImportError:  # Allows: python src/task5_semantic_search.py
    from task4_chunking_indexing import (  # type: ignore
        COLLECTION_NAME,
        EMBEDDED_JSONL,
        EMBEDDING_MODEL,
        QDRANT_PATH,
        _call_openai_with_retries,
        _cosine_similarity,
        _load_dotenv_if_available,
    )

QUERY_CACHE_PATH = Path(__file__).parent.parent / "data" / "index" / "query_embedding_cache.json"
SEMANTIC_QUERY_GUARD = os.getenv("SEMANTIC_QUERY_GUARD", "1") != "0"
_CORPUS_SIGNAL_CACHE: tuple[str, set[str]] | None = None

_QUERY_STOPWORDS = {
    "bao",
    "bao_nhieu",
    "bo",
    "cho",
    "cong",
    "cua",
    "gia",
    "gi",
    "hom",
    "la",
    "nay",
    "ngon",
    "nau",
    "nhieu",
    "pho",
    "theo",
    "thuc",
    "trong",
}

_DOMAIN_PHRASES = (
    "ma tuy",
    "chat ma tuy",
    "phong chong ma tuy",
    "tang tru",
    "trai phep",
    "cai nghien",
    "nghien ma tuy",
    "chat cam",
    "tien chat",
    "nghi dinh 105",
    "nghi dinh 28",
    "luat phong chong ma tuy",
)


def semantic_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Dense semantic search using the same OpenAI embedding model as Task 4.

    Returns:
        List of {'content': str, 'score': float, 'metadata': dict}
        sorted by score descending.
    """
    query = (query or "").strip()
    if not query or top_k <= 0:
        return []

    if SEMANTIC_QUERY_GUARD and not _query_has_corpus_support(query):
        return []

    try:
        query_vector = embed_query(query)
    except Exception as exc:
        print(f"[WARN] Cannot embed semantic query: {exc}")
        return []

    results = _search_qdrant(query_vector, top_k)
    if not results:
        results = _search_jsonl_fallback(query_vector, top_k)

    results = sorted(results, key=lambda item: item["score"], reverse=True)
    return results[:top_k]


def embed_query(query: str) -> list[float]:
    """Embed and cache a query vector."""
    _load_dotenv_if_available()
    cache_key = _query_cache_key(query)
    cache = _load_query_cache()
    if cache_key in cache:
        return cache[cache_key]

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is missing and query embedding is not cached.")

    from openai import OpenAI

    client = OpenAI()
    response = _call_openai_with_retries(
        lambda: client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=_query_embedding_input(query),
            encoding_format="float",
        ),
        operation="semantic query embedding",
    )
    vector = response.data[0].embedding
    cache[cache_key] = vector
    _save_query_cache(cache)
    return vector


def _search_qdrant(query_vector: list[float], top_k: int) -> list[dict]:
    try:
        from qdrant_client import QdrantClient

        client = QdrantClient(path=str(QDRANT_PATH))
        if not client.collection_exists(COLLECTION_NAME):
            return []

        try:
            response = client.query_points(
                collection_name=COLLECTION_NAME,
                query=query_vector,
                limit=top_k,
                with_payload=True,
            )
            points = response.points
        except AttributeError:
            points = client.search(
                collection_name=COLLECTION_NAME,
                query_vector=query_vector,
                limit=top_k,
                with_payload=True,
            )

        return [_format_qdrant_point(point) for point in points]
    except Exception as exc:
        print(f"[WARN] Qdrant semantic search unavailable, using JSONL fallback: {exc}")
        return []


def _search_jsonl_fallback(query_vector: list[float], top_k: int) -> list[dict]:
    if not EMBEDDED_JSONL.exists():
        print(f"[WARN] Missing embedded chunks file: {EMBEDDED_JSONL}. Run Task 4 first.")
        return []

    scored: list[dict] = []
    with EMBEDDED_JSONL.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            embedding = row.get("embedding")
            if not embedding:
                continue
            score = _cosine_similarity(query_vector, embedding)
            scored.append(
                {
                    "content": row.get("content", ""),
                    "score": float(score),
                    "metadata": row.get("metadata", {}),
                }
            )

    scored.sort(key=lambda item: item["score"], reverse=True)
    return scored[:top_k]


def _format_qdrant_point(point) -> dict:
    payload = point.payload or {}
    metadata = payload.get("metadata") or {
        "source": payload.get("source", ""),
        "source_path": payload.get("source_path", ""),
        "type": payload.get("doc_type", ""),
        "title": payload.get("title", ""),
        "section": payload.get("section", ""),
        "url": payload.get("url", ""),
        "chunk_index": payload.get("chunk_index", 0),
    }
    return {
        "content": payload.get("content", ""),
        "score": float(getattr(point, "score", 0.0) or 0.0),
        "metadata": metadata,
    }


def _query_embedding_input(query: str) -> str:
    return (
        "Search query for Vietnamese drug law, controlled substances, "
        f"and related Vietnamese news retrieval:\n{query}"
    )


def _query_cache_key(query: str) -> str:
    raw = f"{EMBEDDING_MODEL}\n{_query_embedding_input(query).strip().lower()}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _load_query_cache() -> dict[str, list[float]]:
    if not QUERY_CACHE_PATH.exists():
        return {}
    try:
        return json.loads(QUERY_CACHE_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _save_query_cache(cache: dict[str, list[float]]) -> None:
    QUERY_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    QUERY_CACHE_PATH.write_text(json.dumps(cache), encoding="utf-8")


def _query_has_corpus_support(query: str) -> bool:
    normalized = _normalize_signal_text(query)
    if any(phrase in normalized for phrase in _DOMAIN_PHRASES):
        return True

    query_terms = _signal_terms(query)
    if not query_terms:
        return False

    corpus_text, corpus_terms = _load_corpus_signal()
    if not corpus_terms:
        return True

    compact_query = " ".join(query_terms)
    if len(compact_query) >= 5 and compact_query in corpus_text:
        return True

    unique_terms = set(query_terms)
    hits = sum(1 for term in unique_terms if term in corpus_terms)
    required_hits = 1 if len(unique_terms) == 1 and len(next(iter(unique_terms))) >= 5 else 2
    return hits >= min(required_hits, len(unique_terms))


def _load_corpus_signal() -> tuple[str, set[str]]:
    global _CORPUS_SIGNAL_CACHE
    if _CORPUS_SIGNAL_CACHE is not None:
        return _CORPUS_SIGNAL_CACHE

    pieces: list[str] = []
    if EMBEDDED_JSONL.exists():
        with EMBEDDED_JSONL.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                metadata = row.get("metadata") or {}
                pieces.append(str(row.get("content", "") or ""))
                pieces.append(str(metadata.get("title", "") or ""))
                pieces.append(str(metadata.get("section", "") or ""))

    corpus_text = _normalize_signal_text(" ".join(pieces))
    corpus_terms = set(re.findall(r"[a-z0-9]+", corpus_text))
    _CORPUS_SIGNAL_CACHE = (corpus_text, corpus_terms)
    return _CORPUS_SIGNAL_CACHE


def _signal_terms(text: str) -> list[str]:
    normalized = _normalize_signal_text(text)
    terms = []
    for token in re.findall(r"[a-z0-9]+", normalized):
        if token in _QUERY_STOPWORDS:
            continue
        if len(token) < 3 and not token.isdigit():
            continue
        terms.append(token)
    return terms


def _normalize_signal_text(text: str) -> str:
    lowered = (text or "").lower().replace("ma tuý", "ma túy")
    stripped = "".join(
        char
        for char in unicodedata.normalize("NFD", lowered)
        if unicodedata.category(char) != "Mn"
    )
    return re.sub(r"\s+", " ", stripped.replace("đ", "d")).strip()


if __name__ == "__main__":
    results = semantic_search("hình phạt cho tội tàng trữ ma túy", top_k=5)
    for result in results:
        print(f"[{result['score']:.3f}] {result['content'][:120]}...")

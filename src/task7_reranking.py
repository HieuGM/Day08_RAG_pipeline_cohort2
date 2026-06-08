"""
Task 7 - Reranking module.

Production path:
    1. Jina Reranker v3 API, if JINA_API_KEY is configured.
    2. OpenAI GPT-5.5 listwise reranker, if OPENAI_API_KEY is configured.

Utility path:
    - RRF for hybrid dense+sparse fusion.
    - MMR for relevance/diversity selection.

The small offline scorer is intentionally only a safety fallback so tests and
offline demos still run when no reranker credential is available.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import time
import unicodedata
from pathlib import Path
from typing import Any, Optional

try:
    from .task4_chunking_indexing import _cosine_similarity, _load_dotenv_if_available
except ImportError:  # Allows: python src/task7_reranking.py
    from task4_chunking_indexing import _cosine_similarity, _load_dotenv_if_available  # type: ignore


# =============================================================================
# Configuration
# =============================================================================

DEFAULT_TOP_K = 5
RERANK_PROVIDER = os.getenv("RERANK_PROVIDER", "auto").strip().lower()

JINA_RERANK_ENDPOINT = "https://api.jina.ai/v1/rerank"
JINA_RERANK_MODEL = os.getenv("JINA_RERANK_MODEL", "jina-reranker-v3")

OPENAI_RERANK_MODEL = os.getenv("OPENAI_RERANK_MODEL", "gpt-5.5")
OPENAI_RERANK_FALLBACK_MODEL = os.getenv("OPENAI_RERANK_FALLBACK_MODEL", "gpt-4.1")
OPENAI_RERANK_REASONING_EFFORT = os.getenv("OPENAI_RERANK_REASONING_EFFORT", "low")

LOCAL_RERANK_MODEL = os.getenv("LOCAL_RERANK_MODEL", "BAAI/bge-reranker-v2-m3")

MAX_RERANK_CANDIDATES = int(os.getenv("MAX_RERANK_CANDIDATES", "50"))
MAX_DOCUMENT_CHARS = int(os.getenv("RERANK_MAX_DOCUMENT_CHARS", "8000"))
REQUEST_TIMEOUT = float(os.getenv("RERANK_REQUEST_TIMEOUT", "60"))
MAX_API_RETRIES = int(os.getenv("RERANK_MAX_API_RETRIES", "3"))

RERANK_CACHE_PATH = Path(__file__).parent.parent / "data" / "index" / "rerank_cache.json"


# =============================================================================
# Public rerankers
# =============================================================================

def rerank_cross_encoder(
    query: str,
    candidates: list[dict],
    top_k: int = DEFAULT_TOP_K,
    provider: Optional[str] = None,
) -> list[dict]:
    """
    Rerank candidates with the strongest configured reranker.

    Provider order in auto mode:
        1. Jina Reranker v3, a dedicated multilingual reranker.
        2. OpenAI GPT-5.5 listwise ranking, useful when only OPENAI_API_KEY exists.
        3. Offline deterministic fallback for tests/offline demos.
    """
    query = (query or "").strip()
    if not query or top_k <= 0 or not candidates:
        return []

    _load_dotenv_if_available()
    provider = (provider or os.getenv("RERANK_PROVIDER", RERANK_PROVIDER) or "auto").lower()
    candidates = candidates[:MAX_RERANK_CANDIDATES]

    if provider in {"heuristic", "offline", "fallback"}:
        return _rerank_offline(query, candidates, top_k)
    if provider in {"local", "local_cross_encoder"}:
        return _rerank_local_cross_encoder(query, candidates, top_k)
    if provider in {"jina", "jina_v3"}:
        return _rerank_jina(query, candidates, top_k, strict=True)
    if provider in {"openai", "llm", "gpt"}:
        return _rerank_openai_listwise(query, candidates, top_k, strict=True)

    if provider != "auto":
        raise ValueError(f"Unknown rerank provider: {provider}")

    if not _external_api_disabled_for_tests():
        if os.getenv("JINA_API_KEY"):
            try:
                return _rerank_jina(query, candidates, top_k, strict=False)
            except Exception as exc:
                print(f"[WARN] Jina rerank failed, trying OpenAI/listwise: {exc}")

        if os.getenv("OPENAI_API_KEY"):
            try:
                return _rerank_openai_listwise(query, candidates, top_k, strict=False)
            except Exception as exc:
                print(f"[WARN] OpenAI listwise rerank failed, using offline fallback: {exc}")

    return _rerank_offline(query, candidates, top_k)


def rerank_mmr(
    query_embedding: Optional[list[float]],
    candidates: list[dict],
    top_k: int = DEFAULT_TOP_K,
    lambda_param: float = 0.7,
) -> list[dict]:
    """
    Maximal Marginal Relevance.

    MMR = lambda * relevance(query, doc)
          - (1 - lambda) * max_similarity(doc, selected_docs)
    """
    if top_k <= 0 or not candidates:
        return []

    lambda_param = min(1.0, max(0.0, float(lambda_param)))
    relevance_scores = _mmr_relevance_scores(query_embedding, candidates)

    selected: list[int] = []
    remaining = set(range(len(candidates)))

    while remaining and len(selected) < top_k:
        best_idx: Optional[int] = None
        best_score = float("-inf")

        for idx in sorted(remaining):
            diversity_penalty = 0.0
            if selected:
                diversity_penalty = max(
                    _candidate_similarity(candidates[idx], candidates[sel_idx])
                    for sel_idx in selected
                )

            mmr_score = (
                lambda_param * relevance_scores[idx]
                - (1.0 - lambda_param) * diversity_penalty
            )
            if mmr_score > best_score:
                best_score = mmr_score
                best_idx = idx

        if best_idx is None:
            break

        selected.append(best_idx)
        remaining.remove(best_idx)

    results: list[dict] = []
    for rank, idx in enumerate(selected, start=1):
        item = _copy_candidate(candidates[idx])
        item["original_score"] = float(candidates[idx].get("score", 0.0) or 0.0)
        item["score"] = float(relevance_scores[idx])
        item["mmr_score"] = float(relevance_scores[idx])
        item["rerank_score"] = float(relevance_scores[idx])
        item["rerank_provider"] = "mmr"
        item["rank"] = rank
        results.append(item)
    return results


def rerank_rrf(
    ranked_lists: list[list[dict]],
    top_k: int = DEFAULT_TOP_K,
    k: int = 60,
) -> list[dict]:
    """
    Reciprocal Rank Fusion for merging outputs from multiple rankers.

    RRF(d) = sum_r 1 / (k + rank_r(d))
    """
    if top_k <= 0 or not ranked_lists:
        return []

    k = max(1, int(k))
    scores: dict[str, float] = {}
    canonical: dict[str, dict] = {}
    original_scores: dict[str, float] = {}

    for source_idx, ranked_list in enumerate(ranked_lists):
        seen_in_list: set[str] = set()
        for rank, item in enumerate(ranked_list, start=1):
            key = _candidate_key(item)
            if key in seen_in_list:
                continue
            seen_in_list.add(key)

            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
            old_original = original_scores.get(key, float("-inf"))
            item_original = float(item.get("score", 0.0) or 0.0)
            if key not in canonical or item_original > old_original:
                canonical[key] = _copy_candidate(item)
                original_scores[key] = item_original
                canonical[key]["best_source_ranker"] = source_idx

    sorted_keys = sorted(scores, key=lambda key: scores[key], reverse=True)
    results: list[dict] = []
    for rank, key in enumerate(sorted_keys[:top_k], start=1):
        item = _copy_candidate(canonical[key])
        item["original_score"] = float(original_scores.get(key, item.get("score", 0.0)) or 0.0)
        item["score"] = float(scores[key])
        item["rrf_score"] = float(scores[key])
        item["rerank_score"] = float(scores[key])
        item["rerank_provider"] = "rrf"
        item["rank"] = rank
        results.append(item)
    return results


# =============================================================================
# Main rerank interface
# =============================================================================

def rerank(
    query: str,
    candidates: list[dict],
    top_k: int = DEFAULT_TOP_K,
    method: str = "cross_encoder",
) -> list[dict]:
    """
    Unified reranking interface.

    Args:
        query: User query.
        candidates: Retrieval candidates with content, score, metadata.
        top_k: Number of final results.
        method: cross_encoder, jina, openai, local, mmr, rrf, or fallback.
    """
    method = (method or "cross_encoder").strip().lower()
    if method in {"cross_encoder", "best", "auto"}:
        return rerank_cross_encoder(query, candidates, top_k)
    if method in {"jina", "jina_v3"}:
        return rerank_cross_encoder(query, candidates, top_k, provider="jina")
    if method in {"openai", "llm", "gpt"}:
        return rerank_cross_encoder(query, candidates, top_k, provider="openai")
    if method in {"local", "local_cross_encoder"}:
        return rerank_cross_encoder(query, candidates, top_k, provider="local")
    if method in {"heuristic", "offline", "fallback"}:
        return _rerank_offline(query, candidates, top_k)
    if method == "mmr":
        return rerank_mmr(None, candidates, top_k)
    if method == "rrf":
        return rerank_rrf([candidates], top_k)
    raise ValueError(f"Unknown rerank method: {method}")


# =============================================================================
# Jina Reranker v3
# =============================================================================

def _rerank_jina(query: str, candidates: list[dict], top_k: int, strict: bool) -> list[dict]:
    api_key = os.getenv("JINA_API_KEY")
    if not api_key:
        raise RuntimeError("JINA_API_KEY is missing.")

    model = os.getenv("JINA_RERANK_MODEL", JINA_RERANK_MODEL)
    cache_key = _cache_key("jina", model, query, candidates, top_k)
    cached = _load_cached_rerank(cache_key)
    if cached is not None:
        return cached

    import requests

    payload = {
        "model": model,
        "query": query,
        "documents": [_candidate_text(candidate) for candidate in candidates],
        "top_n": min(top_k, len(candidates)),
        "return_documents": False,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    def call() -> dict:
        response = requests.post(
            JINA_RERANK_ENDPOINT,
            headers=headers,
            json=payload,
            timeout=REQUEST_TIMEOUT,
        )
        if response.status_code == 429:
            wait = _retry_after_seconds(response.headers.get("Retry-After"))
            time.sleep(wait)
        response.raise_for_status()
        return response.json()

    try:
        data = _call_with_retries(call, "Jina reranker")
        results = _results_from_ranked_indices(
            data.get("results", []),
            candidates,
            top_k,
            provider=f"jina:{model}",
        )
        _save_cached_rerank(cache_key, results)
        return results
    except Exception:
        if strict:
            raise
        raise


# =============================================================================
# OpenAI listwise reranker
# =============================================================================

def _rerank_openai_listwise(
    query: str,
    candidates: list[dict],
    top_k: int,
    strict: bool,
) -> list[dict]:
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is missing.")

    model = os.getenv("OPENAI_RERANK_MODEL", OPENAI_RERANK_MODEL)
    cache_key = _cache_key("openai", model, query, candidates, top_k)
    cached = _load_cached_rerank(cache_key)
    if cached is not None:
        return cached

    from openai import OpenAI

    client = OpenAI()
    payload = {
        "query": query,
        "top_k": min(top_k, len(candidates)),
        "candidates": [
            {
                "id": idx,
                "text": _candidate_text(candidate),
                "metadata": _compact_metadata(candidate.get("metadata", {})),
                "retrieval_score": float(candidate.get("score", 0.0) or 0.0),
            }
            for idx, candidate in enumerate(candidates)
        ],
    }

    def call_with_model(model_name: str) -> list[dict]:
        request: dict[str, Any] = {
            "model": model_name,
            "instructions": _openai_rerank_instructions(),
            "input": json.dumps(payload, ensure_ascii=False),
            "text": {"format": _openai_rerank_schema()},
            "temperature": 0,
            "max_output_tokens": max(256, min(4096, top_k * 120 + 180)),
            "store": False,
        }
        if _supports_reasoning(model_name):
            request["reasoning"] = {
                "effort": os.getenv(
                    "OPENAI_RERANK_REASONING_EFFORT",
                    OPENAI_RERANK_REASONING_EFFORT,
                )
            }
        response = client.responses.create(**request)
        data = json.loads(_extract_response_text(response))
        return _results_from_openai_payload(data, candidates, top_k, model_name)

    try:
        results = _call_with_retries(
            lambda: call_with_model(model),
            f"OpenAI reranker ({model})",
        )
        used_model = model
    except Exception as primary_exc:
        fallback_model = os.getenv("OPENAI_RERANK_FALLBACK_MODEL", OPENAI_RERANK_FALLBACK_MODEL)
        if fallback_model and fallback_model != model:
            try:
                results = _call_with_retries(
                    lambda: call_with_model(fallback_model),
                    f"OpenAI reranker ({fallback_model})",
                )
                used_model = fallback_model
            except Exception:
                if strict:
                    raise primary_exc
                raise primary_exc
        elif strict:
            raise
        else:
            raise

    _save_cached_rerank(_cache_key("openai", used_model, query, candidates, top_k), results)
    return results


def _openai_rerank_instructions() -> str:
    return (
        "You are a high-precision listwise reranker for a Vietnamese RAG system "
        "about drug law, controlled substances, penalties, enforcement, and "
        "related Vietnamese news. Rank passages by usefulness for answering the "
        "query. Prefer exact legal provisions, definitions, thresholds, penalties, "
        "decree/circular details, and direct evidence. Penalize duplicates, vague "
        "mentions, and passages that are only topically adjacent. Return only the "
        "structured JSON requested by the schema."
    )


def _openai_rerank_schema() -> dict[str, Any]:
    return {
        "type": "json_schema",
        "name": "rerank_response",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "results": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "id": {"type": "integer"},
                            "score": {
                                "type": "number",
                                "description": "Relevance score from 0.0 to 1.0.",
                            },
                            "reason": {
                                "type": "string",
                                "description": "Short reason for ranking this candidate.",
                            },
                        },
                        "required": ["id", "score", "reason"],
                    },
                }
            },
            "required": ["results"],
        },
    }


def _supports_reasoning(model_name: str) -> bool:
    normalized = model_name.lower()
    return normalized.startswith(("gpt-5", "o1", "o3", "o4"))


def _results_from_openai_payload(
    data: dict,
    candidates: list[dict],
    top_k: int,
    model_name: str,
) -> list[dict]:
    raw_results = data.get("results", [])
    ranked: list[dict] = []
    used: set[int] = set()

    for result in raw_results:
        try:
            idx = int(result["id"])
        except (KeyError, TypeError, ValueError):
            continue
        if idx < 0 or idx >= len(candidates) or idx in used:
            continue
        used.add(idx)
        score = _clamp_score(float(result.get("score", 0.0) or 0.0))
        item = _copy_candidate(candidates[idx])
        item["original_score"] = float(candidates[idx].get("score", 0.0) or 0.0)
        item["score"] = score
        item["rerank_score"] = score
        item["rerank_provider"] = f"openai:{model_name}"
        item["rerank_reason"] = str(result.get("reason", ""))[:300]
        ranked.append(item)

    if len(ranked) < min(top_k, len(candidates)):
        ranked.extend(_append_missing_by_original_score(candidates, used, top_k - len(ranked)))

    return _ranked_with_positions(ranked[:top_k])


# =============================================================================
# Optional local cross-encoder
# =============================================================================

def _rerank_local_cross_encoder(query: str, candidates: list[dict], top_k: int) -> list[dict]:
    try:
        from sentence_transformers import CrossEncoder
    except ImportError as exc:
        raise RuntimeError(
            "sentence-transformers is not installed. Install it or use "
            "RERANK_PROVIDER=jina/openai."
        ) from exc

    model_name = os.getenv("LOCAL_RERANK_MODEL", LOCAL_RERANK_MODEL)
    model = CrossEncoder(model_name)
    pairs = [(query, _candidate_text(candidate)) for candidate in candidates]
    scores = model.predict(pairs)
    ranked: list[dict] = []
    for candidate, score in zip(candidates, scores):
        item = _copy_candidate(candidate)
        item["original_score"] = float(candidate.get("score", 0.0) or 0.0)
        item["score"] = float(score)
        item["rerank_score"] = float(score)
        item["rerank_provider"] = f"local:{model_name}"
        ranked.append(item)
    ranked.sort(key=lambda item: item["score"], reverse=True)
    return _ranked_with_positions(ranked[:top_k])


# =============================================================================
# Offline safety fallback
# =============================================================================

def _rerank_offline(query: str, candidates: list[dict], top_k: int) -> list[dict]:
    query_tokens = _tokenize(query)
    original_scores = [float(candidate.get("score", 0.0) or 0.0) for candidate in candidates]
    normalized_original = _minmax_normalize(original_scores)

    ranked: list[dict] = []
    for idx, candidate in enumerate(candidates):
        text = _candidate_text(candidate)
        lexical_score = _lexical_overlap_score(query_tokens, _tokenize(text))
        phrase_score = _phrase_score(query, text)
        metadata_score = _metadata_match_score(query, candidate.get("metadata", {}))
        score = (
            0.45 * lexical_score
            + 0.30 * phrase_score
            + 0.15 * normalized_original[idx]
            + 0.10 * metadata_score
        )
        item = _copy_candidate(candidate)
        item["original_score"] = float(candidate.get("score", 0.0) or 0.0)
        item["score"] = float(score)
        item["rerank_score"] = float(score)
        item["rerank_provider"] = "offline_fallback"
        ranked.append(item)

    ranked.sort(key=lambda item: item["score"], reverse=True)
    return _ranked_with_positions(ranked[:top_k])


# =============================================================================
# Shared result helpers
# =============================================================================

def _results_from_ranked_indices(
    raw_results: list[dict],
    candidates: list[dict],
    top_k: int,
    provider: str,
) -> list[dict]:
    ranked: list[dict] = []
    used: set[int] = set()

    for result in raw_results:
        try:
            idx = int(result["index"])
        except (KeyError, TypeError, ValueError):
            continue
        if idx < 0 or idx >= len(candidates) or idx in used:
            continue
        used.add(idx)
        score = float(
            result.get("relevance_score", result.get("score", result.get("rerank_score", 0.0)))
            or 0.0
        )
        item = _copy_candidate(candidates[idx])
        item["original_score"] = float(candidates[idx].get("score", 0.0) or 0.0)
        item["score"] = score
        item["rerank_score"] = score
        item["rerank_provider"] = provider
        ranked.append(item)

    if len(ranked) < min(top_k, len(candidates)):
        ranked.extend(_append_missing_by_original_score(candidates, used, top_k - len(ranked)))

    return _ranked_with_positions(ranked[:top_k])


def _append_missing_by_original_score(
    candidates: list[dict],
    used: set[int],
    limit: int,
) -> list[dict]:
    if limit <= 0:
        return []

    remaining = [
        (idx, candidate)
        for idx, candidate in enumerate(candidates)
        if idx not in used
    ]
    remaining.sort(key=lambda pair: float(pair[1].get("score", 0.0) or 0.0), reverse=True)

    results: list[dict] = []
    for _, candidate in remaining[:limit]:
        item = _copy_candidate(candidate)
        original = float(candidate.get("score", 0.0) or 0.0)
        item["original_score"] = original
        item["score"] = original
        item["rerank_score"] = original
        item["rerank_provider"] = "original_score_backfill"
        results.append(item)
    return results


def _ranked_with_positions(results: list[dict]) -> list[dict]:
    for rank, item in enumerate(results, start=1):
        item["rank"] = rank
    return results


def _copy_candidate(candidate: dict) -> dict:
    item = dict(candidate)
    item["metadata"] = dict(candidate.get("metadata") or {})
    return item


def _candidate_text(candidate: dict) -> str:
    metadata = candidate.get("metadata") or {}
    prefix_parts = [
        str(metadata.get("title", "") or ""),
        str(metadata.get("section", "") or ""),
        str(metadata.get("source", "") or metadata.get("source_path", "") or ""),
        str(metadata.get("url", "") or ""),
    ]
    prefix = "\n".join(part for part in prefix_parts if part.strip())
    content = str(candidate.get("content", "") or "")
    text = f"{prefix}\n\n{content}" if prefix else content
    return _compact_text(text, MAX_DOCUMENT_CHARS)


def _compact_metadata(metadata: dict) -> dict:
    allowed_keys = [
        "title",
        "section",
        "source",
        "source_path",
        "url",
        "type",
        "doc_type",
        "chunk_index",
    ]
    return {key: metadata[key] for key in allowed_keys if key in metadata and metadata[key]}


def _compact_text(text: str, max_chars: int) -> str:
    text = re.sub(r"\s+", " ", text or "").strip()
    if len(text) <= max_chars:
        return text
    head = max_chars // 2
    tail = max_chars - head
    return f"{text[:head]} ... {text[-tail:]}"


def _candidate_key(candidate: dict) -> str:
    metadata = candidate.get("metadata") or {}
    for field in ("chunk_id", "id"):
        if metadata.get(field):
            return str(metadata[field])

    source = metadata.get("source_path") or metadata.get("source") or metadata.get("url")
    chunk_index = metadata.get("chunk_index")
    if source is not None and chunk_index is not None:
        return f"{source}::{chunk_index}"

    content = str(candidate.get("content", "") or "")
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


# =============================================================================
# MMR helpers
# =============================================================================

def _mmr_relevance_scores(
    query_embedding: Optional[list[float]],
    candidates: list[dict],
) -> list[float]:
    if query_embedding:
        scores: list[float] = []
        for candidate in candidates:
            embedding = candidate.get("embedding")
            if embedding:
                scores.append(float(_cosine_similarity(query_embedding, embedding)))
            else:
                scores.append(float(candidate.get("score", 0.0) or 0.0))
        return _minmax_normalize(scores)

    original_scores = [float(candidate.get("score", 0.0) or 0.0) for candidate in candidates]
    return _minmax_normalize(original_scores)


def _candidate_similarity(left: dict, right: dict) -> float:
    left_embedding = left.get("embedding")
    right_embedding = right.get("embedding")
    if left_embedding and right_embedding:
        return float(_cosine_similarity(left_embedding, right_embedding))

    left_tokens = set(_tokenize(_candidate_text(left)))
    right_tokens = set(_tokenize(_candidate_text(right)))
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


# =============================================================================
# Text scoring helpers
# =============================================================================

def _tokenize(text: str) -> list[str]:
    normalized = _normalize_text(text)
    tokens = re.findall(r"[a-z0-9]+", normalized)
    return [token for token in tokens if len(token) > 1 or token.isdigit()]


def _normalize_text(text: str) -> str:
    lowered = (text or "").lower()
    stripped = "".join(
        char
        for char in unicodedata.normalize("NFD", lowered)
        if unicodedata.category(char) != "Mn"
    )
    return stripped.replace("đ", "d")


def _lexical_overlap_score(query_tokens: list[str], document_tokens: list[str]) -> float:
    if not query_tokens or not document_tokens:
        return 0.0
    query_counts: dict[str, int] = {}
    for token in query_tokens:
        query_counts[token] = query_counts.get(token, 0) + 1

    doc_counts: dict[str, int] = {}
    for token in document_tokens:
        doc_counts[token] = min(3, doc_counts.get(token, 0) + 1)

    overlap = 0
    for token, query_count in query_counts.items():
        overlap += min(query_count, doc_counts.get(token, 0))
    recall = overlap / max(1, sum(query_counts.values()))
    precision = overlap / max(1, sum(doc_counts.values()))
    if recall + precision == 0:
        return 0.0
    return 2 * recall * precision / (recall + precision)


def _phrase_score(query: str, text: str) -> float:
    normalized_query = _normalize_text(query)
    normalized_text = _normalize_text(text)
    if not normalized_query or not normalized_text:
        return 0.0

    score = 0.0
    if normalized_query in normalized_text:
        score += 0.6

    query_terms = [term for term in _tokenize(query) if len(term) > 2]
    if len(query_terms) >= 2:
        bigrams = [" ".join(query_terms[idx:idx + 2]) for idx in range(len(query_terms) - 1)]
        matches = sum(1 for bigram in bigrams if bigram in normalized_text)
        score += 0.4 * (matches / max(1, len(bigrams)))

    return min(1.0, score)


def _metadata_match_score(query: str, metadata: dict) -> float:
    if not metadata:
        return 0.0
    metadata_text = " ".join(str(value) for value in metadata.values() if value)
    return max(
        _phrase_score(query, metadata_text),
        _lexical_overlap_score(_tokenize(query), _tokenize(metadata_text)),
    )


def _minmax_normalize(scores: list[float]) -> list[float]:
    if not scores:
        return []
    min_score = min(scores)
    max_score = max(scores)
    if math.isclose(max_score, min_score):
        return [1.0 if max_score > 0 else 0.0 for _ in scores]
    return [(score - min_score) / (max_score - min_score) for score in scores]


def _clamp_score(score: float) -> float:
    return min(1.0, max(0.0, score))


# =============================================================================
# API/cache helpers
# =============================================================================

def _external_api_disabled_for_tests() -> bool:
    return bool(os.getenv("PYTEST_CURRENT_TEST")) and os.getenv("RERANK_ALLOW_API_IN_TESTS") != "1"


def _call_with_retries(func, operation: str):
    last_exc: Optional[Exception] = None
    for attempt in range(1, MAX_API_RETRIES + 1):
        try:
            return func()
        except Exception as exc:
            last_exc = exc
            if attempt >= MAX_API_RETRIES:
                break
            time.sleep(min(8.0, 0.75 * (2 ** (attempt - 1))))
    raise RuntimeError(f"{operation} failed after {MAX_API_RETRIES} attempts: {last_exc}")


def _retry_after_seconds(value: Optional[str]) -> float:
    if not value:
        return 1.0
    try:
        return min(30.0, max(0.1, float(value)))
    except ValueError:
        return 1.0


def _cache_key(
    provider: str,
    model: str,
    query: str,
    candidates: list[dict],
    top_k: int,
) -> str:
    fingerprint = {
        "provider": provider,
        "model": model,
        "query": query,
        "top_k": top_k,
        "candidates": [
            {
                "key": _candidate_key(candidate),
                "content_hash": hashlib.sha256(
                    str(candidate.get("content", "") or "").encode("utf-8")
                ).hexdigest(),
            }
            for candidate in candidates
        ],
    }
    raw = json.dumps(fingerprint, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _load_cached_rerank(cache_key: str) -> Optional[list[dict]]:
    cache = _load_rerank_cache()
    cached = cache.get(cache_key)
    if isinstance(cached, list):
        return cached
    return None


def _save_cached_rerank(cache_key: str, results: list[dict]) -> None:
    cache = _load_rerank_cache()
    cache[cache_key] = results
    RERANK_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    RERANK_CACHE_PATH.write_text(
        json.dumps(cache, ensure_ascii=False),
        encoding="utf-8",
    )


def _load_rerank_cache() -> dict:
    if not RERANK_CACHE_PATH.exists():
        return {}
    try:
        return json.loads(RERANK_CACHE_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def _extract_response_text(response) -> str:
    if getattr(response, "output_text", None):
        return response.output_text

    pieces: list[str] = []
    for output_item in getattr(response, "output", []) or []:
        for content_part in getattr(output_item, "content", []) or []:
            text = getattr(content_part, "text", None)
            if text:
                pieces.append(text)
    return "\n".join(pieces).strip()


if __name__ == "__main__":
    dummy_candidates = [
        {"content": "Điều 248: Tội tàng trữ trái phép chất ma tuý", "score": 0.8, "metadata": {}},
        {"content": "Nghệ sĩ X bị bắt vì sử dụng ma tuý", "score": 0.7, "metadata": {}},
        {"content": "Hình phạt tù từ 2-7 năm cho tội tàng trữ", "score": 0.6, "metadata": {}},
    ]
    results = rerank("hình phạt tàng trữ ma tuý", dummy_candidates, top_k=2)
    for result in results:
        print(f"[{result['score']:.3f}] {result['content']}")

"""
Task 6 - Lexical search with Vietnamese-aware BM25.

The index is built over Task 4 chunks, not whole documents. That makes lexical
results citation-friendly and easier to fuse with semantic search later.
"""

from __future__ import annotations

import json
import math
import os
import re
import unicodedata
from collections import Counter
from pathlib import Path

try:
    from .task4_chunking_indexing import CHUNKS_JSONL, chunk_documents, load_documents
except ImportError:  # Allows: python src/task6_lexical_search.py
    from task4_chunking_indexing import CHUNKS_JSONL, chunk_documents, load_documents  # type: ignore

CORPUS: list[dict] = []
_BM25_INDEX = None

LEGAL_QUERY_RE = re.compile(r"\bđiều\s+\d+[a-zA-Z]?\b", re.IGNORECASE)
MIN_LEXICAL_TOP_SCORE = float(os.getenv("MIN_LEXICAL_TOP_SCORE", "18"))


def build_bm25_index(corpus: list[dict]):
    """
    Build a BM25 index. Uses rank_bm25 when installed; otherwise a compact local
    BM25 implementation with the same scoring idea.
    """
    tokenized_corpus = [_tokenize_document(doc["content"], doc.get("metadata", {})) for doc in corpus]
    try:
        from rank_bm25 import BM25Okapi

        return BM25Okapi(tokenized_corpus)
    except Exception:
        return _SimpleBM25(tokenized_corpus)


def lexical_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Lexical BM25 search.

    Returns:
        List of {'content': str, 'score': float, 'metadata': dict}
        sorted by score descending.
    """
    query = (query or "").strip()
    if not query or top_k <= 0:
        return []

    corpus, bm25 = _get_or_build_index()
    if not corpus:
        return []

    query_tokens = _tokenize_query(query)
    if not query_tokens:
        return []

    scores = bm25.get_scores(query_tokens)
    ranked_indices = sorted(range(len(scores)), key=lambda index: scores[index], reverse=True)

    results: list[dict] = []
    for index in ranked_indices[: max(top_k * 4, top_k)]:
        base_score = float(scores[index])
        boost = _lexical_boost(query, corpus[index])
        score = base_score + boost
        if score <= 0:
            continue
        results.append(
            {
                "content": corpus[index]["content"],
                "score": score,
                "metadata": corpus[index].get("metadata", {}),
            }
        )

    results.sort(key=lambda item: item["score"], reverse=True)
    min_score = 0.0 if _has_strong_domain_signal(query) else MIN_LEXICAL_TOP_SCORE
    if results and results[0]["score"] < min_score:
        return []
    return results[:top_k]


def load_corpus() -> list[dict]:
    """Load Task 4 chunks if available; otherwise build deterministic chunks."""
    if CHUNKS_JSONL.exists():
        rows: list[dict] = []
        with CHUNKS_JSONL.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    row = json.loads(line)
                    if row.get("content"):
                        rows.append({"content": row["content"], "metadata": row.get("metadata", {})})
        if rows:
            return rows

    return chunk_documents(load_documents(), use_semantic=False)


def _get_or_build_index():
    global CORPUS, _BM25_INDEX
    if _BM25_INDEX is None:
        CORPUS = load_corpus()
        _BM25_INDEX = build_bm25_index(CORPUS)
    return CORPUS, _BM25_INDEX


def _tokenize_query(text: str) -> list[str]:
    tokens = _tokenize_text(text)
    tokens.extend(_phrase_tokens(text))
    return tokens


def _tokenize_document(text: str, metadata: dict) -> list[str]:
    enriched_text = "\n".join(
        [
            metadata.get("title", ""),
            metadata.get("section", ""),
            metadata.get("source", ""),
            text,
        ]
    )
    tokens = _tokenize_text(enriched_text)
    tokens.extend(_phrase_tokens(enriched_text))
    return tokens


def _tokenize_text(text: str) -> list[str]:
    normalized = _normalize_for_search(text)
    base_tokens = re.findall(r"[0-9a-zA-ZÀ-ỹĐđ]+", normalized)
    tokens: list[str] = []

    for token in base_tokens:
        if len(token) <= 1 and not token.isdigit():
            continue
        tokens.append(token)
        ascii_token = _strip_accents(token)
        if ascii_token != token and (len(ascii_token) >= 3 or token.isdigit()):
            tokens.append(ascii_token)

    for size in (2, 3):
        for index in range(0, max(len(base_tokens) - size + 1, 0)):
            ngram = "_".join(base_tokens[index : index + size])
            tokens.append(ngram)
            ascii_ngram = _strip_accents(ngram)
            if ascii_ngram != ngram:
                tokens.append(ascii_ngram)

    return tokens


def _phrase_tokens(text: str) -> list[str]:
    normalized = _normalize_for_search(text)
    phrases = []
    phrase_patterns = [
        (r"ma\s+t[úu]y", "ma_tuy"),
        (r"chất\s+ma\s+t[úu]y", "chat_ma_tuy"),
        (r"phòng\s+chống\s+ma\s+t[úu]y", "phong_chong_ma_tuy"),
        (r"tàng\s+trữ", "tang_tru"),
        (r"trái\s+phép", "trai_phep"),
        (r"sử\s+dụng\s+trái\s+phép", "su_dung_trai_phep"),
        (r"người\s+sử\s+dụng", "nguoi_su_dung"),
        (r"người\s+mẫu", "nguoi_mau"),
        (r"ca\s+sĩ", "ca_si"),
        (r"diễn\s+viên", "dien_vien"),
    ]
    for pattern, phrase in phrase_patterns:
        if re.search(pattern, normalized):
            phrases.append(phrase)

    for match in LEGAL_QUERY_RE.findall(normalized):
        phrases.append(match.replace(" ", "_"))

    return phrases


def _lexical_boost(query: str, doc: dict) -> float:
    query_norm = _normalize_for_search(query)
    content_norm = _normalize_for_search(doc["content"])
    metadata = doc.get("metadata", {})
    metadata_norm = _normalize_for_search(
        " ".join(
            [
                metadata.get("title", ""),
                metadata.get("section", ""),
                metadata.get("source", ""),
                metadata.get("source_path", ""),
            ]
        )
    )

    boost = 0.0
    if query_norm and query_norm in content_norm:
        boost += 4.0

    for legal_ref in LEGAL_QUERY_RE.findall(query_norm):
        if legal_ref in content_norm or legal_ref in metadata_norm:
            boost += 6.0

    query_terms = set(_strip_accents(token) for token in re.findall(r"[0-9a-zA-ZÀ-ỹĐđ]+", query_norm))
    metadata_terms = set(_strip_accents(token) for token in re.findall(r"[0-9a-zA-ZÀ-ỹĐđ]+", metadata_norm))
    boost += min(len(query_terms & metadata_terms) * 0.35, 2.0)

    for phrase in ("ma túy", "ma tuý", "chất ma túy", "tàng trữ", "trái phép", "người mẫu", "ca sĩ"):
        if phrase in query_norm and phrase in content_norm:
            boost += 1.25

    return boost


def _has_strong_domain_signal(query: str) -> bool:
    normalized = _normalize_for_search(query)
    ascii_normalized = _strip_accents(normalized)
    return any(
        phrase in normalized or phrase in ascii_normalized
        for phrase in (
            "ma túy",
            "ma tuy",
            "chất ma túy",
            "chat ma tuy",
            "tàng trữ",
            "tang tru",
            "trái phép",
            "trai phep",
            "cai nghiện",
            "cai nghien",
            "chất cấm",
            "chat cam",
            "tiền chất",
            "tien chat",
            "nghiện",
            "nghien",
        )
    )


def _normalize_for_search(text: str) -> str:
    text = unicodedata.normalize("NFC", text or "").lower()
    # Both spellings appear in Vietnamese sources and queries.
    return text.replace("ma tuý", "ma túy").replace("chất ma tuý", "chất ma túy")


def _strip_accents(text: str) -> str:
    decomposed = unicodedata.normalize("NFD", text)
    stripped = "".join(ch for ch in decomposed if unicodedata.category(ch) != "Mn")
    return stripped.replace("đ", "d").replace("Đ", "D")


class _SimpleBM25:
    def __init__(self, tokenized_corpus: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.tokenized_corpus = tokenized_corpus
        self.k1 = k1
        self.b = b
        self.doc_freqs = [Counter(doc) for doc in tokenized_corpus]
        self.doc_lengths = [len(doc) for doc in tokenized_corpus]
        self.avg_doc_length = sum(self.doc_lengths) / max(len(self.doc_lengths), 1)
        self.idf = self._compute_idf()

    def _compute_idf(self) -> dict[str, float]:
        document_count = len(self.tokenized_corpus)
        term_document_counts: Counter[str] = Counter()
        for doc in self.tokenized_corpus:
            term_document_counts.update(set(doc))

        return {
            term: math.log(1 + (document_count - count + 0.5) / (count + 0.5))
            for term, count in term_document_counts.items()
        }

    def get_scores(self, query_tokens: list[str]) -> list[float]:
        scores: list[float] = []
        for doc_index, freqs in enumerate(self.doc_freqs):
            doc_length = self.doc_lengths[doc_index] or 1
            score = 0.0
            for token in query_tokens:
                term_frequency = freqs.get(token, 0)
                if term_frequency == 0:
                    continue
                denominator = term_frequency + self.k1 * (
                    1 - self.b + self.b * doc_length / max(self.avg_doc_length, 1)
                )
                score += self.idf.get(token, 0.0) * term_frequency * (self.k1 + 1) / denominator
            scores.append(score)
        return scores


if __name__ == "__main__":
    results = lexical_search("Điều 248 tàng trữ trái phép chất ma túy", top_k=5)
    for result in results:
        print(f"[{result['score']:.3f}] {result['content'][:120]}...")

"""
Task 9 — Retrieval Pipeline Hoàn Chỉnh.

Kết hợp semantic search + lexical search + reranking + PageIndex fallback
thành một pipeline thống nhất.

Logic:
    1. Chạy semantic_search + lexical_search song song
    2. Merge kết quả (RRF hoặc weighted fusion)
    3. Rerank
    4. Nếu top result score < threshold → fallback sang PageIndex
    5. Return top_k results
"""

from __future__ import annotations

import os

from .task5_semantic_search import semantic_search
from .task6_lexical_search import lexical_search
from .task7_reranking import rerank, rerank_rrf
from .task8_pageindex_vectorless import pageindex_search


# =============================================================================
# CONFIGURATION
# =============================================================================

SCORE_THRESHOLD = 0.3   # Nếu best score < threshold → fallback PageIndex
DEFAULT_TOP_K = 5
# "cross_encoder" delegates to Task 7 auto mode: Jina first when JINA_API_KEY
# exists, then OpenAI/listwise, then an offline safety fallback.
RERANK_METHOD = os.getenv("RERANK_METHOD", "cross_encoder")
RETRIEVAL_PREFETCH_MULTIPLIER = int(os.getenv("RETRIEVAL_PREFETCH_MULTIPLIER", "3"))
MIN_RRF_SCORE = float(os.getenv("MIN_RRF_SCORE", "0.015"))
OFFLINE_RERANK_THRESHOLD = float(os.getenv("OFFLINE_RERANK_THRESHOLD", "0.12"))
JINA_RERANK_THRESHOLD = float(os.getenv("JINA_RERANK_THRESHOLD", "0.05"))
OPENAI_RERANK_THRESHOLD = float(os.getenv("OPENAI_RERANK_THRESHOLD", "0.20"))
LOCAL_RERANK_THRESHOLD = float(os.getenv("LOCAL_RERANK_THRESHOLD", "0.0"))


def retrieve(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    score_threshold: float = SCORE_THRESHOLD,
    use_reranking: bool = True,
) -> list[dict]:
    """
    Retrieval pipeline hoàn chỉnh với fallback logic.

    Pipeline:
        Query
          ├→ Semantic Search → results_dense
          ├→ Lexical Search  → results_sparse
          │
          ├→ Merge (RRF) → merged_results
          ├→ Rerank → reranked_results
          │
          └→ If best_score < threshold:
                └→ PageIndex Vectorless → fallback_results

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả cuối cùng
        score_threshold: Ngưỡng điểm tối thiểu cho hybrid results
        use_reranking: Có áp dụng reranking hay không

    Returns:
        List of {
            'content': str,
            'score': float,
            'metadata': dict,
            'source': str  # 'hybrid' hoặc 'pageindex'
        }
    """
    query = (query or "").strip()
    if not query:
        return []
    if top_k <= 0:
        return []

    prefetch_k = max(top_k * RETRIEVAL_PREFETCH_MULTIPLIER, top_k)

    # Step 1: semantic + lexical search
    dense_results = semantic_search(query, top_k=prefetch_k)
    sparse_results = lexical_search(query, top_k=prefetch_k)
    _tag_ranker(dense_results, "semantic")
    _tag_ranker(sparse_results, "lexical")

    # Step 2: Merge using RRF
    merged = rerank_rrf([dense_results, sparse_results], top_k=prefetch_k)
    for item in merged:
        item["source"] = "hybrid"
        item.setdefault("metadata", {})["retrieval_stage"] = "rrf_fusion"

    # Step 3: Rerank
    if use_reranking and merged:
        try:
            final_results = rerank(query, merged, top_k=top_k, method=RERANK_METHOD)
            for item in final_results:
                item["source"] = "hybrid"
                item.setdefault("metadata", {})["retrieval_stage"] = "reranked"
        except Exception as exc:
            print(f"  [WARN] Reranker failed, using RRF results: {exc}")
            final_results = merged[:top_k]
    else:
        final_results = merged[:top_k]

    # Step 4: Check threshold → fallback to PageIndex
    if _should_fallback(final_results, score_threshold):
        best_score = final_results[0]["score"] if final_results else 0.0
        print(f"  [WARN] Hybrid score ({best_score:.3f}) "
              f"< threshold ({score_threshold}). Fallback -> PageIndex")
        fallback = pageindex_search(query, top_k=top_k)
        return fallback

    return final_results[:top_k]


def _tag_ranker(results: list[dict], ranker: str) -> None:
    for rank, item in enumerate(results, start=1):
        metadata = item.setdefault("metadata", {})
        metadata["retriever"] = ranker
        metadata["retriever_rank"] = rank


def _should_fallback(results: list[dict], score_threshold: float) -> bool:
    if not results:
        return True

    top = results[0]
    provider = str(top.get("rerank_provider", ""))
    top_score = float(top.get("score", 0.0) or 0.0)

    if provider == "offline_fallback":
        return top_score < OFFLINE_RERANK_THRESHOLD

    if provider.startswith("jina:"):
        return top_score < JINA_RERANK_THRESHOLD

    if provider.startswith("openai:"):
        return top_score < OPENAI_RERANK_THRESHOLD

    if provider.startswith("local:"):
        return top_score < LOCAL_RERANK_THRESHOLD

    if provider == "rrf" or top.get("rrf_score") is not None:
        return top_score < MIN_RRF_SCORE

    return top_score < score_threshold


if __name__ == "__main__":
    test_queries = [
        "Hình phạt cho tội tàng trữ trái phép chất ma tuý",
        "Nghệ sĩ nào bị bắt vì sử dụng ma tuý năm 2024",
        "Luật phòng chống ma tuý 2021 quy định gì về cai nghiện",
    ]

    for q in test_queries:
        print(f"\nQuery: {q}")
        print("-" * 60)
        results = retrieve(q, top_k=3)
        for i, r in enumerate(results, 1):
            print(f"  {i}. [{r['score']:.3f}] [{r['source']}] {r['content'][:80]}...")

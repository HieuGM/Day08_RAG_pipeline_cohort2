"""
Task 10 - Generation with citations.

This module keeps generation grounded in retrieved chunks:
    1. Retrieve or accept context chunks.
    2. Reorder context to reduce "lost in the middle".
    3. Format each chunk with a stable citation label.
    4. Generate with OpenAI Responses API when available.
    5. Fall back to an extractive cited answer if the LLM is unavailable.
"""

from __future__ import annotations

import os
import re
from typing import Any, Optional

from dotenv import load_dotenv

try:
    from .task9_retrieval_pipeline import retrieve
except ImportError:  # Allows: python src/task10_generation.py
    from task9_retrieval_pipeline import retrieve  # type: ignore

load_dotenv()


# =============================================================================
# Configuration
# =============================================================================

# Five chunks usually fit comfortably while still giving legal/news questions
# enough evidence. The reorder step keeps the strongest chunks near the edges.
TOP_K = int(os.getenv("GENERATION_TOP_K", "5"))

# RAG answers should be factual and stable, so temperature is low. top_p is kept
# moderately high to avoid brittle phrasing while staying grounded in context.
TOP_P = float(os.getenv("GENERATION_TOP_P", "0.9"))
TEMPERATURE = float(os.getenv("GENERATION_TEMPERATURE", "0.2"))

GENERATION_MODEL = os.getenv("GENERATION_MODEL", "gpt-5.5")
GENERATION_FALLBACK_MODELS = [
    model.strip()
    for model in os.getenv("GENERATION_FALLBACK_MODELS", "gpt-4.1,gpt-4o-mini").split(",")
    if model.strip()
]
GENERATION_REASONING_EFFORT = os.getenv("GENERATION_REASONING_EFFORT", "low")
MAX_CONTEXT_CHARS = int(os.getenv("GENERATION_MAX_CONTEXT_CHARS", "14000"))
MAX_CHUNK_CHARS = int(os.getenv("GENERATION_MAX_CHUNK_CHARS", "2600"))

INSUFFICIENT_EVIDENCE_MESSAGE = "Tôi không thể xác minh thông tin này từ nguồn hiện có."


# =============================================================================
# Prompt
# =============================================================================

SYSTEM_PROMPT = """Bạn là trợ lý RAG trả lời bằng tiếng Việt về pháp luật ma túy Việt Nam và tin tức liên quan.

Chỉ sử dụng thông tin trong CONTEXT. Mỗi mệnh đề thực tế phải có citation ngay sau mệnh đề đó, dùng đúng nhãn citation được cung cấp, ví dụ [Luật Phòng chống ma túy 2021, Điều 2] hoặc [Tuổi Trẻ, 2024].

Nếu CONTEXT không chứa đủ bằng chứng để trả lời, hãy nói: "Tôi không thể xác minh thông tin này từ nguồn hiện có." Không suy đoán, không dùng kiến thức ngoài context.

Trả lời ngắn gọn, trực tiếp, ưu tiên các điều khoản, định nghĩa, đối tượng, trách nhiệm, thủ tục, hoặc sự kiện báo chí có trong nguồn."""


# =============================================================================
# Document reordering
# =============================================================================

def reorder_for_llm(chunks: list[dict]) -> list[dict]:
    """
    Place high-scoring chunks at the beginning and end of context.

    For input sorted by relevance [1, 2, 3, 4, 5], output is [1, 3, 5, 4, 2].
    """
    if len(chunks) <= 2:
        return list(chunks)

    reordered: list[dict] = []
    for index in range(0, len(chunks), 2):
        reordered.append(chunks[index])
    for index in range(len(chunks) - 1 - (len(chunks) % 2), 0, -2):
        reordered.append(chunks[index])
    return reordered


# =============================================================================
# Context formatting
# =============================================================================

def format_context(chunks: list[dict]) -> str:
    """
    Format chunks for the model with explicit citation labels.
    """
    context_parts: list[str] = []
    used_chars = 0

    for index, chunk in enumerate(chunks, start=1):
        metadata = chunk.get("metadata", {}) or {}
        citation = chunk.get("citation") or _citation_label(metadata, index)
        source_name = _source_name(metadata, index)
        doc_type = metadata.get("type") or metadata.get("doc_type") or "unknown"
        section = metadata.get("section") or metadata.get("title") or metadata.get("section_path") or ""
        url = metadata.get("url", "")
        score = float(chunk.get("score", 0.0) or 0.0)

        content = _compact_text(str(chunk.get("content", "") or ""), MAX_CHUNK_CHARS)
        header = (
            f"[Document {index}]\n"
            f"Citation: [{citation}]\n"
            f"Source: {source_name}\n"
            f"Type: {doc_type}\n"
            f"Score: {score:.3f}"
        )
        if section:
            header += f"\nSection: {section}"
        if url:
            header += f"\nURL: {url}"

        block = f"{header}\nContent:\n{content}"
        if used_chars + len(block) > MAX_CONTEXT_CHARS:
            break
        context_parts.append(block)
        used_chars += len(block)

    return "\n\n---\n\n".join(context_parts)


def enrich_sources(chunks: list[dict]) -> list[dict]:
    """Copy source chunks and attach citation labels for UI/display."""
    enriched: list[dict] = []
    for index, chunk in enumerate(chunks, start=1):
        item = dict(chunk)
        item["metadata"] = dict(chunk.get("metadata") or {})
        item["citation"] = _citation_label(item["metadata"], index)
        enriched.append(item)
    return enriched


# =============================================================================
# Generation
# =============================================================================

def generate_with_citation(
    query: str,
    top_k: int = TOP_K,
    context_chunks: Optional[list[dict]] = None,
) -> dict:
    """
    End-to-end RAG answer with citations.
    """
    query = (query or "").strip()
    if not query:
        return _empty_generation("Câu hỏi trống.")

    chunks = context_chunks if context_chunks is not None else retrieve(query, top_k=top_k)
    if not chunks:
        return _empty_generation(INSUFFICIENT_EVIDENCE_MESSAGE)

    reordered = reorder_for_llm(enrich_sources(chunks))
    context = format_context(reordered)
    if not context.strip():
        return _empty_generation(INSUFFICIENT_EVIDENCE_MESSAGE)

    user_message = (
        f"CONTEXT:\n{context}\n\n"
        f"QUESTION:\n{query}\n\n"
        "Yêu cầu: trả lời chỉ dựa trên CONTEXT và dùng citation label đã cho."
    )

    try:
        answer, model = _generate_openai(user_message)
    except Exception as exc:
        print(f"[WARN] LLM generation failed, using extractive fallback: {exc}")
        answer = _generate_extractive_answer(query, reordered)
        model = "extractive_fallback"

    return {
        "answer": answer.strip() or INSUFFICIENT_EVIDENCE_MESSAGE,
        "sources": reordered,
        "retrieval_source": chunks[0].get("source", "hybrid") if chunks else "none",
        "model": model,
    }


def _generate_openai(user_message: str) -> tuple[str, str]:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is missing.")

    from openai import OpenAI

    client = OpenAI(api_key=api_key)
    models = [GENERATION_MODEL, *GENERATION_FALLBACK_MODELS]
    errors: list[str] = []

    for model in dict.fromkeys(models):
        try:
            answer = _responses_api_generate(client, model, user_message)
            return answer, model
        except Exception as exc:
            errors.append(f"{model}: {exc}")
            try:
                answer = _chat_completions_generate(client, model, user_message)
                return answer, model
            except Exception as chat_exc:
                errors.append(f"{model}/chat: {chat_exc}")

    raise RuntimeError("; ".join(errors[-4:]))


def _responses_api_generate(client, model: str, user_message: str) -> str:
    request: dict[str, Any] = {
        "model": model,
        "instructions": SYSTEM_PROMPT,
        "input": user_message,
        "store": False,
        "max_output_tokens": 900,
    }
    if _supports_reasoning(model):
        request["reasoning"] = {"effort": GENERATION_REASONING_EFFORT}
        request["text"] = {"verbosity": "medium"}
    else:
        request["temperature"] = TEMPERATURE
        request["top_p"] = TOP_P

    response = client.responses.create(**request)
    return _extract_response_text(response)


def _chat_completions_generate(client, model: str, user_message: str) -> str:
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        temperature=TEMPERATURE,
        top_p=TOP_P,
    )
    return response.choices[0].message.content or ""


def _generate_extractive_answer(query: str, chunks: list[dict]) -> str:
    if not chunks:
        return INSUFFICIENT_EVIDENCE_MESSAGE

    query_terms = set(_signal_terms(query))
    sentences: list[str] = []
    seen: set[str] = set()

    for chunk in chunks[:4]:
        citation = chunk.get("citation") or _citation_label(chunk.get("metadata", {}), len(sentences) + 1)
        for sentence in _split_sentences(str(chunk.get("content", "") or "")):
            normalized = _normalize_text(sentence)
            if normalized in seen:
                continue
            seen.add(normalized)
            terms = set(_signal_terms(sentence))
            if query_terms and len(query_terms & terms) == 0:
                continue
            cleaned = sentence.strip()
            if cleaned:
                sentences.append(f"{cleaned} [{citation}]")
            if len(sentences) >= 4:
                break
        if len(sentences) >= 4:
            break

    if not sentences:
        top = chunks[0]
        citation = top.get("citation") or _citation_label(top.get("metadata", {}), 1)
        excerpt = _compact_text(str(top.get("content", "") or ""), 420)
        if not excerpt:
            return INSUFFICIENT_EVIDENCE_MESSAGE
        return f"{excerpt} [{citation}]"

    return " ".join(sentences)


# =============================================================================
# Helpers
# =============================================================================

def _empty_generation(answer: str) -> dict:
    return {
        "answer": answer,
        "sources": [],
        "retrieval_source": "none",
        "model": "none",
    }


def _citation_label(metadata: dict, index: int) -> str:
    title = str(metadata.get("title") or metadata.get("section") or metadata.get("document") or "")
    source_path = str(metadata.get("source_path") or metadata.get("source") or "")
    url = str(metadata.get("url") or "")
    published = str(metadata.get("published_at") or metadata.get("date") or "")
    section = str(metadata.get("section") or metadata.get("title") or "")

    haystack = " ".join([title, source_path, url]).lower()
    if "luat-phong-chong-ma-tuy-2021" in haystack or "luật phòng" in haystack:
        base = "Luật Phòng chống ma túy 2021"
    elif "nghi-dinh-105-2021" in haystack or "105/2021" in haystack:
        base = "Nghị định 105/2021/NĐ-CP"
    elif "nghi-dinh-28-2026" in haystack or "28/2026" in haystack:
        base = "Nghị định 28/2026/NĐ-CP"
    elif "tuoitre.vn" in haystack or "tuổi trẻ" in haystack:
        base = "Tuổi Trẻ"
    elif "vnexpress.net" in haystack or "vnexpress" in haystack:
        base = "VnExpress"
    else:
        base = _clean_title(title) or f"Nguồn {index}"

    year = _extract_year(published or source_path or title)
    if year and year not in base:
        base = f"{base}, {year}"

    legal_section = _extract_legal_section(section)
    if legal_section and legal_section not in base:
        base = f"{base}, {legal_section}"

    return base


def _source_name(metadata: dict, index: int) -> str:
    return (
        str(metadata.get("title") or "")
        or str(metadata.get("source") or "")
        or str(metadata.get("source_path") or "")
        or str(metadata.get("url") or "")
        or f"Nguồn {index}"
    )


def _clean_title(title: str) -> str:
    title = re.sub(r"\s+", " ", title or "").strip()
    return title[:90]


def _extract_year(text: str) -> str:
    match = re.search(r"(20\d{2}|19\d{2})", text or "")
    return match.group(1) if match else ""


def _extract_legal_section(text: str) -> str:
    match = re.search(r"(Điều\s+\d+[a-zA-Z]?)", text or "", flags=re.IGNORECASE)
    if not match:
        return ""
    return match.group(1).strip()


def _compact_text(text: str, max_chars: int) -> str:
    text = re.sub(r"\n{3,}", "\n\n", text or "").strip()
    if len(text) <= max_chars:
        return text
    head = max_chars // 2
    tail = max_chars - head
    return f"{text[:head]}\n...\n{text[-tail:]}"


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


def _supports_reasoning(model: str) -> bool:
    return model.lower().startswith(("gpt-5", "o1", "o3", "o4"))


def _split_sentences(text: str) -> list[str]:
    compact = re.sub(r"\s+", " ", text or "").strip()
    if not compact:
        return []
    return [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?。])\s+|(?<=\.)\s+(?=[A-ZÀ-ỸĐ])", compact)
        if len(sentence.strip()) > 24
    ]


def _signal_terms(text: str) -> list[str]:
    normalized = _normalize_text(text)
    stopwords = {
        "cac",
        "cho",
        "cua",
        "duoc",
        "hay",
        "khi",
        "la",
        "mot",
        "nao",
        "nhung",
        "theo",
        "thi",
        "trong",
        "ve",
        "voi",
    }
    return [
        token
        for token in re.findall(r"[a-z0-9]+", normalized)
        if len(token) >= 3 and token not in stopwords
    ]


def _normalize_text(text: str) -> str:
    import unicodedata

    lowered = (text or "").lower().replace("ma tuý", "ma túy")
    stripped = "".join(
        char
        for char in unicodedata.normalize("NFD", lowered)
        if unicodedata.category(char) != "Mn"
    )
    return stripped.replace("đ", "d")


if __name__ == "__main__":
    test_queries = [
        "Luật Phòng chống ma túy 2021 định nghĩa chất ma túy là gì?",
        "Chi Dân An Tây bị bắt vì tội gì?",
        "Giá bitcoin hôm nay là bao nhiêu?",
    ]

    for question in test_queries:
        print(f"\n{'=' * 70}")
        print(f"Q: {question}")
        print("=" * 70)
        result = generate_with_citation(question)
        print(result["answer"])
        print(f"[Sources: {len(result['sources'])} | via {result['retrieval_source']} | model {result['model']}]")

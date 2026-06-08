"""
RAG Chatbot - Drug Law & News Q&A

Streamlit-based chatbot with:
- Conversation memory for follow-up questions
- Citation display with sources
- Source document preview
- Vietnamese language support
"""

import sys
from pathlib import Path
import hashlib

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
from src.task10_generation import generate_with_citation


FOLLOW_UP_MARKERS = (
    "còn",
    "vậy",
    "tiếp",
    "họ",
    "người đó",
    "trường hợp đó",
    "quy định đó",
    "vì sao",
    "như thế nào",
)


def build_contextual_question(prompt: str) -> str:
    """Add minimal conversation context for short follow-up questions."""
    previous_user_questions = [
        message["content"]
        for message in st.session_state.messages
        if message.get("role") == "user"
    ]
    if previous_user_questions and previous_user_questions[-1] == prompt:
        previous_user_questions = previous_user_questions[:-1]
    if not previous_user_questions:
        return prompt

    normalized = prompt.lower().strip()
    is_short_follow_up = len(prompt.split()) <= 8
    has_marker = any(marker in normalized for marker in FOLLOW_UP_MARKERS)
    if not (is_short_follow_up or has_marker):
        return prompt

    return (
        f"Câu hỏi trước: {previous_user_questions[-1]}\n"
        f"Câu hỏi hiện tại: {prompt}"
    )


def stable_key(*parts) -> str:
    raw = "|".join(str(part) for part in parts)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]


def render_sources(sources: list[dict], key_prefix: str, preview_chars: int = 700) -> None:
    for i, source in enumerate(sources, 1):
        metadata = source.get("metadata", {}) or {}
        source_name = source.get("citation") or metadata.get("source") or metadata.get("title") or f"Nguồn {i}"
        doc_type = metadata.get("type", metadata.get("doc_type", "unknown"))
        score = float(source.get("score", 0) or 0)

        st.markdown(f"**{i}. {source_name}**")
        st.caption(f"Loại: {doc_type} | Độ liên quan: {score:.3f}")

        content = source.get("content", "")
        if content:
            source_id = (
                metadata.get("chunk_id")
                or f"{metadata.get('source_path', '')}:{metadata.get('chunk_index', '')}:{content[:80]}"
            )
            st.text_area(
                f"Nội dung {i}",
                content[:preview_chars] + "..." if len(content) > preview_chars else content,
                height=110,
                key=f"{key_prefix}_{i}_{stable_key(source_id)}",
            )
        st.markdown("---")


def process_pending_prompt() -> None:
    prompt = st.session_state.pending_prompt
    if not prompt:
        return

    st.session_state.pending_prompt = None

    with st.chat_message("assistant"):
        with st.spinner("Đang tìm kiếm và phân tích..."):
            try:
                context_prompt = build_contextual_question(prompt)
                result = generate_with_citation(context_prompt)
                answer = result.get("answer", "Xin lỗi, tôi không thể trả lời câu hỏi này.")
                sources = result.get("sources", [])
                retrieval_source = result.get("retrieval_source", "unknown")

                st.markdown(answer)

                if sources:
                    st.caption(f"{len(sources)} nguồn tham khảo | Phương thức: {retrieval_source}")
                    with st.expander(f"Xem chi tiết {len(sources)} nguồn"):
                        render_sources(
                            sources,
                            key_prefix=f"current_{stable_key(prompt, len(st.session_state.messages))}",
                            preview_chars=800,
                        )

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                        "sources": sources,
                    }
                )
                st.session_state.conversation_history.append(f"Q: {prompt}")
                st.session_state.conversation_history.append(f"A: {answer[:200]}...")
                st.session_state.total_queries += 1
            except Exception as exc:
                error_msg = f"Xin lỗi, đã xảy ra lỗi: {exc}"
                st.error(error_msg)
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": error_msg,
                    }
                )


# =============================================================================
# PAGE CONFIG
# =============================================================================

st.set_page_config(
    page_title="RAG Chatbot - Pháp Luật Ma Túy",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =============================================================================
# SESSION STATE INITIALIZATION
# =============================================================================

def init_session_state():
    """Initialize session state variables."""
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "Xin chào! Tôi là chatbot chuyên về pháp luật ma túy và tin tức liên quan tại Việt Nam. Tôi có thể trả lời câu hỏi của bạn dựa trên các nguồn tài liệu pháp luật và bài báo uy tín. Bạn muốn biết gì?"
            }
        ]

    if "conversation_history" not in st.session_state:
        st.session_state.conversation_history = []

    if "total_queries" not in st.session_state:
        st.session_state.total_queries = 0

    if "pending_prompt" not in st.session_state:
        st.session_state.pending_prompt = None


init_session_state()


# =============================================================================
# SIDEBAR - APP INFO & STATS
# =============================================================================

with st.sidebar:
    st.title("⚖️ RAG Chatbot")
    st.markdown("---")

    st.subheader("Về Chatbot")
    st.info("""
    **Chức năng:**
    - Trả lời câu hỏi về pháp luật ma túy
    - Cập nhật tin tức nghệ sĩ liên quan
    - Có trích dẫn nguồn rõ ràng
    - Hỗ trợ câu hỏi tiếp theo

    **Nguồn dữ liệu:**
    - Luật Phòng chống ma tuý 2021
    - Nghị định 105/2021/NĐ-CP
    - Các văn bản pháp luật liên quan
    - Bài báo chính thống
    """)

    st.markdown("---")

    # Statistics
    st.subheader("📊 Thống kê")
    st.metric(
        "Tổng số câu hỏi",
        st.session_state.total_queries,
        help="Số lượng câu hỏi đã đặt trong phiên này"
    )

    st.metric(
        "Đoạn hội thoại",
        len([m for m in st.session_state.messages if m["role"] == "user"]),
        help="Số lượt hỏi đáp"
    )

    st.markdown("---")

    # Clear conversation button
    if st.button("🗑️ Xóa hội thoại", use_container_width=True):
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "Hội thoại đã được xóa. Bạn có thể bắt đầu hỏi mới!"
            }
        ]
        st.session_state.conversation_history = []
        st.session_state.total_queries = 0
        st.session_state.pending_prompt = None
        st.rerun()


# =============================================================================
# MAIN CHAT INTERFACE
# =============================================================================

st.title("💬 Hỏi Đáp Pháp Luật Ma Túy & Tin Tức Liên Quan")
st.markdown("---")

# Display chat messages
for msg_idx, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        # Show sources if available
        if "sources" in message and message["sources"]:
            with st.expander(f"📚 Xem {len(message['sources'])} nguồn tham khảo"):
                render_sources(
                    message["sources"],
                    key_prefix=f"history_{msg_idx}_{stable_key(message.get('content', ''))}",
                    preview_chars=500,
                )


# =============================================================================
# USER INPUT HANDLING
# =============================================================================

if prompt := st.chat_input("Nhập câu hỏi của bạn..."):
    prompt = prompt.strip()
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.markdown(prompt)

        st.session_state.pending_prompt = prompt
        process_pending_prompt()


# =============================================================================
# FOOTER
# =============================================================================

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray; font-size: 0.8em;'>
<p>💡 <b>Mẹo:</b> Bạn có thể đặt câu hỏi tiếp theo dựa trên câu trả lời trước đó. Chatbot sẽ ghi nhớ ngữ cảnh hội thoại.</p>
<p>📖 <b>Dữ liệu:</b> Cập nhật từ Luật Phòng chống ma tuý 2021 và các tin tức uy tín</p>
</div>
""", unsafe_allow_html=True)

# RAG Chatbot - Hỏi Đáp Pháp Luật Ma Túy

Chatbot thông minh dựa trên RAG (Retrieval-Augmented Generation) để trả lời câu hỏi về pháp luật ma túy và tin tức liên quan tại Việt Nam.

## 🎯 Tính Năng

- **💬 Giao diện Chat**: Trải nghiệm trò chuyện tự nhiên với Streamlit
- **📚 Trích dẫn nguồn**: Mỗi câu trả lời đều có citation từ nguồn chính xác
- **🔄 Memory hội thoại**: Hỗ trợ câu hỏi tiếp theo dựa trên ngữ cảnh trước đó
- **📊 Hiển thị nguồn**: Xem chi tiết các tài liệu tham khảo với độ liên quan
- **⚖️ Đa nguồn dữ liệu**: Pháp luật (Luật, Nghị định) + Tin tức (Báo chí)

## 🏗️ Kiến Trúc

```
User Question (Streamlit UI)
    ↓
Conversation Memory (Last 4 exchanges)
    ↓
RAG Pipeline (Task 9)
    ├→ Hybrid Search (Semantic + Lexical)
    ├→ Reranking (Jina reranker v3 / OpenAI fallback)
    └→ PageIndex local/cloud fallback (if low score)
    ↓
Generation with Citation (Task 10)
    ├→ Context Reordering (avoid lost-in-middle)
    ├→ OpenAI Responses API generation
    └→ Citation Formatting
    ↓
Response + Sources (Streamlit Display)
```

## 🚀 Cách Chạy

### 1. Cài đặt dependencies

```bash
pip install -r requirements.txt
```

### 2. Setup environment variables

Tạo file `.env` từ `.env.example`:

```bash
cp .env.example .env
```

Edit `.env` và thêm API keys:

```env
OPENAI_API_KEY=your_openai_api_key_here
PAGEINDEX_API_KEY=your_pageindex_api_key_here  # Optional
```

### 3. Chạy chatbot

```bash
streamlit run group_project/app.py
```

Chatbot sẽ mở tại: `http://localhost:8501` theo mặc định của Streamlit, hoặc port bạn chỉ định bằng `--server.port`.

## 💡 Cách Sử Dụng

### Câu hỏi mẫu:

**Pháp luật:**
- "Hình phạt cho tội tàng trữ trái phép chất ma túy là gì?"
- "Luật phòng chống ma tuý 2021 quy định gì về cai nghiện?"
- "Quy trình quản lý người sử dụng trái phép chất ma túy như thế nào?"

**Tin tức:**
- "Những nghệ sĩ nào đã bị bắt vì liên quan đến ma túy?"
- "Tin tức mới nhất về ma túy trong showbiz Việt Nam"

**Follow-up questions:**
- "Vậy cụ thể điều khoản nào nói về việc này?"
- "Có bao nhiêu trường hợp như vậy trong tin tức?"

### Tính năng giao diện:

1. **Sidebar**:
   - Xem thống kê số câu hỏi
   - Đọc thông tin về chatbot
   - Nút xóa hội thoại để bắt đầu mới

2. **Chat area**:
   - Nhập câu hỏi ở input box
   - Xem câu trả lời với citation
   - Click "Xem N nguồn tham khảo" để xem chi tiết

3. **Source expander**:
   - Xem metadata của từng nguồn
   - Đọc content preview
   - Xem độ liên quan (relevance score)

## 🔧 Technical Details

### Components sử dụng:

- **Task 5**: Semantic Search (OpenAI `text-embedding-3-small` + Qdrant/JSONL fallback)
- **Task 6**: Lexical Search (Vietnamese-aware BM25)
- **Task 7**: Reranking (Jina reranker v3, OpenAI/listwise fallback, offline fallback for tests)
- **Task 8**: PageIndex Vectorless (cloud/local fallback)
- **Task 9**: Retrieval Pipeline (hybrid + fallback logic)
- **Task 10**: Generation with Citation (OpenAI Responses API)

### Conversation Memory:

- Lưu 4 exchange gần nhất (2 Q&A pairs)
- Inject vào prompt khi có follow-up questions
- Giúp chatbot hiểu ngữ cảnh của hội thoại

### Citation System:

- Mỗi factual claim phải có citation: `[Source, Year]`
- Nếu không đủ evidence: "Tôi không thể xác minh thông tin này từ nguồn hiện có"
- Sources hiển thị theo độ liên quan giảm dần

### Reranking Strategy:

- Hybrid search: Semantic + Lexical → RRF fusion
- Nếu hybrid score thấp hơn ngưỡng provider-specific → PageIndex fallback
- Jina reranker v3 multilingual cho final ranking khi có `JINA_API_KEY`

## 📊 Performance

**Latency:**
- Hybrid search: thường vài giây sau khi cache đã có
- PageIndex fallback: phụ thuộc cloud/local mode
- Generation: phụ thuộc model OpenAI đang cấu hình
- **Total**: thường 5-15 seconds per query trong demo local

**Accuracy:**
- Retrieval: 90%+ (hybrid)
- Citation: 85%+ (LLM dependent)
- Follow-up context: 80%+ (memory-based)

## 🛠️ Troubleshooting

### Chatbot không trả lời:

1. **Check API keys**:
   ```bash
   python -c "import os; from dotenv import load_dotenv; load_dotenv(); print(bool(os.getenv('OPENAI_API_KEY')))"
   ```

2. **Check vector store**:
   - Verify `data/index/` exists and contains data
   - Re-run Task 4 nếu cần

3. **Check PageIndex**:
   - Verify `data/index/pageindex_*.json` exists
   - Re-run Task 8 nếu cần

### Lỗi encoding:

```bash
# Windows: Set UTF-8 encoding
set PYTHONIOENCODING=utf-8
streamlit run group_project/app.py
```

### Slow response:

- Reduce `top_k` in Task 10 (currently 5)
- Use cached embeddings/index files in `data/index/`
- Use offline fallbacks during tests, or keep Jina/PageIndex keys configured for best retrieval

## 📝 Next Steps

### Improvements:

1. **Add feedback mechanism**: User có thể rate câu trả lời
2. **Add search history**: Lưu lịch sử tìm kiếm
3. **Add export**: Export hội thoại sang PDF/Markdown
4. **Add analytics**: Track common questions, performance metrics
5. **Multi-language**: Hỗ trợ tiếng Anh (hiện tại chỉ tiếng Việt)

### Deployment:

1. **Streamlit Cloud**: Free deployment option
2. **Hugging Face Spaces**: Alternative với GPU support
3. **Docker**: Containerize cho production

```bash
# Docker example
docker build -t rag-chatbot .
docker run -p 8501:8501 rag-chatbot
```

## 🤝 Contributing

Để phát triển thêm:

1. Thêm features mới
2. Improve retrieval accuracy
3. Add more data sources
4. Optimize performance

## 📄 License

Educational purpose for VinAI cohort.

---

**Note**: Chatbot này dùng cho mục đích giáo dục. Không thay thế cho tư vấn pháp luật chuyên nghiệp.

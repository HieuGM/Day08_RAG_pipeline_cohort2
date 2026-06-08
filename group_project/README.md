# Bài Tập Nhóm - RAG Chatbot Pháp Luật Ma Túy

## Mục Tiêu

Nhóm xây dựng chatbot RAG trả lời câu hỏi về pháp luật ma túy Việt Nam và các bài báo liên quan. Sản phẩm tích hợp các task cá nhân từ crawling, chuẩn hóa dữ liệu, chunking/indexing, retrieval, reranking, PageIndex fallback đến generation có citation.

## Deliverables

| Hạng mục | File / lệnh kiểm tra | Trạng thái |
|---|---|---|
| Streamlit chatbot | `group_project/app.py`, `group_project/chatbot_app.py` | Hoàn thành |
| Citation + source display | Task 10 + source expander trong UI | Hoàn thành |
| Follow-up questions | `build_contextual_question()` trong chatbot | Hoàn thành |
| Golden dataset 15+ Q&A | `group_project/evaluation/golden_dataset.json` | 16 cases |
| Evaluation pipeline | `group_project/evaluation/eval_pipeline.py` | Hoàn thành |
| A/B comparison | Hybrid rerank vs hybrid no rerank | Hoàn thành |
| Evaluation report | `group_project/evaluation/results.md` | Hoàn thành |
| Automated tests | `pytest tests/ -v` | Hoàn thành |

## Kiến Trúc Hệ Thống

```text
Streamlit UI
  -> conversation memory
  -> Task 10 generate_with_citation
      -> Task 9 retrieve
          -> Task 5 semantic search
              -> OpenAI text-embedding-3-small + Qdrant/JSONL fallback
          -> Task 6 lexical search
              -> Vietnamese-aware BM25
          -> Task 7 fusion + reranking
              -> RRF + Jina reranker v3 when JINA_API_KEY exists
              -> OpenAI/listwise fallback
              -> offline fallback during tests
          -> Task 8 PageIndex vectorless fallback
              -> PageIndex cloud when configured
              -> local structural fallback
      -> context reorder to reduce lost-in-the-middle
      -> OpenAI generation with citation
  -> answer + source previews
```

## Cấu Hình Chính

| Thành phần | Lựa chọn |
|---|---|
| Chunking | Legal-aware structural chunks + SemanticChunker-style split for long sections |
| Embedding | OpenAI `text-embedding-3-small`, 1536 dimensions |
| Vector store | Qdrant local path, JSONL cosine fallback |
| Lexical | BM25 with Vietnamese normalization and phrase boosts |
| Reranking | Jina reranker v3 by default when key exists; OpenAI/listwise fallback; offline fallback in tests |
| Vectorless fallback | PageIndex API when ready; local PageIndex-style structural search otherwise |
| Generation | OpenAI Responses API, low temperature, citations required |

## Cách Chạy

```bash
pip install -r requirements.txt
```

Tạo `.env` từ `.env.example` và điền key cần dùng:

```bash
cp .env.example .env
```

Chạy chatbot:

```bash
streamlit run group_project/app.py
```

Chạy evaluation:

```bash
python group_project/evaluation/eval_pipeline.py
```

Chạy DeepEval adapter nếu đã cài `deepeval` và có model credentials:

```bash
EVAL_FRAMEWORK=deepeval python group_project/evaluation/eval_pipeline.py
```

PowerShell:

```powershell
$env:EVAL_FRAMEWORK="deepeval"; python group_project/evaluation/eval_pipeline.py
```

Chạy toàn bộ test:

```bash
pytest tests/ -v
```

## Evaluation

Golden dataset có 16 câu hỏi bao phủ:

- Định nghĩa và điều khoản trong Luật Phòng chống ma túy 2021
- Nghị định 105/2021/NĐ-CP
- Nghị định 28/2026/NĐ-CP
- Bài báo về Chi Dân, An Tây, Trúc Phương, Andrea Aybar, Lệ Hằng

Kết quả hiện tại trong `group_project/evaluation/results.md`:

| Metric | Hybrid + rerank | Hybrid no rerank |
|---|---:|---:|
| Faithfulness | 0.996 | 0.999 |
| Answer Relevance | 0.912 | 0.880 |
| Context Recall | 0.861 | 0.871 |
| Context Precision | 0.988 | 0.988 |
| Average | 0.939 | 0.934 |

## Query Demo

Các câu nên dùng khi demo:

- `ma túy là gì`
- `Chất gây nghiện là gì?`
- `Luật Phòng chống ma túy 2021 nghiêm cấm những hành vi nào?`
- `Chi Dân An Tây bị điều tra về tội gì?`
- `Andrea Aybar bị tình nghi liên quan đến vấn đề gì?`
- Bad query: `công thức nấu phở bò ngon`

## Phân Công Công Việc

| Thành viên | MSSV | Nhiệm vụ | Trạng thái |
|---|---|---|---|
| Nguyễn Tùng Lâm | 2A202600555 | Crawling dữ liệu, chuẩn hóa legal/news | Hoàn thành |
| Cao Đặng Quốc Vương | 2A202600738 | Chunking, indexing, semantic search | Hoàn thành |
| Đỗ Phan Hà | 2A202600543 | Lexical search, fusion, reranking | Hoàn thành |
| Giáp Minh Hiếu | 2A202600667 | PageIndex vectorless fallback | Hoàn thành |
| Nguyễn Thành Vinh | 2A202600971 | Generation có citation, prompt/reorder | Hoàn thành |
| Đỗ Đức Anh | 2A202600976 | Streamlit chatbot, evaluation, docs/tests | Hoàn thành |

## Ghi Chú Demo

- App entrypoint là `group_project/app.py`; không chạy `chatbot_app.py` trực tiếp khi demo để tránh khác hành vi entrypoint.
- Khi không có API key ngoài, test vẫn chạy nhờ offline fallback.
- Khi có `JINA_API_KEY`, Task 7 ưu tiên Jina reranker v3.
- Khi có `PAGEINDEX_API_KEY` và manifest sẵn sàng, Task 8 ưu tiên PageIndex cloud; nếu không có thì dùng local structural fallback.

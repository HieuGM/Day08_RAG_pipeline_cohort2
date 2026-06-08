# RAG Evaluation Results

## Framework sử dụng

Default run: deterministic heuristic retrieval-grounding evaluation for offline demo readiness.

Framework adapter: `EVAL_FRAMEWORK=deepeval python group_project/evaluation/eval_pipeline.py` runs the DeepEval adapter when `deepeval` and model credentials are installed.

Dataset: 16 golden Q&A cases.

## Overall Scores

| Metric | Config A (hybrid + rerank) | Config B (hybrid no rerank) | Delta |
|--------|---------------------------|-----------------------------|-------|
| Faithfulness | 0.996 | 0.999 | -0.003 |
| Answer Relevance | 0.912 | 0.880 | +0.032 |
| Context Recall | 0.861 | 0.871 | -0.010 |
| Context Precision | 0.988 | 0.988 | +0.000 |
| Average | 0.939 | 0.934 | +0.005 |

## A/B Comparison Analysis

**Config A:** hybrid semantic + lexical retrieval, RRF fusion, reranking, PageIndex fallback.

**Config B:** hybrid semantic + lexical retrieval, RRF fusion, no reranking, PageIndex fallback.

**Kết luận:** Config A is preferred when reranking is available because it improves ordering for mixed legal/news queries. Config B remains useful as a faster fallback for low-latency demos.

## Worst Performers (Bottom 3)

| # | Question | Avg | Faithfulness | Relevance | Recall | Precision | Source Count |
|---|----------|-----|--------------|-----------|--------|-----------|--------------|
| 1 | Chất hướng thần được định nghĩa như thế nào? | 0.754 | 1.000 | 0.714 | 0.500 | 0.800 | 5 |
| 2 | Nghị định 28/2026 ban hành nội dung gì liên quan đến ma túy? | 0.887 | 1.000 | 0.800 | 0.750 | 1.000 | 5 |
| 3 | Cá nhân và gia đình có trách nhiệm gì trong phòng, chống ma túy? | 0.903 | 1.000 | 0.700 | 0.913 | 1.000 | 5 |

## Recommendations

### Cải tiến 1
**Action:** Add Bộ luật Hình sự drug-crime articles if the chatbot must answer criminal penalty questions.  
**Expected impact:** Better recall for penalty-specific questions.

### Cải tiến 2
**Action:** Run the included DeepEval adapter after installing `deepeval` for model-judged grading.  
**Expected impact:** More defensible faithfulness and answer-relevance scores for formal review.

### Cải tiến 3
**Action:** Add more golden questions from real user logs after chatbot demo.  
**Expected impact:** Better coverage of follow-up questions and news/legal ambiguity.

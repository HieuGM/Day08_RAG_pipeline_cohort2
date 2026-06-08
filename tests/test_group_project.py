"""
Group project submission tests.

These tests focus on the deliverables required in group_project/README.md:
chatbot app, citation generation, golden dataset, A/B evaluation, and report.
They avoid external API calls so the suite stays stable on grading machines.
"""

from __future__ import annotations

import json
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch


PROJECT_DIR = Path(__file__).parent.parent
GROUP_DIR = PROJECT_DIR / "group_project"
EVAL_DIR = GROUP_DIR / "evaluation"

sys.path.insert(0, str(PROJECT_DIR))


class TestGroupDataset(unittest.TestCase):
    def test_golden_dataset_has_required_shape(self):
        dataset_path = EVAL_DIR / "golden_dataset.json"
        self.assertTrue(dataset_path.exists(), "Missing golden_dataset.json")

        data = json.loads(dataset_path.read_text(encoding="utf-8"))
        self.assertGreaterEqual(len(data), 15, "Golden dataset must contain at least 15 cases")

        required_keys = {"question", "expected_answer", "expected_context"}
        for index, item in enumerate(data, start=1):
            self.assertTrue(required_keys <= item.keys(), f"Case {index} missing required keys")
            for key in required_keys:
                self.assertIsInstance(item[key], str, f"Case {index}.{key} must be a string")
                self.assertGreater(len(item[key].strip()), 10, f"Case {index}.{key} is too short")

    def test_dataset_covers_legal_and_news_questions(self):
        data = json.loads((EVAL_DIR / "golden_dataset.json").read_text(encoding="utf-8"))
        joined_questions = "\n".join(item["question"].lower() for item in data)
        self.assertIn("luật", joined_questions)
        self.assertTrue(any(term in joined_questions for term in ("chi dân", "an tây", "andrea", "lệ hằng")))


class TestGroupEvaluation(unittest.TestCase):
    def test_compare_configs_returns_two_configs_and_four_metrics(self):
        from group_project.evaluation import eval_pipeline

        dataset = [
            {
                "question": "Luật Phòng chống ma túy định nghĩa chất ma túy là gì?",
                "expected_answer": "Chất ma túy là chất gây nghiện hoặc chất hướng thần.",
                "expected_context": "Luật Phòng chống ma túy 2021, Điều 2",
            },
            {
                "question": "Chi Dân An Tây bị điều tra về tội gì?",
                "expected_answer": "Tổ chức sử dụng trái phép chất ma túy.",
                "expected_context": "Tuổi Trẻ, 2024",
            },
        ]

        def fake_retrieve(question: str, top_k: int = 5, use_reranking: bool = True):
            marker = "reranked" if use_reranking else "no-rerank"
            return [
                {
                    "content": (
                        f"{question}. Chất ma túy là chất gây nghiện hoặc chất hướng thần. "
                        f"Tổ chức sử dụng trái phép chất ma túy. {marker}"
                    ),
                    "score": 1.0,
                    "metadata": {"title": "Luật Phòng chống ma túy 2021", "section": "Điều 2"},
                    "source": "hybrid",
                }
            ][:top_k]

        with patch.object(eval_pipeline, "retrieve", fake_retrieve):
            comparison = eval_pipeline.compare_configs(None, dataset)

        self.assertEqual(set(comparison), {"hybrid_rerank", "hybrid_no_rerank"})
        for config_result in comparison.values():
            summary = config_result["summary"]
            for metric in ("faithfulness", "answer_relevance", "context_recall", "context_precision"):
                self.assertIn(metric, summary)
                self.assertGreaterEqual(summary[metric], 0.0)
                self.assertLessEqual(summary[metric], 1.0)

    def test_results_report_contains_required_sections(self):
        report_path = EVAL_DIR / "results.md"
        self.assertTrue(report_path.exists(), "Missing evaluation results.md")
        report = report_path.read_text(encoding="utf-8")
        for required in (
            "Overall Scores",
            "A/B Comparison",
            "Worst Performers",
            "Recommendations",
            "Faithfulness",
            "Context Recall",
            "Context Precision",
        ):
            self.assertIn(required, report)


class TestGroupChatbot(unittest.TestCase):
    def test_streamlit_entrypoint_executes_app_on_each_rerun(self):
        app_path = GROUP_DIR / "app.py"
        content = app_path.read_text(encoding="utf-8")
        self.assertIn("runpy.run_path", content)
        self.assertNotIn("from chatbot_app import *", content)

    def test_streamlit_app_renders_without_exception(self):
        try:
            from streamlit.testing.v1 import AppTest
        except Exception as exc:
            self.skipTest(f"Streamlit testing API unavailable: {exc}")

        at = AppTest.from_file(str(GROUP_DIR / "app.py"))
        at.run(timeout=60)
        self.assertEqual(len(at.exception), 0)
        self.assertGreaterEqual(len(at.chat_input), 1)

    def test_generation_fallback_returns_cited_answer_from_context(self):
        from src.task10_generation import generate_with_citation

        context = [
            {
                "content": "Chất ma túy là chất gây nghiện, chất hướng thần do Chính phủ ban hành danh mục.",
                "score": 1.0,
                "metadata": {
                    "title": "Luật Phòng chống ma túy 2021",
                    "section": "Điều 2. Giải thích từ ngữ",
                    "source_path": "data/landing/legal/luat-phong-chong-ma-tuy-2021.pdf",
                },
                "source": "hybrid",
            }
        ]

        with patch.dict(os.environ, {"OPENAI_API_KEY": ""}, clear=False):
            result = generate_with_citation("Chất ma túy là gì?", context_chunks=context)

        self.assertIn("answer", result)
        self.assertIn("[Luật Phòng chống ma túy 2021, Điều 2]", result["answer"])
        self.assertGreaterEqual(len(result.get("sources", [])), 1)

    def test_bad_query_with_no_context_returns_insufficient_evidence(self):
        from src.task10_generation import INSUFFICIENT_EVIDENCE_MESSAGE, generate_with_citation

        result = generate_with_citation("Công thức nấu phở bò ngon?", context_chunks=[])
        self.assertEqual(result["answer"], INSUFFICIENT_EVIDENCE_MESSAGE)
        self.assertEqual(result["sources"], [])


if __name__ == "__main__":
    unittest.main(verbosity=2)

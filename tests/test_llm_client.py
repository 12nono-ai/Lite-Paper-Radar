from datetime import datetime
import json
import unittest
from unittest import mock

from arxiv_llm_watch.config import AppConfig
from arxiv_llm_watch.llm_client import LLMClient
from arxiv_llm_watch.models import Paper


class _FakeHttpResponse:
    def __init__(self, payload: dict) -> None:
        self.payload = payload

    def read(self) -> bytes:
        return json.dumps(self.payload).encode("utf-8")

    def __enter__(self) -> "_FakeHttpResponse":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None


class LLMClientTests(unittest.TestCase):
    def test_openai_compatible_provider_posts_chat_completion(self) -> None:
        captured = {}

        def fake_urlopen(req, timeout=0):
            headers = {key.lower(): value for key, value in req.header_items()}
            captured["url"] = req.full_url
            captured["authorization"] = headers.get("authorization")
            captured["custom_header"] = headers.get("x-test")
            captured["timeout"] = timeout
            captured["body"] = json.loads(req.data.decode("utf-8"))
            return _FakeHttpResponse(
                {
                    "choices": [
                        {
                            "message": {
                                "content": json.dumps(
                                    {
                                        "is_llm_related": True,
                                        "llm_score": 0.93,
                                        "relevance_reason": "Relevant",
                                        "topics": ["reasoning"],
                                        "summary_zh": "中文摘要",
                                        "summary_en": "English summary",
                                        "background": {"zh": "背景", "en": "Background"},
                                        "problem": {"zh": "问题", "en": "Problem"},
                                        "method": {"zh": "方法", "en": "Method"},
                                        "findings": {"zh": "结果", "en": "Findings"},
                                        "limitations": {"zh": "局限", "en": "Limitations"},
                                    }
                                )
                            }
                        }
                    ]
                }
            )

        config = AppConfig(
            llm_provider="openai_compatible",
            llm_api_key="test-key",
            llm_base_url="https://api.example.com/v1",
            llm_model="gpt-test",
            llm_api_path="/chat/completions",
            llm_headers={"X-Test": "1"},
            arxiv_categories=["cs.CL"],
            arxiv_keywords=[],
            arxiv_max_results=10,
            lookback_days=7,
            topic_recent_days=7,
            topic_baseline_days=7,
            topic_limit=5,
            report_paper_limit=5,
            analysis_limit_per_run=6,
            data_dir=mock.MagicMock(),
            reports_dir=mock.MagicMock(),
            db_path=mock.MagicMock(),
            llm_temperature=0.2,
        )
        client = LLMClient(config)
        paper = Paper(
            entry_id="id-1",
            title="Reasoning with LLMs",
            summary="This paper studies reasoning in LLMs.",
            published=datetime(2026, 3, 10),
            updated=datetime(2026, 3, 10),
            primary_category="cs.CL",
            categories=["cs.CL"],
            authors=["Author"],
        )

        with mock.patch("arxiv_llm_watch.llm_client.request.urlopen", side_effect=fake_urlopen):
            analysis = client.classify_and_summarize(paper)

        self.assertEqual(captured["url"], "https://api.example.com/v1/chat/completions")
        self.assertEqual(captured["authorization"], "Bearer test-key")
        self.assertEqual(captured["custom_header"], "1")
        self.assertEqual(captured["body"]["model"], "gpt-test")
        self.assertEqual(captured["body"]["temperature"], 0.2)
        self.assertEqual(captured["body"]["messages"][0]["role"], "system")
        self.assertTrue(analysis.is_llm_related)
        self.assertEqual(analysis.summary.zh, "中文摘要")
        self.assertEqual(analysis.method.en, "Method")


if __name__ == "__main__":
    unittest.main()

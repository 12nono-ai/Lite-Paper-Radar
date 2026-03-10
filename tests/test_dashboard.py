from datetime import datetime
from pathlib import Path
import tempfile
import unittest

from arxiv_llm_watch.config import AppConfig
from arxiv_llm_watch.dashboard import DashboardController, INDEX_HTML, PAPER_HTML
from arxiv_llm_watch.models import Paper, PaperAnalysis, SectionText
from arxiv_llm_watch.storage import Storage


class DashboardTests(unittest.TestCase):
    def test_build_state_returns_dashboard_payload(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            reports_dir = root / "reports"
            reports_dir.mkdir()
            report_path = reports_dir / "daily_20260308_101500.md"
            report_path.write_text("# report", encoding="utf-8")

            config = AppConfig(
                ark_api_key="key",
                ark_base_url="https://ark.cn-beijing.volces.com/api/v3",
                ark_model="demo-model",
                arxiv_categories=["cs.CL"],
                arxiv_keywords=["reasoning", "alignment"],
                arxiv_max_results=10,
                lookback_days=7,
                topic_recent_days=7,
                topic_baseline_days=7,
                topic_limit=5,
                report_paper_limit=5,
                analysis_limit_per_run=6,
                data_dir=root / "data",
                reports_dir=reports_dir,
                db_path=root / "data" / "test.db",
                llm_temperature=0.2,
            )
            config.ensure_directories()

            storage = Storage(config.db_path)
            storage.initialize()
            try:
                paper = Paper(
                    entry_id="http://arxiv.org/abs/1234.5678",
                    title="Test LLM Paper",
                    summary="This work studies LLM alignment.",
                    published=datetime(2026, 3, 8),
                    updated=datetime(2026, 3, 8),
                    primary_category="cs.CL",
                    categories=["cs.CL"],
                    authors=["A"],
                )
                storage.save_papers([paper], "2026-03-08T00:00:00+00:00")
                storage.save_analysis(
                    paper.entry_id,
                    PaperAnalysis(
                        is_llm_related=True,
                        relevance_reason="Relevant",
                        llm_score=0.95,
                        topics=["alignment"],
                        summary=SectionText(zh="中文", en="English"),
                        background=SectionText(zh="背景", en="Background"),
                        problem=SectionText(zh="问题", en="Problem"),
                        method=SectionText(zh="方法", en="Method"),
                        findings=SectionText(zh="结果", en="Findings"),
                        limitations=SectionText(zh="局限", en="Limits"),
                    ),
                )
                run_id = storage.start_run("test", {"analysis_limit": 1})
                storage.finish_run(run_id, status="success", fetched_count=1, analysis_batch_count=1, analyzed_count=1)
            finally:
                storage.close()

            controller = DashboardController(config)
            state = controller.build_state()

            self.assertEqual(state["config"]["analysis_limit_per_run"], 6)
            self.assertEqual(state["config"]["query_keywords"], ["reasoning", "alignment"])
            self.assertIn("cs.CL", state["available_categories"])
            self.assertEqual(len(state["analyzed_papers"]), 1)
            self.assertEqual(state["analyzed_papers"][0]["title"], "Test LLM Paper")
            self.assertIn("alignment & safety", state["analyzed_papers"][0]["tracked_topics"])
            self.assertEqual(state["category_mix"][0]["category"], "cs.CL")
            self.assertEqual(state["reports"][0]["name"], "daily_20260308_101500.md")
            self.assertEqual(state["filter_facets"]["total"], 1)
            self.assertEqual(len(state["run_history"]), 1)
            self.assertEqual(state["topic_trends"][0]["momentum_label"], "New (0 -> 1)")

            search = controller.search_papers(query="alignment", status="analyzed", category="all", limit=5)
            self.assertEqual(len(search["papers"]), 1)
            self.assertEqual(search["papers"][0]["analysis_status"], "analyzed")
            self.assertEqual(search["total"], 1)
            self.assertEqual(search["page"], 1)
            self.assertEqual(search["sort"], "published_desc")
            self.assertEqual(search["tracked_topic"], "")

            detail = controller.get_paper("http://arxiv.org/abs/1234.5678")
            self.assertIsNotNone(detail)
            self.assertEqual(detail["background_en"], "Background")
            self.assertEqual(detail["related_papers"], [])

    def test_dashboard_routes_return_separate_views(self) -> None:
        overview = INDEX_HTML.replace("__PAGE_VIEW__", "overview")
        papers = INDEX_HTML.replace("__PAGE_VIEW__", "papers")
        reports = INDEX_HTML.replace("__PAGE_VIEW__", "reports")

        self.assertIn('body data-view="overview"', overview)
        self.assertIn('body data-view="papers"', papers)
        self.assertIn('body data-view="reports"', reports)
        self.assertIn('href="/papers"', overview)
        self.assertIn("总览 - LLM Paper Radar", overview)
        self.assertIn("论文详情 - LLM Paper Radar", PAPER_HTML)
        self.assertIn("Loading paper detail", PAPER_HTML)
        self.assertIn("run.history.in_progress", overview)
        self.assertIn('id="run-history-fold"', overview)
        self.assertIn('id="generate-period-7-button"', reports)
        self.assertIn('id="generate-period-30-button"', reports)
        self.assertIn('id="generate-custom-period-report-button"', reports)
        self.assertNotIn('id="period-mode"', reports)

    def test_controller_recovers_orphaned_running_records(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config = AppConfig(
                ark_api_key="key",
                ark_base_url="https://ark.cn-beijing.volces.com/api/v3",
                ark_model="demo-model",
                arxiv_categories=["cs.CL"],
                arxiv_keywords=[],
                arxiv_max_results=10,
                lookback_days=7,
                topic_recent_days=7,
                topic_baseline_days=7,
                topic_limit=5,
                report_paper_limit=5,
                analysis_limit_per_run=6,
                data_dir=root / "data",
                reports_dir=root / "reports",
                db_path=root / "data" / "test.db",
                llm_temperature=0.2,
            )
            config.ensure_directories()
            storage = Storage(config.db_path)
            storage.initialize()
            try:
                storage.start_run("dashboard", {"analysis_limit": 1})
            finally:
                storage.close()

            controller = DashboardController(config)
            state = controller.build_state()
            self.assertEqual(state["run_history"][0]["status"], "failed")
            self.assertIn("未完成", state["run_history"][0]["error_message"])

    def test_controller_generates_period_report(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config = AppConfig(
                ark_api_key="key",
                ark_base_url="https://ark.cn-beijing.volces.com/api/v3",
                ark_model="demo-model",
                arxiv_categories=["cs.CL"],
                arxiv_keywords=[],
                arxiv_max_results=10,
                lookback_days=7,
                topic_recent_days=7,
                topic_baseline_days=7,
                topic_limit=5,
                report_paper_limit=5,
                analysis_limit_per_run=6,
                data_dir=root / "data",
                reports_dir=root / "reports",
                db_path=root / "data" / "test.db",
                llm_temperature=0.2,
            )
            config.ensure_directories()
            storage = Storage(config.db_path)
            storage.initialize()
            try:
                paper = Paper(
                    entry_id="id-period",
                    title="Reasoning period paper",
                    summary="Reasoning",
                    published=datetime(2026, 3, 8),
                    updated=datetime(2026, 3, 8),
                    primary_category="cs.CL",
                    categories=["cs.CL"],
                    authors=["author"],
                )
                storage.save_papers([paper], "2026-03-10T00:00:00+00:00")
                storage.save_analysis(
                    paper.entry_id,
                    PaperAnalysis(
                        is_llm_related=True,
                        relevance_reason="Relevant",
                        llm_score=0.91,
                        topics=["reasoning"],
                        summary=SectionText(zh="中文", en="English"),
                    ),
                )
            finally:
                storage.close()

            controller = DashboardController(config)
            summary = controller.generate_period_report(
                start_date="2026-03-03",
                end_date="2026-03-09",
                paper_limit=5,
            )
            self.assertTrue(summary["report_name"].startswith("period_20260303_20260309_"))
            self.assertEqual(summary["paper_count"], 1)


if __name__ == "__main__":
    unittest.main()

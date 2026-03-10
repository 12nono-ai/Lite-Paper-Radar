from datetime import datetime
from pathlib import Path
import tempfile
import unittest

from arxiv_llm_watch.models import Paper, PaperAnalysis, SectionText
from arxiv_llm_watch.storage import Storage


class StorageTests(unittest.TestCase):
    def test_list_pending_papers_respects_limit(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "test.db"
            storage = Storage(db_path)
            storage.initialize()
            try:
                papers = [
                    Paper(
                        entry_id=f"id-{index}",
                        title=f"title-{index}",
                        summary="summary",
                        published=datetime(2026, 3, index + 1),
                        updated=datetime(2026, 3, index + 1),
                        primary_category="cs.CL",
                        categories=["cs.CL"],
                        authors=["author"],
                    )
                    for index in range(10)
                ]
                storage.save_papers(papers, "2026-03-08T00:00:00+00:00")

                pending = storage.list_pending_papers(limit=6)
                self.assertEqual(len(pending), 6)
                self.assertEqual(pending[0].entry_id, "id-9")
                self.assertEqual(pending[-1].entry_id, "id-4")
            finally:
                storage.close()

    def test_search_papers_filters_by_status_and_query(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "test.db"
            storage = Storage(db_path)
            storage.initialize()
            try:
                pending_paper = Paper(
                    entry_id="id-pending",
                    title="Pending alignment paper",
                    summary="Studies alignment.",
                    published=datetime(2026, 3, 1),
                    updated=datetime(2026, 3, 1),
                    primary_category="cs.CL",
                    categories=["cs.CL"],
                    authors=["author"],
                )
                analyzed_paper = Paper(
                    entry_id="id-analyzed",
                    title="Analyzed reasoning paper",
                    summary="Studies reasoning.",
                    published=datetime(2026, 3, 2),
                    updated=datetime(2026, 3, 2),
                    primary_category="cs.AI",
                    categories=["cs.AI"],
                    authors=["author"],
                )
                storage.save_papers([pending_paper, analyzed_paper], "2026-03-08T00:00:00+00:00")
                storage.save_analysis(
                    analyzed_paper.entry_id,
                    PaperAnalysis(
                        is_llm_related=True,
                        relevance_reason="Relevant",
                        llm_score=0.9,
                        topics=["reasoning"],
                        summary=SectionText(zh="中文", en="English"),
                    ),
                )

                results = storage.search_papers(query="reasoning", status="analyzed", category="all", limit=10)
                self.assertEqual(len(results), 1)
                self.assertEqual(results[0]["entry_id"], "id-analyzed")
                self.assertIn("reasoning", results[0]["tracked_topics"])
            finally:
                storage.close()

    def test_search_papers_matches_tracked_topics(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "test.db"
            storage = Storage(db_path)
            storage.initialize()
            try:
                paper = Paper(
                    entry_id="id-mech",
                    title="Attention sinks in language models",
                    summary="We analyze activation spikes and internal mechanisms in LLMs.",
                    published=datetime(2026, 3, 3),
                    updated=datetime(2026, 3, 3),
                    primary_category="cs.CL",
                    categories=["cs.CL"],
                    authors=["author"],
                )
                storage.save_papers([paper], "2026-03-08T00:00:00+00:00")
                storage.save_analysis(
                    paper.entry_id,
                    PaperAnalysis(
                        is_llm_related=True,
                        relevance_reason="Relevant",
                        llm_score=0.92,
                        topics=["attention sink research"],
                        summary=SectionText(zh="中文", en="English"),
                    ),
                )

                results = storage.search_papers(
                    query="mechanistic interpretability",
                    status="analyzed",
                    category="all",
                    limit=10,
                )
                self.assertEqual(len(results), 1)
                self.assertEqual(results[0]["entry_id"], "id-mech")
            finally:
                storage.close()

    def test_search_papers_supports_offset_pagination(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "test.db"
            storage = Storage(db_path)
            storage.initialize()
            try:
                papers = [
                    Paper(
                        entry_id=f"id-page-{index}",
                        title=f"Reasoning paper {index}",
                        summary="Studies reasoning in LLMs.",
                        published=datetime(2026, 3, index + 1),
                        updated=datetime(2026, 3, index + 1),
                        primary_category="cs.CL",
                        categories=["cs.CL"],
                        authors=["author"],
                    )
                    for index in range(5)
                ]
                storage.save_papers(papers, "2026-03-08T00:00:00+00:00")
                for paper in papers:
                    storage.save_analysis(
                        paper.entry_id,
                        PaperAnalysis(
                            is_llm_related=True,
                            relevance_reason="Relevant",
                            llm_score=0.9,
                            topics=["reasoning"],
                            summary=SectionText(zh="中文", en="English"),
                        ),
                    )

                first_page = storage.search_papers(query="reasoning", status="analyzed", category="all", limit=2, offset=0)
                second_page = storage.search_papers(query="reasoning", status="analyzed", category="all", limit=2, offset=2)
                self.assertEqual(len(first_page), 2)
                self.assertEqual(len(second_page), 2)
                self.assertNotEqual(first_page[0]["entry_id"], second_page[0]["entry_id"])
                self.assertEqual(storage.count_search_papers(query="reasoning", status="analyzed", category="all"), 5)
            finally:
                storage.close()

    def test_search_papers_supports_sorting_and_related_lookup(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "test.db"
            storage = Storage(db_path)
            storage.initialize()
            try:
                papers = [
                    Paper(
                        entry_id="id-score-low",
                        title="Judge paper A",
                        summary="Studies LLM judges.",
                        published=datetime(2026, 3, 2),
                        updated=datetime(2026, 3, 2),
                        primary_category="cs.AI",
                        categories=["cs.AI"],
                        authors=["author"],
                    ),
                    Paper(
                        entry_id="id-score-high",
                        title="Judge paper B",
                        summary="Studies LLM judges and bias.",
                        published=datetime(2026, 3, 3),
                        updated=datetime(2026, 3, 3),
                        primary_category="cs.AI",
                        categories=["cs.AI"],
                        authors=["author"],
                    ),
                    Paper(
                        entry_id="id-related",
                        title="Judge paper C",
                        summary="Studies LLM-as-a-Judge systems.",
                        published=datetime(2026, 3, 4),
                        updated=datetime(2026, 3, 4),
                        primary_category="cs.AI",
                        categories=["cs.AI"],
                        authors=["author"],
                    ),
                ]
                storage.save_papers(papers, "2026-03-08T00:00:00+00:00")
                storage.save_analysis(
                    "id-score-low",
                    PaperAnalysis(
                        is_llm_related=True,
                        relevance_reason="Relevant",
                        llm_score=0.60,
                        topics=["LLM-as-a-Judge"],
                        summary=SectionText(zh="中文", en="English"),
                    ),
                )
                storage.save_analysis(
                    "id-score-high",
                    PaperAnalysis(
                        is_llm_related=True,
                        relevance_reason="Relevant",
                        llm_score=0.95,
                        topics=["LLM-as-a-Judge", "LLM evaluation"],
                        summary=SectionText(zh="中文", en="English"),
                    ),
                )
                storage.save_analysis(
                    "id-related",
                    PaperAnalysis(
                        is_llm_related=True,
                        relevance_reason="Relevant",
                        llm_score=0.90,
                        topics=["LLM evaluation"],
                        summary=SectionText(zh="中文", en="English"),
                    ),
                )

                sorted_results = storage.search_papers(
                    query="judge",
                    status="analyzed",
                    category="all",
                    limit=10,
                    sort="score_desc",
                )
                self.assertEqual(sorted_results[0]["entry_id"], "id-score-high")

                related = storage.find_related_papers("id-score-high", limit=5)
                self.assertTrue(related)
                self.assertEqual(related[0]["entry_id"], "id-related")
                self.assertIn("evaluation & llm as a judge", related[0]["shared_tracked_topics"])
            finally:
                storage.close()

    def test_list_report_papers_between_and_count_status_between(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "test.db"
            storage = Storage(db_path)
            storage.initialize()
            try:
                papers = [
                    Paper(
                        entry_id="id-current",
                        title="Current reasoning paper",
                        summary="Studies reasoning.",
                        published=datetime(2026, 3, 8),
                        updated=datetime(2026, 3, 8),
                        primary_category="cs.CL",
                        categories=["cs.CL"],
                        authors=["author"],
                    ),
                    Paper(
                        entry_id="id-previous",
                        title="Previous alignment paper",
                        summary="Studies alignment.",
                        published=datetime(2026, 3, 2),
                        updated=datetime(2026, 3, 2),
                        primary_category="cs.AI",
                        categories=["cs.AI"],
                        authors=["author"],
                    ),
                ]
                storage.save_papers(papers, "2026-03-10T00:00:00+00:00")
                storage.save_analysis(
                    "id-current",
                    PaperAnalysis(
                        is_llm_related=True,
                        relevance_reason="Relevant",
                        llm_score=0.9,
                        topics=["reasoning"],
                        summary=SectionText(zh="中文", en="English"),
                    ),
                )
                storage.save_analysis(
                    "id-previous",
                    PaperAnalysis(
                        is_llm_related=True,
                        relevance_reason="Relevant",
                        llm_score=0.85,
                        topics=["alignment"],
                        summary=SectionText(zh="中文", en="English"),
                    ),
                )

                current = storage.list_report_papers_between(
                    start=datetime(2026, 3, 3),
                    end=datetime(2026, 3, 10),
                    limit=10,
                )
                counts = storage.count_status_between(
                    start=datetime(2026, 3, 3),
                    end=datetime(2026, 3, 10),
                )
                self.assertEqual(len(current), 1)
                self.assertEqual(current[0]["entry_id"], "id-current")
                self.assertEqual(counts["analyzed"], 1)
                self.assertEqual(counts["total"], 1)
            finally:
                storage.close()

    def test_get_paper_and_category_mix_include_detail_fields(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "test.db"
            storage = Storage(db_path)
            storage.initialize()
            try:
                paper = Paper(
                    entry_id="id-detail",
                    title="Judge alignment study",
                    summary="This work studies LLM judges and alignment.",
                    published=datetime(2026, 3, 4),
                    updated=datetime(2026, 3, 5),
                    primary_category="cs.AI",
                    categories=["cs.AI", "cs.CL"],
                    authors=["author-a", "author-b"],
                )
                storage.save_papers([paper], "2026-03-08T00:00:00+00:00")
                storage.save_analysis(
                    paper.entry_id,
                    PaperAnalysis(
                        is_llm_related=True,
                        relevance_reason="Relevant",
                        llm_score=0.91,
                        topics=["LLM-as-a-Judge"],
                        summary=SectionText(zh="中文", en="English"),
                        background=SectionText(zh="背景", en="Background"),
                    ),
                )

                detail = storage.get_paper("id-detail")
                self.assertIsNotNone(detail)
                self.assertIn("evaluation & llm as a judge", detail["tracked_topics"])
                self.assertEqual(detail["authors"], ["author-a", "author-b"])
                self.assertEqual(detail["background_zh"], "背景")

                category_mix = storage.count_recent_categories(days=10, status="analyzed")
                self.assertEqual(category_mix[0]["category"], "cs.AI")
                self.assertEqual(category_mix[0]["count"], 1)
            finally:
                storage.close()

    def test_feedback_filters_and_requeue_flow(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "test.db"
            storage = Storage(db_path)
            storage.initialize()
            try:
                paper = Paper(
                    entry_id="id-feedback",
                    title="Reasoning control paper",
                    summary="Studies reasoning control in LLMs.",
                    published=datetime(2026, 3, 5),
                    updated=datetime(2026, 3, 5),
                    primary_category="cs.CL",
                    categories=["cs.CL"],
                    authors=["author"],
                )
                storage.save_papers([paper], "2026-03-08T00:00:00+00:00")
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

                storage.update_paper_feedback(
                    paper.entry_id,
                    starred=True,
                    ignored=True,
                    manual_topics=["control", "reasoning"],
                    manual_note="Track this for controllability.",
                )
                updated = storage.get_paper(paper.entry_id)
                self.assertTrue(updated["is_starred"])
                self.assertTrue(updated["is_ignored"])
                self.assertIn("control", updated["tracked_topics"])

                starred = storage.search_papers(flag="starred", status="all", category="all", limit=10)
                self.assertEqual(len(starred), 1)
                manual = storage.search_papers(tracked_topic="control", status="all", category="all", limit=10)
                self.assertEqual(len(manual), 1)

                storage.requeue_paper(paper.entry_id)
                pending = storage.list_pending_papers(limit=10)
                self.assertEqual(len(pending), 1)
                requeued = storage.get_paper(paper.entry_id)
                self.assertFalse(requeued["is_ignored"])
                self.assertEqual(requeued["analysis_status"], "pending")
            finally:
                storage.close()

    def test_facets_and_run_history(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "test.db"
            storage = Storage(db_path)
            storage.initialize()
            try:
                paper = Paper(
                    entry_id="id-run",
                    title="Alignment report paper",
                    summary="Studies alignment in LLMs.",
                    published=datetime(2026, 3, 7),
                    updated=datetime(2026, 3, 7),
                    primary_category="cs.AI",
                    categories=["cs.AI"],
                    authors=["author"],
                )
                storage.save_papers([paper], "2026-03-08T00:00:00+00:00")
                storage.save_analysis(
                    paper.entry_id,
                    PaperAnalysis(
                        is_llm_related=True,
                        relevance_reason="Relevant",
                        llm_score=0.88,
                        topics=["alignment"],
                        summary=SectionText(zh="中文", en="English"),
                    ),
                )
                storage.update_paper_feedback(
                    paper.entry_id,
                    starred=True,
                    manual_topics=["alignment watch"],
                    manual_note="Review later",
                )

                facets = storage.build_search_facets()
                self.assertEqual(facets["total"], 1)
                self.assertEqual(facets["flags"]["starred"], 1)
                self.assertEqual(facets["flags"]["manual"], 1)
                self.assertTrue(facets["tracked_topics"])

                run_id = storage.start_run("test", {"analysis_limit": 2})
                storage.finish_run(
                    run_id,
                    status="success",
                    fetched_count=5,
                    analysis_batch_count=2,
                    analyzed_count=1,
                    filtered_count=1,
                    report_path="reports/daily_test.md",
                )
                history = storage.list_run_history(limit=5)
                self.assertEqual(len(history), 1)
                self.assertEqual(history[0]["source"], "test")
                self.assertEqual(history[0]["fetched_count"], 5)
            finally:
                storage.close()

    def test_mark_interrupted_runs_closes_orphaned_records(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "test.db"
            storage = Storage(db_path)
            storage.initialize()
            try:
                storage.start_run("dashboard", {"analysis_limit": 2})
                updated = storage.mark_interrupted_runs()
                self.assertEqual(updated, 1)
                history = storage.list_run_history(limit=5)
                self.assertEqual(history[0]["status"], "failed")
                self.assertTrue(history[0]["finished_at"])
                self.assertIn("未完成", history[0]["error_message"])
            finally:
                storage.close()


if __name__ == "__main__":
    unittest.main()

from datetime import datetime
from pathlib import Path
import tempfile
import unittest

from arxiv_llm_watch.models import TopicTrend
from arxiv_llm_watch.reporter import (
    build_window_delta,
    render_markdown_report,
    render_period_markdown_report,
    write_period_report,
    write_report,
)


class ReporterTests(unittest.TestCase):
    def test_report_includes_window_delta_section(self) -> None:
        current = [
            {
                "entry_id": "id-new",
                "title": "New reasoning paper",
                "primary_category": "cs.CL",
                "published": "2026-03-08T00:00:00+00:00",
                "tracked_topics": ["reasoning"],
                "topics": ["reasoning"],
                "summary_zh": "中文",
                "summary_en": "English",
                "background_zh": "",
                "background_en": "",
                "problem_zh": "",
                "problem_en": "",
                "method_zh": "",
                "method_en": "",
                "findings_zh": "",
                "findings_en": "",
                "limitations_zh": "",
                "limitations_en": "",
            }
        ]
        previous = [
            {
                "entry_id": "id-old",
                "title": "Old alignment paper",
                "primary_category": "cs.AI",
                "published": "2026-03-01T00:00:00+00:00",
                "tracked_topics": ["alignment & safety"],
                "topics": ["alignment"],
            }
        ]
        delta = build_window_delta(current, previous)
        report = render_markdown_report(
            generated_at=datetime(2026, 3, 8),
            report_papers=current,
            topic_trends=[TopicTrend(name="reasoning", current_count=1, baseline_count=0, growth=1.0)],
            status_counts={"analyzed": 1},
            window_delta=delta,
        )
        self.assertIn("## Since Last Window", report)
        self.assertIn("New In This Window", report)
        self.assertIn("No Longer In This Window", report)
        self.assertIn("New (0 -> 1)", report)

    def test_write_report_uses_timestamped_filename(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = write_report("demo", Path(temp_dir), datetime(2026, 3, 9, 11, 58, 7))
        self.assertEqual(path.name, "daily_20260309_115807.md")

    def test_period_report_renders_comparison_sections(self) -> None:
        current = [
            {
                "entry_id": "id-new",
                "title": "New reasoning paper",
                "primary_category": "cs.CL",
                "published": "2026-03-08T00:00:00+00:00",
                "tracked_topics": ["reasoning"],
                "topics": ["reasoning"],
                "summary_zh": "中文",
                "summary_en": "English",
            }
        ]
        previous = [
            {
                "entry_id": "id-old",
                "title": "Old safety paper",
                "primary_category": "cs.AI",
                "published": "2026-03-01T00:00:00+00:00",
                "tracked_topics": ["alignment & safety"],
                "topics": ["alignment"],
                "summary_zh": "中文旧",
                "summary_en": "English old",
            }
        ]
        report = render_period_markdown_report(
            generated_at=datetime(2026, 3, 10, 12, 0, 0),
            start=datetime(2026, 3, 3),
            end=datetime(2026, 3, 10),
            current_papers=current,
            previous_papers=previous,
            current_status_counts={"total": 2, "analyzed": 1},
        )
        self.assertIn("# Period Report - 2026-03-03 to 2026-03-09", report)
        self.assertIn("## Topic Comparison Matrix", report)
        self.assertIn("## Representative Papers", report)

    def test_write_period_report_uses_period_filename(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = write_period_report(
                "demo",
                Path(temp_dir),
                start=datetime(2026, 3, 3),
                end=datetime(2026, 3, 10),
                generated_at=datetime(2026, 3, 10, 12, 34, 56),
            )
        self.assertEqual(path.name, "period_20260303_20260309_123456.md")


if __name__ == "__main__":
    unittest.main()

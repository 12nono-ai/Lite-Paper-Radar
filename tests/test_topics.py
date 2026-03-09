from datetime import datetime, timezone
import unittest

from arxiv_llm_watch.topics import compute_topic_trends, extract_tracked_topics, format_topic_momentum, passes_keyword_gate


class TopicTests(unittest.TestCase):
    def test_keyword_gate_accepts_llm_terms(self) -> None:
        accepted = passes_keyword_gate(
            title="Efficient Large Language Model Post-Training",
            summary="We study instruction tuning for reasoning.",
            categories=["cs.CL"],
        )
        self.assertTrue(accepted)

    def test_topic_trends_rank_recent_growth(self) -> None:
        now = datetime(2026, 3, 8, tzinfo=timezone.utc)
        papers = [
            {"published": "2026-03-07T00:00:00+00:00", "topics": ["alignment", "reasoning"]},
            {"published": "2026-03-06T00:00:00+00:00", "topics": ["alignment"]},
            {"published": "2026-02-28T00:00:00+00:00", "topics": ["alignment"]},
        ]
        trends = compute_topic_trends(papers, recent_days=7, baseline_days=7, top_n=5, now=now)
        self.assertEqual(trends[0].name, "alignment")
        self.assertEqual(trends[0].current_count, 2)
        self.assertEqual(trends[0].baseline_count, 1)

    def test_extract_tracked_topics_maps_free_form_topics(self) -> None:
        tracked = extract_tracked_topics(
            title="The Spike, the Sparse and the Sink",
            summary="We study massive activations and attention sinks in large language models.",
            categories=["cs.CL"],
            raw_topics=["LLM Internal Belief Analysis", "attention sink research"],
        )
        self.assertIn("mechanistic interpretability", tracked)

    def test_topic_trends_prefer_tracked_topics_when_present(self) -> None:
        now = datetime(2026, 3, 8, tzinfo=timezone.utc)
        papers = [
            {
                "published": "2026-03-07T00:00:00+00:00",
                "topics": ["LLM-as-a-Judge"],
                "tracked_topics": ["evaluation & llm as a judge"],
            },
            {
                "published": "2026-03-06T00:00:00+00:00",
                "topics": ["LLM evaluation"],
                "tracked_topics": ["evaluation & llm as a judge"],
            },
        ]
        trends = compute_topic_trends(papers, recent_days=7, baseline_days=7, top_n=5, now=now)
        self.assertEqual(trends[0].name, "evaluation & llm as a judge")

    def test_format_topic_momentum_handles_new_and_percentage_cases(self) -> None:
        self.assertEqual(format_topic_momentum(5, 0), "New (0 -> 5)")
        self.assertEqual(format_topic_momentum(6, 2), "+200% (2 -> 6)")
        self.assertEqual(format_topic_momentum(2, 4), "-50% (4 -> 2)")

    def test_extract_tracked_topics_does_not_match_rag_inside_average(self) -> None:
        tracked = extract_tracked_topics(
            title="Bias-Bounded Evaluation",
            summary="We provide average bias guarantees for LLM judges.",
            categories=["cs.AI"],
            raw_topics=["LLM-as-a-Judge"],
        )
        self.assertNotIn("rag & retrieval", tracked)


if __name__ == "__main__":
    unittest.main()

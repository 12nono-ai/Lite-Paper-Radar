from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
from typing import Dict

from .config import AppConfig
from .fetcher import ArxivFetcher
from .llm_client import LLMClient
from .reporter import (
    build_fallback_report_comparison,
    build_window_delta,
    render_markdown_report,
    write_report,
)
from .storage import Storage
from .topics import compute_topic_trends, passes_keyword_gate


class ArxivLLMWatchPipeline:
    def __init__(self, config: AppConfig) -> None:
        self.config = config

    def run(
        self,
        lookback_days: int | None = None,
        max_results: int | None = None,
        report_paper_limit: int | None = None,
        analysis_limit: int | None = None,
        query_keywords: list[str] | None = None,
        run_source: str = "cli",
    ) -> Dict[str, object]:
        effective_config = replace(
            self.config,
            lookback_days=lookback_days if lookback_days is not None else self.config.lookback_days,
            arxiv_keywords=query_keywords if query_keywords is not None else self.config.arxiv_keywords,
            arxiv_max_results=max_results if max_results is not None else self.config.arxiv_max_results,
            report_paper_limit=report_paper_limit
            if report_paper_limit is not None
            else self.config.report_paper_limit,
            analysis_limit_per_run=analysis_limit
            if analysis_limit is not None
            else self.config.analysis_limit_per_run,
        )
        effective_config.ensure_directories()
        effective_config.validate()

        generated_at = datetime.now(timezone.utc)
        storage = Storage(effective_config.db_path)
        storage.initialize()
        overrides = {
            "lookback_days": lookback_days,
            "max_results": max_results,
            "report_paper_limit": report_paper_limit,
            "analysis_limit": analysis_limit,
            "query_keywords": query_keywords,
        }
        run_id = storage.start_run(
            source=run_source,
            overrides={key: value for key, value in overrides.items() if value is not None},
        )

        try:
            fetcher = ArxivFetcher()
            papers = fetcher.fetch_recent_papers(
                categories=effective_config.arxiv_categories,
                max_results=effective_config.arxiv_max_results,
                lookback_days=effective_config.lookback_days,
                keywords=effective_config.arxiv_keywords,
            )
            storage.save_papers(papers, fetched_at_iso=generated_at.isoformat())

            client = LLMClient(effective_config)
            pending_papers = storage.list_pending_papers(effective_config.analysis_limit_per_run)
            analyzed_count = 0
            filtered_count = 0
            rejected_count = 0
            error_count = 0
            for paper in pending_papers:
                if not passes_keyword_gate(paper.title, paper.summary, paper.categories):
                    storage.mark_filtered(paper.entry_id, "keyword gate rejected as non-LLM")
                    filtered_count += 1
                    continue
                try:
                    analysis = client.classify_and_summarize(paper)
                except Exception as exc:
                    storage.mark_error(paper.entry_id, str(exc))
                    error_count += 1
                    continue

                if not analysis.is_llm_related:
                    storage.mark_rejected(paper.entry_id, analysis.relevance_reason, analysis.llm_score)
                    rejected_count += 1
                    continue
                storage.save_analysis(paper.entry_id, analysis)
                analyzed_count += 1

            status_counts = storage.count_recent_by_status(effective_config.lookback_days)
            topic_window_days = effective_config.topic_recent_days + effective_config.topic_baseline_days
            topic_papers = storage.list_topic_window(topic_window_days)
            topic_trends = compute_topic_trends(
                topic_papers,
                recent_days=effective_config.topic_recent_days,
                baseline_days=effective_config.topic_baseline_days,
                top_n=effective_config.topic_limit,
                now=generated_at,
            )
            report_papers = storage.list_report_papers(
                days=effective_config.lookback_days,
                limit=effective_config.report_paper_limit,
            )
            current_window_papers = storage.list_report_papers_window(
                days=effective_config.lookback_days,
                offset_days=0,
                limit=None,
            )
            previous_window_papers = storage.list_report_papers_window(
                days=effective_config.lookback_days,
                offset_days=effective_config.lookback_days,
                limit=None,
            )
            comparison = build_fallback_report_comparison(report_papers, topic_trends)
            window_delta = build_window_delta(current_window_papers, previous_window_papers)
            report_text = render_markdown_report(
                generated_at=generated_at,
                report_papers=report_papers,
                topic_trends=topic_trends,
                status_counts=status_counts,
                comparison=comparison,
                window_delta=window_delta,
            )
            report_path = write_report(report_text, effective_config.reports_dir, generated_at)
            summary = {
                "fetched_count": len(papers),
                "analysis_batch_count": len(pending_papers),
                "analyzed_count": analyzed_count,
                "filtered_count": filtered_count,
                "rejected_count": rejected_count,
                "error_count": error_count,
                "status_counts": status_counts,
                "report_path": str(report_path),
                "query_keywords": effective_config.arxiv_keywords,
            }
            storage.finish_run(
                run_id,
                status="success",
                fetched_count=len(papers),
                analysis_batch_count=len(pending_papers),
                analyzed_count=analyzed_count,
                filtered_count=filtered_count,
                rejected_count=rejected_count,
                error_count=error_count,
                report_path=str(report_path),
            )
            return summary
        except Exception as exc:
            storage.finish_run(run_id, status="failed", error_message=str(exc))
            raise
        finally:
            storage.close()

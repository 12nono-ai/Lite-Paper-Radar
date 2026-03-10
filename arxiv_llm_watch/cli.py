from __future__ import annotations

import argparse
import json
from datetime import datetime, time, timedelta, timezone

from .config import AppConfig
from .dashboard import DashboardServer
from .pipeline import ArxivLLMWatchPipeline
from .reporter import (
    build_fallback_report_comparison,
    render_period_markdown_report,
    write_period_report,
)
from .storage import Storage
from .topics import compute_topic_trends


def _parse_optional_csv(raw: str | None) -> list[str] | None:
    if raw is None:
        return None
    return [item.strip() for item in raw.split(",") if item.strip()]


def _resolve_period_window(days: int | None, start_date: str | None, end_date: str | None) -> tuple[datetime, datetime]:
    if bool(start_date) != bool(end_date):
        raise ValueError("start-date and end-date must be provided together")
    if start_date and end_date:
        start = datetime.combine(datetime.strptime(start_date, "%Y-%m-%d").date(), time.min, tzinfo=timezone.utc)
        end = datetime.combine(datetime.strptime(end_date, "%Y-%m-%d").date(), time.min, tzinfo=timezone.utc) + timedelta(days=1)
        if end <= start:
            raise ValueError("end-date must be later than start-date")
        return start, end

    effective_days = max(1, int(days or 7))
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=effective_days)
    return start, end


def _generate_period_report(
    config: AppConfig,
    start: datetime,
    end: datetime,
    paper_limit: int | None = None,
) -> dict:
    config.ensure_directories()
    generated_at = datetime.now(timezone.utc)
    storage = Storage(config.db_path)
    storage.initialize()
    try:
        current_papers = storage.list_report_papers_between(start=start, end=end, limit=paper_limit or config.report_paper_limit)
        previous_start = start - (end - start)
        previous_papers = storage.list_report_papers_between(start=previous_start, end=start, limit=None)
        current_status_counts = storage.count_status_between(start=start, end=end)
        topic_trends = compute_topic_trends(
            current_papers + previous_papers,
            recent_days=max(1, (end - start).days),
            baseline_days=max(1, (end - start).days),
            top_n=config.topic_limit,
            now=end,
        )
        comparison = build_fallback_report_comparison(current_papers, topic_trends)
        report_text = render_period_markdown_report(
            generated_at=generated_at,
            start=start,
            end=end,
            current_papers=current_papers,
            previous_papers=previous_papers,
            current_status_counts=current_status_counts,
            comparison=comparison,
        )
        report_path = write_period_report(report_text, config.reports_dir, start=start, end=end, generated_at=generated_at)
        return {
            "report_path": str(report_path),
            "start": start.isoformat(),
            "end": end.isoformat(),
            "paper_count": len(current_papers),
            "previous_paper_count": len(previous_papers),
        }
    finally:
        storage.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Daily arXiv LLM watch pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Fetch, analyze, and generate a report")
    run_parser.add_argument("--lookback-days", type=int, default=None, help="Override LOOKBACK_DAYS")
    run_parser.add_argument("--max-results", type=int, default=None, help="Override ARXIV_MAX_RESULTS")
    run_parser.add_argument("--report-paper-limit", type=int, default=None, help="Override REPORT_PAPER_LIMIT")
    run_parser.add_argument("--analysis-limit", type=int, default=None, help="Override ANALYSIS_LIMIT_PER_RUN")
    run_parser.add_argument(
        "--query-keywords",
        default=None,
        help="Comma-separated fetch keywords applied on top of arXiv categories",
    )

    period_parser = subparsers.add_parser("period-report", help="Generate a summary/comparison report for a time window")
    period_parser.add_argument("--days", type=int, default=7, help="Use the most recent N days")
    period_parser.add_argument("--start-date", default=None, help="Start date in YYYY-MM-DD")
    period_parser.add_argument("--end-date", default=None, help="End date in YYYY-MM-DD")
    period_parser.add_argument("--paper-limit", type=int, default=None, help="Representative papers to render")

    dashboard_parser = subparsers.add_parser("dashboard", help="Start the local web dashboard")
    dashboard_parser.add_argument("--host", default="127.0.0.1", help="Bind host")
    dashboard_parser.add_argument("--port", type=int, default=8765, help="Bind port")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    config = AppConfig.from_env()

    if args.command == "run":
        pipeline = ArxivLLMWatchPipeline(config)
        summary = pipeline.run(
            lookback_days=args.lookback_days,
            max_results=args.max_results,
            report_paper_limit=args.report_paper_limit,
            analysis_limit=args.analysis_limit,
            query_keywords=_parse_optional_csv(args.query_keywords),
            run_source="cli",
        )
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return

    if args.command == "period-report":
        start, end = _resolve_period_window(args.days, args.start_date, args.end_date)
        summary = _generate_period_report(config, start=start, end=end, paper_limit=args.paper_limit)
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return

    if args.command == "dashboard":
        server = DashboardServer(config=config, host=args.host, port=args.port)
        server.serve()


if __name__ == "__main__":
    main()

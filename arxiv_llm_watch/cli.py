from __future__ import annotations

import argparse
import json

from .config import AppConfig
from .dashboard import DashboardServer
from .pipeline import ArxivLLMWatchPipeline


def _parse_optional_csv(raw: str | None) -> list[str] | None:
    if raw is None:
        return None
    return [item.strip() for item in raw.split(",") if item.strip()]


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

    if args.command == "dashboard":
        server = DashboardServer(config=config, host=args.host, port=args.port)
        server.serve()


if __name__ == "__main__":
    main()

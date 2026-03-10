from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import datetime, time, timedelta, timezone
from pathlib import Path

from .config import AppConfig, normalize_llm_provider
from .dashboard import DashboardServer
from .pipeline import ArxivLLMWatchPipeline
from .reporter import (
    build_fallback_report_comparison,
    render_period_markdown_report,
    write_period_report,
)
from .storage import Storage
from .topics import compute_topic_trends


SUPPORTED_PROVIDERS = ("ark", "openai_compatible")


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


def _build_env_template(provider: str) -> str:
    normalized = normalize_llm_provider(provider)
    if normalized not in SUPPORTED_PROVIDERS:
        raise ValueError(f"Unsupported provider: {provider}")

    defaults = {
        "ark": {
            "base_url": "https://ark.cn-beijing.volces.com/api/v3",
            "model": "doubao-seed-2-0-pro-260215",
        },
        "openai_compatible": {
            "base_url": "https://api.openai.com/v1",
            "model": "gpt-4.1-mini",
        },
    }[normalized]

    return "\n".join(
        [
            f"LLM_PROVIDER={normalized}",
            "LLM_API_KEY=",
            f"LLM_BASE_URL={defaults['base_url']}",
            f"LLM_MODEL={defaults['model']}",
            "LLM_API_PATH=/chat/completions",
            "LLM_HEADERS_JSON=",
            "",
            "# Legacy Ark keys remain supported as fallback:",
            "# ARK_API_KEY=",
            "# ARK_BASE_URL=",
            "# ARK_MODEL=",
            "",
            "ARXIV_CATEGORIES=cs.CL,cs.AI,cs.LG,stat.ML",
            "ARXIV_KEYWORDS=",
            "ARXIV_MAX_RESULTS=250",
            "LOOKBACK_DAYS=2",
            "TOPIC_RECENT_DAYS=7",
            "TOPIC_BASELINE_DAYS=7",
            "TOPIC_LIMIT=8",
            "REPORT_PAPER_LIMIT=12",
            "ANALYSIS_LIMIT_PER_RUN=6",
            "DATA_DIR=data",
            "REPORTS_DIR=reports",
            "DB_PATH=data/arxiv_llm_watch.db",
            "LLM_TEMPERATURE=0.2",
            "",
        ]
    )


def _init_env_file(env_path: Path, provider: str, force: bool = False) -> dict:
    normalized = normalize_llm_provider(provider)
    if env_path.exists() and not force:
        raise FileExistsError(f"{env_path} already exists. Use --force to overwrite it.")
    env_path.write_text(_build_env_template(normalized), encoding="utf-8")
    return {
        "env_path": str(env_path),
        "provider": normalized,
        "next_steps": [
            f"Fill in LLM_API_KEY in {env_path.name}",
            "Run `python3 -m arxiv_llm_watch.cli doctor`",
            "Run `python3 -m arxiv_llm_watch.cli dashboard`",
        ],
    }


def _doctor_report(config: AppConfig, env_path: Path) -> dict:
    checks = []

    def add_check(name: str, ok: bool, detail: str, fix: str = "") -> None:
        checks.append(
            {
                "name": name,
                "ok": ok,
                "detail": detail,
                "fix": fix,
            }
        )

    python_ok = sys.version_info >= (3, 9)
    add_check(
        "python",
        python_ok,
        f"Python {platform.python_version()}",
        "Use Python 3.9+ and prefer the project .venv" if not python_ok else "",
    )
    add_check(
        "env_file",
        env_path.exists(),
        f"{env_path} {'found' if env_path.exists() else 'missing'}",
        "Run `python3 -m arxiv_llm_watch.cli init --provider ark` or `--provider openai_compatible`"
        if not env_path.exists()
        else "",
    )
    add_check(
        "provider",
        config.llm_provider in SUPPORTED_PROVIDERS,
        f"Configured provider: {config.llm_provider}",
        "Set LLM_PROVIDER to `ark` or `openai_compatible`" if config.llm_provider not in SUPPORTED_PROVIDERS else "",
    )
    add_check(
        "api_key",
        bool(config.llm_api_key),
        "LLM API key configured" if config.llm_api_key else "LLM API key missing",
        "Fill in LLM_API_KEY in .env" if not config.llm_api_key else "",
    )
    add_check(
        "base_url",
        bool(config.llm_base_url),
        config.llm_base_url or "LLM base URL missing",
        "Fill in LLM_BASE_URL in .env" if not config.llm_base_url else "",
    )
    add_check(
        "model",
        bool(config.llm_model),
        config.llm_model or "LLM model missing",
        "Fill in LLM_MODEL in .env" if not config.llm_model else "",
    )
    if config.llm_provider == "ark":
        try:
            import volcenginesdkarkruntime  # noqa: F401

            add_check("provider_dependency", True, "Ark SDK installed")
        except ImportError:
            add_check(
                "provider_dependency",
                False,
                "Ark SDK missing",
                "Install dependencies with `pip install -e .` or `pip install -r requirements.txt`",
            )
    else:
        add_check("provider_dependency", True, "OpenAI-compatible mode uses the Python standard library")

    config.ensure_directories()
    add_check("data_dir", config.data_dir.exists(), f"Data dir: {config.data_dir}")
    add_check("reports_dir", config.reports_dir.exists(), f"Reports dir: {config.reports_dir}")

    ok_count = sum(1 for check in checks if check["ok"])
    return {
        "ok": ok_count == len(checks),
        "checks": checks,
        "summary": {
            "passed": ok_count,
            "total": len(checks),
            "provider": config.llm_provider,
            "provider_label": config.llm_provider_label,
            "model": config.llm_model,
            "env_path": str(env_path),
        },
    }


def _print_doctor_report(report: dict) -> None:
    summary = report["summary"]
    print(f"Setup check: {summary['passed']}/{summary['total']} passed")
    print(f"Provider: {summary['provider_label']} ({summary['provider']})")
    print(f"Model: {summary['model'] or '(missing)'}")
    print(f"Env file: {summary['env_path']}")
    print("")
    for check in report["checks"]:
        prefix = "[OK]" if check["ok"] else "[FAIL]"
        print(f"{prefix} {check['name']}: {check['detail']}")
        if check["fix"]:
            print(f"  next: {check['fix']}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Daily arXiv LLM watch pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    init_parser = subparsers.add_parser("init", help="Create a starter .env file")
    init_parser.add_argument(
        "--provider",
        default="ark",
        help="LLM provider template to use: ark or openai_compatible",
    )
    init_parser.add_argument("--env-path", default=".env", help="Where to write the env file")
    init_parser.add_argument("--force", action="store_true", help="Overwrite an existing env file")

    doctor_parser = subparsers.add_parser("doctor", help="Check whether the local setup is ready")
    doctor_parser.add_argument("--env-path", default=".env", help="Env file to inspect")
    doctor_parser.add_argument("--json", action="store_true", help="Print the report as JSON")

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

    if args.command == "init":
        summary = _init_env_file(Path(args.env_path), provider=args.provider, force=args.force)
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return

    config = AppConfig.from_env(Path(getattr(args, "env_path", ".env")))

    if args.command == "doctor":
        report = _doctor_report(config, Path(args.env_path))
        if args.json:
            print(json.dumps(report, indent=2, ensure_ascii=False))
            return
        _print_doctor_report(report)
        if not report["ok"]:
            raise SystemExit(1)
        return

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

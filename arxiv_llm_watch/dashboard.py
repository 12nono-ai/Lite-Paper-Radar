from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List
from urllib.parse import parse_qs, urlparse

from .config import AppConfig
from .pipeline import ArxivLLMWatchPipeline
from .storage import Storage
from .topics import compute_topic_trends, format_topic_momentum


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _coerce_optional_int(value: Any) -> int | None:
    if value in (None, "", 0):
        return None
    return int(value)


def _coerce_csv_override(value: Any) -> List[str] | None:
    if value is None:
        return None
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return [item.strip() for item in str(value).split(",") if item.strip()]


def _first_query_value(values: Dict[str, List[str]], key: str, default: str = "") -> str:
    return values.get(key, [default])[0]


@dataclass
class RunState:
    running: bool = False
    last_started_at: str = ""
    last_finished_at: str = ""
    last_error: str = ""
    last_result: Dict[str, Any] = field(default_factory=dict)
    last_overrides: Dict[str, Any] = field(default_factory=dict)


class DashboardController:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self._lock = threading.Lock()
        self._run_state = RunState()
        self._recover_orphaned_runs()

    def _recover_orphaned_runs(self) -> None:
        self.config.ensure_directories()
        storage = Storage(self.config.db_path)
        storage.initialize()
        try:
            storage.mark_interrupted_runs()
        finally:
            storage.close()

    def build_state(self) -> Dict[str, Any]:
        self.config.ensure_directories()
        storage = Storage(self.config.db_path)
        storage.initialize()

        try:
            total_status = storage.count_by_status()
            recent_status = storage.count_recent_by_status(self.config.lookback_days)
            topic_trends = compute_topic_trends(
                storage.list_topic_window(self.config.topic_recent_days + self.config.topic_baseline_days),
                recent_days=self.config.topic_recent_days,
                baseline_days=self.config.topic_baseline_days,
                top_n=self.config.topic_limit,
            )
            analyzed = storage.list_report_papers(
                days=self.config.lookback_days,
                limit=self.config.report_paper_limit,
            )
            pending = storage.list_papers_by_status("pending", limit=10)
            rejected = storage.list_papers_by_status("rejected", limit=6)
            errors = storage.list_papers_by_status("error", limit=6)
            categories = sorted(set(self.config.arxiv_categories) | set(storage.list_primary_categories()))
            category_mix = storage.count_recent_categories(self.config.lookback_days, status="analyzed")
            filter_facets = storage.build_search_facets()
            run_history = storage.list_run_history(limit=8)
        finally:
            storage.close()

        reports = self._list_reports(limit=8)

        with self._lock:
            run_state = {
                "running": self._run_state.running,
                "last_started_at": self._run_state.last_started_at,
                "last_finished_at": self._run_state.last_finished_at,
                "last_error": self._run_state.last_error,
                "last_result": self._run_state.last_result,
                "last_overrides": self._run_state.last_overrides,
            }

        return {
            "generated_at": _utc_now_iso(),
            "config": {
                "categories": self.config.arxiv_categories,
                "query_keywords": self.config.arxiv_keywords,
                "lookback_days": self.config.lookback_days,
                "max_results": self.config.arxiv_max_results,
                "analysis_limit_per_run": self.config.analysis_limit_per_run,
                "report_paper_limit": self.config.report_paper_limit,
                "model": self.config.ark_model,
            },
            "available_categories": categories,
            "filter_facets": filter_facets,
            "category_mix": category_mix,
            "status_counts": {
                "total": total_status,
                "recent": recent_status,
            },
            "topic_trends": [
                {
                    "name": trend.name,
                    "current_count": trend.current_count,
                    "baseline_count": trend.baseline_count,
                    "growth": trend.growth,
                    "momentum_label": format_topic_momentum(trend.current_count, trend.baseline_count),
                }
                for trend in topic_trends
            ],
            "analyzed_papers": analyzed,
            "pending_papers": pending,
            "rejected_papers": rejected,
            "error_papers": errors,
            "reports": reports,
            "run_state": run_state,
            "run_history": run_history,
        }

    def search_papers(
        self,
        query: str = "",
        status: str = "analyzed",
        category: str = "all",
        tracked_topic: str = "",
        flag: str = "all",
        days: int | None = None,
        limit: int = 12,
        page: int = 1,
        sort: str = "published_desc",
    ) -> Dict[str, Any]:
        page = max(1, int(page or 1))
        self.config.ensure_directories()
        storage = Storage(self.config.db_path)
        storage.initialize()
        try:
            total = storage.count_search_papers(
                query=query.strip(),
                status=status,
                category=category,
                tracked_topic=tracked_topic,
                flag=flag,
                days=days,
                sort=sort,
            )
            total_pages = max(1, (total + limit - 1) // limit) if total else 1
            page = min(page, total_pages)
            offset = (page - 1) * limit
            papers = storage.search_papers(
                query=query.strip(),
                status=status,
                category=category,
                tracked_topic=tracked_topic,
                flag=flag,
                days=days,
                limit=limit,
                offset=offset,
                sort=sort,
            )
        finally:
            storage.close()
        return {
            "query": query,
            "status": status,
            "category": category,
            "tracked_topic": tracked_topic,
            "flag": flag,
            "days": days,
            "limit": limit,
            "page": page,
            "offset": offset,
            "total": total,
            "total_pages": total_pages,
            "sort": sort,
            "papers": papers,
        }

    def get_paper(self, entry_id: str) -> Dict[str, Any] | None:
        self.config.ensure_directories()
        storage = Storage(self.config.db_path)
        storage.initialize()
        try:
            paper = storage.get_paper(entry_id)
            if paper is None:
                return None
            paper["related_papers"] = storage.find_related_papers(entry_id, limit=6)
            return paper
        finally:
            storage.close()

    def apply_paper_action(self, entry_id: str, payload: Dict[str, Any]) -> Dict[str, Any] | None:
        self.config.ensure_directories()
        storage = Storage(self.config.db_path)
        storage.initialize()
        try:
            if payload.get("requeue"):
                storage.requeue_paper(entry_id)
            else:
                manual_topics = payload.get("manual_topics")
                storage.update_paper_feedback(
                    entry_id,
                    starred=payload.get("starred"),
                    ignored=payload.get("ignored"),
                    manual_topics=manual_topics if isinstance(manual_topics, list) else None,
                    manual_note=payload.get("manual_note"),
                )
            paper = storage.get_paper(entry_id)
            if paper is None:
                return None
            paper["related_papers"] = storage.find_related_papers(entry_id, limit=6)
            return paper
        finally:
            storage.close()

    def start_run(self, overrides: Dict[str, Any]) -> Dict[str, Any]:
        with self._lock:
            if self._run_state.running:
                raise RuntimeError("A pipeline run is already in progress.")
            self._run_state.running = True
            self._run_state.last_started_at = _utc_now_iso()
            self._run_state.last_finished_at = ""
            self._run_state.last_error = ""
            self._run_state.last_result = {}
            self._run_state.last_overrides = overrides

        thread = threading.Thread(target=self._run_pipeline, args=(overrides,), daemon=True)
        thread.start()
        return self.build_state()["run_state"]

    def _run_pipeline(self, overrides: Dict[str, Any]) -> None:
        try:
            pipeline = ArxivLLMWatchPipeline(self.config)
            result = pipeline.run(
                lookback_days=overrides.get("lookback_days"),
                max_results=overrides.get("max_results"),
                report_paper_limit=overrides.get("report_paper_limit"),
                analysis_limit=overrides.get("analysis_limit"),
                query_keywords=overrides.get("query_keywords"),
                run_source="dashboard",
            )
            with self._lock:
                self._run_state.last_result = result
        except Exception as exc:
            with self._lock:
                self._run_state.last_error = str(exc)
        finally:
            with self._lock:
                self._run_state.running = False
                self._run_state.last_finished_at = _utc_now_iso()

    def _list_reports(self, limit: int) -> List[Dict[str, Any]]:
        report_files = sorted(
            self.config.reports_dir.glob("*.md"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        results = []
        for path in report_files[:limit]:
            stat = path.stat()
            results.append(
                {
                    "name": path.name,
                    "size_bytes": stat.st_size,
                    "modified_at": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
                }
            )
        return results


class DashboardServer:
    def __init__(self, config: AppConfig, host: str, port: int) -> None:
        self.config = config
        self.host = host
        self.port = port
        self.controller = DashboardController(config)

    def serve(self) -> None:
        handler_class = self._build_handler()
        server = ThreadingHTTPServer((self.host, self.port), handler_class)
        print(f"Dashboard running at http://{self.host}:{self.port}")
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            server.server_close()

    def _build_handler(self) -> type[BaseHTTPRequestHandler]:
        controller = self.controller
        config = self.config

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                parsed = urlparse(self.path)
                if parsed.path == "/":
                    self._send_html(INDEX_HTML.replace("__PAGE_VIEW__", "overview"))
                    return
                if parsed.path == "/papers":
                    self._send_html(INDEX_HTML.replace("__PAGE_VIEW__", "papers"))
                    return
                if parsed.path == "/reports":
                    self._send_html(INDEX_HTML.replace("__PAGE_VIEW__", "reports"))
                    return
                if parsed.path == "/paper":
                    self._send_html(PAPER_HTML)
                    return
                if parsed.path == "/api/state":
                    self._send_json(controller.build_state())
                    return
                if parsed.path == "/api/papers":
                    query = parse_qs(parsed.query)
                    payload = controller.search_papers(
                        query=_first_query_value(query, "query"),
                        status=_first_query_value(query, "status", "analyzed"),
                        category=_first_query_value(query, "category", "all"),
                        tracked_topic=_first_query_value(query, "tracked_topic"),
                        flag=_first_query_value(query, "flag", "all"),
                        days=_coerce_optional_int(_first_query_value(query, "days")),
                        limit=int(_first_query_value(query, "limit", "12")),
                        page=int(_first_query_value(query, "page", "1")),
                        sort=_first_query_value(query, "sort", "published_desc"),
                    )
                    self._send_json(payload)
                    return
                if parsed.path == "/api/paper":
                    query = parse_qs(parsed.query)
                    entry_id = _first_query_value(query, "entry_id")
                    if not entry_id:
                        self._send_json({"error": "Missing entry_id"}, status=HTTPStatus.BAD_REQUEST)
                        return
                    payload = controller.get_paper(entry_id)
                    if payload is None:
                        self._send_json({"error": "Paper not found"}, status=HTTPStatus.NOT_FOUND)
                        return
                    self._send_json(payload)
                    return
                if parsed.path.startswith("/reports/"):
                    self._serve_report(parsed.path.removeprefix("/reports/"))
                    return
                self.send_error(HTTPStatus.NOT_FOUND)

            def do_POST(self) -> None:
                parsed = urlparse(self.path)
                payload = self._read_json_body()
                if parsed.path == "/api/run":
                    overrides = {
                        "lookback_days": _coerce_optional_int(payload.get("lookback_days")),
                        "max_results": _coerce_optional_int(payload.get("max_results")),
                        "report_paper_limit": _coerce_optional_int(payload.get("report_paper_limit")),
                        "analysis_limit": _coerce_optional_int(payload.get("analysis_limit")),
                        "query_keywords": _coerce_csv_override(payload.get("query_keywords")),
                    }
                    try:
                        run_state = controller.start_run(overrides)
                    except RuntimeError as exc:
                        self._send_json({"error": str(exc)}, status=HTTPStatus.CONFLICT)
                        return
                    self._send_json({"ok": True, "run_state": run_state}, status=HTTPStatus.ACCEPTED)
                    return
                if parsed.path == "/api/paper/action":
                    entry_id = str(payload.get("entry_id") or "").strip()
                    if not entry_id:
                        self._send_json({"error": "Missing entry_id"}, status=HTTPStatus.BAD_REQUEST)
                        return
                    updated = controller.apply_paper_action(entry_id, payload)
                    if updated is None:
                        self._send_json({"error": "Paper not found"}, status=HTTPStatus.NOT_FOUND)
                        return
                    self._send_json({"ok": True, "paper": updated})
                    return
                self.send_error(HTTPStatus.NOT_FOUND)

            def log_message(self, format: str, *args: Any) -> None:
                return

            def _read_json_body(self) -> Dict[str, Any]:
                content_length = int(self.headers.get("Content-Length", "0"))
                if content_length <= 0:
                    return {}
                raw = self.rfile.read(content_length).decode("utf-8")
                if not raw.strip():
                    return {}
                return json.loads(raw)

            def _serve_report(self, filename: str) -> None:
                safe_name = Path(filename).name
                report_path = config.reports_dir / safe_name
                if not report_path.exists() or report_path.suffix != ".md":
                    self.send_error(HTTPStatus.NOT_FOUND)
                    return
                content = report_path.read_text(encoding="utf-8")
                self._send_text(content, "text/markdown; charset=utf-8")

            def _send_json(self, payload: Dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
                content = json.dumps(payload, ensure_ascii=False).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(content)))
                self.end_headers()
                self.wfile.write(content)

            def _send_html(self, content: str) -> None:
                self._send_text(content, "text/html; charset=utf-8")

            def _send_text(self, content: str, content_type: str) -> None:
                body = content.encode("utf-8")
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        return Handler


INDEX_HTML = """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>arXiv LLM Watch</title>
  <style>
    :root {
      --ink: #14213d;
      --paper: #f6f1e8;
      --sand: #e8dcc8;
      --accent: #b23a48;
      --teal: #0d5c63;
      --gold: #c58b2a;
      --muted: #6b6f76;
      --card: rgba(255, 251, 245, 0.8);
      --line: rgba(20, 33, 61, 0.14);
      --shadow: 0 18px 40px rgba(20, 33, 61, 0.12);
      --radius: 24px;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      color: var(--ink);
      background:
        radial-gradient(circle at top left, rgba(197, 139, 42, 0.28), transparent 28%),
        radial-gradient(circle at 88% 10%, rgba(178, 58, 72, 0.20), transparent 20%),
        linear-gradient(180deg, #fbf7f0 0%, #efe7da 100%);
      font-family: "Iowan Old Style", "Palatino Linotype", "Source Han Serif SC", serif;
    }
    .shell {
      width: min(1360px, calc(100% - 30px));
      margin: 20px auto 48px;
    }
    .hero {
      position: relative;
      overflow: hidden;
      padding: 30px;
      border-radius: 34px;
      background: linear-gradient(135deg, rgba(13, 92, 99, 0.96), rgba(20, 33, 61, 0.98));
      color: #f9f4ec;
      box-shadow: var(--shadow);
    }
    .hero::before,
    .hero::after {
      content: "";
      position: absolute;
      border-radius: 999px;
      opacity: 0.18;
      pointer-events: none;
    }
    .hero::before {
      width: 320px;
      height: 320px;
      right: -80px;
      top: -100px;
      background: radial-gradient(circle, #fff, transparent 70%);
    }
    .hero::after {
      width: 220px;
      height: 220px;
      left: -70px;
      bottom: -90px;
      background: radial-gradient(circle, #c58b2a, transparent 70%);
    }
    .eyebrow {
      font-size: 12px;
      letter-spacing: 0.2em;
      text-transform: uppercase;
      opacity: 0.72;
    }
    .title {
      margin: 10px 0 8px;
      font-size: clamp(34px, 5vw, 64px);
      line-height: 0.95;
      max-width: 10ch;
    }
    .subtitle {
      margin: 0;
      max-width: 62ch;
      line-height: 1.6;
      color: rgba(249, 244, 236, 0.86);
      font-size: 15px;
    }
    .hero-meta {
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      margin-top: 18px;
    }
    .topnav {
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 12px;
      margin-top: 22px;
      max-width: 920px;
    }
    .topnav a {
      display: block;
      padding: 16px 18px;
      border-radius: 22px;
      color: rgba(249, 244, 236, 0.84);
      background: linear-gradient(180deg, rgba(255, 255, 255, 0.10), rgba(255, 255, 255, 0.06));
      border: 1px solid rgba(255, 255, 255, 0.12);
      box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.08);
      transition: transform 150ms ease, background 150ms ease, border-color 150ms ease, box-shadow 150ms ease;
    }
    .topnav a:hover {
      text-decoration: none;
      transform: translateY(-1px);
    }
    .topnav a.active {
      background: linear-gradient(180deg, rgba(255, 251, 245, 0.96), rgba(255, 247, 236, 0.88));
      color: var(--ink);
      border-color: rgba(255, 255, 255, 0.36);
      box-shadow: 0 18px 34px rgba(10, 18, 30, 0.18);
    }
    .nav-label {
      display: block;
      font-size: 15px;
      font-weight: 700;
      letter-spacing: 0.08em;
      text-transform: uppercase;
    }
    .nav-copy {
      display: block;
      margin-top: 8px;
      font-size: 12px;
      line-height: 1.55;
      letter-spacing: 0.01em;
      text-transform: none;
      opacity: 0.82;
    }
    .chip {
      padding: 8px 12px;
      border-radius: 999px;
      background: rgba(255, 255, 255, 0.12);
      border: 1px solid rgba(255, 255, 255, 0.12);
      font-size: 13px;
    }
    .panel-action {
      color: var(--teal);
      font-size: 13px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
    }
    .grid {
      display: grid;
      grid-template-columns: 1.2fr 0.8fr;
      gap: 18px;
      margin-top: 18px;
    }
    .panel {
      background: var(--card);
      border: 1px solid var(--line);
      border-radius: var(--radius);
      box-shadow: var(--shadow);
      backdrop-filter: blur(10px);
    }
    .panel-head {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      padding: 20px 22px 0;
    }
    .panel-title {
      margin: 0;
      font-size: 24px;
    }
    .panel-subtitle {
      color: var(--muted);
      font-size: 13px;
    }
    .panel-body {
      padding: 18px 22px 22px;
    }
    .stats {
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 12px;
    }
    .stat {
      border-radius: 18px;
      padding: 16px;
      background: linear-gradient(180deg, rgba(255,255,255,0.78), rgba(255,255,255,0.48));
      border: 1px solid rgba(20, 33, 61, 0.08);
    }
    .stat-label {
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      color: var(--muted);
    }
    .stat-value {
      margin-top: 8px;
      font-size: 34px;
      line-height: 1;
    }
    .control-grid,
    .search-grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 12px;
      margin-bottom: 16px;
    }
    .search-grid {
      grid-template-columns: repeat(4, minmax(0, 1fr));
    }
    .search-wide {
      grid-column: span 2;
    }
    label {
      display: block;
      font-size: 12px;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      color: var(--muted);
      margin-bottom: 6px;
    }
    input,
    select {
      width: 100%;
      border: 1px solid rgba(20, 33, 61, 0.12);
      background: rgba(255,255,255,0.86);
      border-radius: 14px;
      padding: 12px 14px;
      color: var(--ink);
      font: inherit;
    }
    button {
      appearance: none;
      border: 0;
      border-radius: 16px;
      background: linear-gradient(135deg, var(--accent), #d96b3d);
      color: white;
      padding: 14px 18px;
      font: inherit;
      cursor: pointer;
      box-shadow: 0 14px 24px rgba(178, 58, 72, 0.24);
      transition: transform 150ms ease, box-shadow 150ms ease, opacity 150ms ease;
    }
    button.secondary {
      background: linear-gradient(135deg, var(--teal), #2b7a78);
      box-shadow: 0 14px 24px rgba(13, 92, 99, 0.20);
    }
    button:hover { transform: translateY(-1px); }
    button:disabled {
      opacity: 0.56;
      cursor: wait;
      transform: none;
      box-shadow: none;
    }
    .muted {
      color: var(--muted);
      font-size: 14px;
      line-height: 1.6;
    }
    .status-banner {
      border-radius: 18px;
      padding: 14px 16px;
      background: rgba(13, 92, 99, 0.08);
      border: 1px solid rgba(13, 92, 99, 0.14);
      margin-top: 12px;
      min-height: 52px;
    }
    .run-history {
      display: grid;
      gap: 10px;
      margin-top: 16px;
    }
    .run-entry {
      border-radius: 22px;
      padding: 18px;
      background: linear-gradient(135deg, rgba(13, 92, 99, 0.10), rgba(197, 139, 42, 0.10));
      border: 1px solid rgba(20, 33, 61, 0.08);
      margin-bottom: 16px;
    }
    .run-entry-head {
      display: flex;
      justify-content: space-between;
      gap: 16px;
      align-items: center;
      flex-wrap: wrap;
    }
    .run-entry-copy {
      flex: 1 1 320px;
    }
    .run-entry-copy .muted {
      margin: 10px 0 0;
    }
    .run-entry-actions {
      display: flex;
      gap: 12px;
      align-items: center;
      flex-wrap: wrap;
      flex: 0 0 auto;
    }
    .run-entry-actions .status-banner {
      margin-top: 0;
      min-width: 280px;
      flex: 1 1 280px;
    }
    .run-card {
      border-radius: 16px;
      padding: 12px 14px;
      background: rgba(255,255,255,0.74);
      border: 1px solid rgba(20, 33, 61, 0.08);
    }
    .topic-bars {
      display: grid;
      gap: 12px;
    }
    .visual-grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 14px;
      margin-bottom: 18px;
    }
    .visual-card {
      border-radius: 18px;
      border: 1px solid rgba(20, 33, 61, 0.08);
      background: rgba(255,255,255,0.62);
      padding: 14px;
    }
    .svg-wrap {
      margin-top: 10px;
      min-height: 220px;
    }
    .topic-row {
      display: grid;
      gap: 8px;
    }
    .topic-filter-button {
      appearance: none;
      border: 0;
      padding: 0;
      margin: 0;
      background: transparent;
      color: inherit;
      text-align: left;
      box-shadow: none;
      border-radius: 0;
    }
    .topic-filter-button:hover {
      transform: none;
      box-shadow: none;
    }
    .topic-headline {
      display: flex;
      justify-content: space-between;
      gap: 12px;
      font-size: 14px;
    }
    .topic-meter {
      position: relative;
      height: 12px;
      border-radius: 999px;
      overflow: hidden;
      background: rgba(20, 33, 61, 0.08);
    }
    .topic-meter-fill {
      position: absolute;
      inset: 0 auto 0 0;
      border-radius: 999px;
      background: linear-gradient(90deg, var(--gold), var(--accent));
    }
    .paper-stack,
    .queue-stack {
      display: grid;
      gap: 14px;
    }
    .paper-card,
    .queue-card {
      border-radius: 20px;
      border: 1px solid rgba(20, 33, 61, 0.08);
      background: linear-gradient(180deg, rgba(255,255,255,0.9), rgba(255,255,255,0.62));
      padding: 18px;
    }
    .paper-title,
    .queue-title {
      margin: 0;
      font-size: 21px;
      line-height: 1.2;
    }
    .paper-title a {
      color: inherit;
      text-decoration: none;
    }
    .paper-title a:hover {
      text-decoration: underline;
    }
    .paper-meta,
    .queue-meta,
    .preview-meta {
      margin: 10px 0 0;
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      color: var(--muted);
      font-size: 12px;
    }
    .topic-pill,
    .status-pill,
    .flag-pill {
      display: inline-block;
      margin: 6px 6px 0 0;
      padding: 6px 10px;
      border-radius: 999px;
      font-size: 12px;
    }
    .topic-pill {
      background: rgba(197, 139, 42, 0.12);
      color: #7b5715;
    }
    .topic-pill.topic-filter {
      appearance: none;
      border: 0;
      cursor: pointer;
      font: inherit;
    }
    .topic-pill.tracked {
      background: rgba(13, 92, 99, 0.12);
      color: var(--teal);
    }
    .topic-label {
      margin-top: 12px;
      font-size: 11px;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      color: var(--muted);
    }
    .status-pill {
      background: rgba(13, 92, 99, 0.12);
      color: var(--teal);
      text-transform: uppercase;
      letter-spacing: 0.08em;
    }
    .status-pill[data-status="pending"] {
      background: rgba(197, 139, 42, 0.16);
      color: #8b5e14;
    }
    .status-pill[data-status="rejected"] {
      background: rgba(178, 58, 72, 0.14);
      color: #9a2d3a;
    }
    .status-pill[data-status="error"] {
      background: rgba(120, 38, 41, 0.16);
      color: #6c1f25;
    }
    .flag-pill {
      background: rgba(20, 33, 61, 0.08);
      color: var(--ink);
    }
    .flag-pill.starred {
      background: rgba(197, 139, 42, 0.18);
      color: #7b5715;
    }
    .flag-pill.ignored {
      background: rgba(120, 38, 41, 0.14);
      color: #6c1f25;
    }
    .flag-pill.manual {
      background: rgba(13, 92, 99, 0.14);
      color: var(--teal);
    }
    .paper-actions {
      margin-top: 12px;
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
    }
    .text-link {
      color: var(--teal);
      font-size: 13px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
    }
    details {
      margin-top: 14px;
      border-top: 1px dashed rgba(20, 33, 61, 0.12);
      padding-top: 14px;
    }
    summary {
      cursor: pointer;
      color: var(--teal);
      font-size: 13px;
      letter-spacing: 0.08em;
      text-transform: uppercase;
    }
    .dual {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 12px;
      margin-top: 14px;
    }
    .dual-section {
      border-radius: 16px;
      padding: 14px;
      background: rgba(246, 241, 232, 0.76);
      border: 1px solid rgba(20, 33, 61, 0.08);
    }
    .dual-section strong {
      display: block;
      margin-bottom: 8px;
      font-size: 13px;
      letter-spacing: 0.06em;
      text-transform: uppercase;
      color: var(--accent);
    }
    .split {
      display: grid;
      grid-template-columns: 0.55fr 1.45fr;
      gap: 16px;
    }
    .preview-frame {
      border-radius: 20px;
      min-height: 420px;
      padding: 18px;
      background: linear-gradient(180deg, rgba(255,255,255,0.92), rgba(255,255,255,0.68));
      border: 1px solid rgba(20, 33, 61, 0.08);
      overflow: auto;
    }
    .preview-frame h1,
    .preview-frame h2,
    .preview-frame h3,
    .preview-frame h4 {
      margin: 0 0 12px;
      line-height: 1.15;
    }
    .preview-frame p,
    .preview-frame li {
      line-height: 1.6;
      margin: 0 0 10px;
    }
    .preview-frame ul {
      margin: 0 0 12px 18px;
      padding: 0;
    }
    .preview-frame table {
      width: 100%;
      border-collapse: collapse;
      margin: 12px 0;
      font-size: 14px;
    }
    .preview-frame th,
    .preview-frame td {
      border-bottom: 1px solid rgba(20, 33, 61, 0.08);
      padding: 8px 10px 8px 0;
      text-align: left;
    }
    .report-list {
      display: grid;
      gap: 10px;
      max-height: 420px;
      overflow: auto;
    }
    .report-item {
      border-radius: 16px;
      padding: 12px 14px;
      border: 1px solid rgba(20, 33, 61, 0.08);
      background: rgba(255,255,255,0.72);
      cursor: pointer;
      transition: transform 120ms ease, border-color 120ms ease;
    }
    .report-item:hover { transform: translateY(-1px); }
    .report-item.active {
      border-color: rgba(13, 92, 99, 0.38);
      background: rgba(13, 92, 99, 0.08);
    }
    .toolbar {
      display: flex;
      gap: 12px;
      flex-wrap: wrap;
      align-items: end;
      margin-bottom: 16px;
    }
    .toolbar .grow { flex: 1 1 240px; }
    .result-toolbar {
      display: flex;
      justify-content: space-between;
      gap: 12px;
      align-items: center;
      flex-wrap: wrap;
      margin-top: 8px;
    }
    .pagination {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      align-items: center;
      margin-top: 18px;
    }
    .pagination button {
      padding: 10px 13px;
      border-radius: 12px;
      font-size: 13px;
      box-shadow: none;
    }
    .pagination button.page-current {
      background: linear-gradient(135deg, var(--ink), #284a72);
    }
    .pagination button.page-ghost {
      background: rgba(20, 33, 61, 0.08);
      color: var(--ink);
    }
    .filter-actions {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      align-items: center;
    }
    .ghost-button {
      appearance: none;
      border: 1px solid rgba(20, 33, 61, 0.12);
      background: rgba(255,255,255,0.76);
      color: var(--ink);
      box-shadow: none;
      padding: 11px 14px;
      border-radius: 12px;
      font: inherit;
      cursor: pointer;
    }
    .ghost-button:hover {
      transform: translateY(-1px);
    }
    .quick-filters {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-top: 12px;
    }
    .quick-chip {
      appearance: none;
      border: 0;
      background: rgba(13, 92, 99, 0.10);
      color: var(--teal);
      border-radius: 999px;
      padding: 8px 12px;
      font-size: 12px;
      cursor: pointer;
    }
    .quick-chip:hover {
      transform: translateY(-1px);
    }
    .match-reasons {
      margin-top: 10px;
      font-size: 12px;
      color: var(--muted);
    }
    mark {
      background: rgba(197, 139, 42, 0.28);
      color: inherit;
      border-radius: 4px;
      padding: 0 2px;
    }
    .fold-panel {
      background: transparent;
    }
    .fold-panel > summary {
      list-style: none;
      cursor: pointer;
    }
    .fold-panel > summary::-webkit-details-marker {
      display: none;
    }
    #run-history-fold > summary {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      padding: 2px 0;
    }
    #run-history-fold > summary::before {
      content: "";
      width: 0;
      height: 0;
      border-top: 5px solid transparent;
      border-bottom: 5px solid transparent;
      border-left: 7px solid var(--muted);
      transition: transform 160ms ease, border-left-color 160ms ease;
      transform-origin: 35% 50%;
    }
    #run-history-fold > summary .topic-label {
      margin: 0;
      line-height: 1;
    }
    #run-history-fold[open] > summary::before {
      transform: rotate(90deg);
      border-left-color: var(--ink);
    }
    body[data-view="overview"] #papers-section .search-grid,
    body[data-view="overview"] #papers-section .result-toolbar,
    body[data-view="overview"] #papers-section .pagination,
    body[data-view="overview"] #papers-section .quick-filters {
      display: none;
    }
    body[data-view="overview"] #reports-section .report-list,
    body[data-view="overview"] #reports-toolbar {
      display: none;
    }
    body[data-view="overview"] #reports-section .split {
      grid-template-columns: 1fr;
    }
    body[data-view="overview"] #report-preview {
      min-height: 260px;
      max-height: 340px;
    }
    #queue-section {
      display: none;
    }
    body[data-view="papers"] #stats-section,
    body[data-view="papers"] #run-section,
    body[data-view="papers"] #topics-section,
    body[data-view="papers"] #reports-section {
      display: none;
    }
    body[data-view="overview"] #reports-section {
      display: none;
    }
    body[data-view="overview"] .grid {
      grid-template-columns: 1fr;
    }
    body[data-view="overview"] #run-section {
      margin-top: 18px;
    }
    body[data-view="overview"] #papers {
      grid-template-columns: repeat(2, minmax(0, 1fr));
    }
    body[data-view="reports"] #stats-section,
    body[data-view="reports"] #topics-section,
    body[data-view="reports"] #papers-section,
    body[data-view="reports"] #run-section {
      display: none;
    }
    body[data-view="reports"] #reports-section {
      margin-top: 18px;
    }
    body[data-view="reports"] .grid {
      grid-template-columns: 1fr;
    }
    body[data-view="papers"] #papers-section .panel-action,
    body[data-view="reports"] #reports-section .panel-action {
      display: none;
    }
    .paper-card.compact {
      padding: 16px 18px;
    }
    .paper-card.compact .paper-title {
      font-size: 18px;
    }
    .paper-card.compact .paper-meta {
      margin-top: 8px;
    }
    .paper-card.compact .topic-label {
      margin-top: 10px;
    }
    .paper-card.compact .paper-actions {
      margin-top: 10px;
    }
    a {
      color: var(--teal);
      text-decoration: none;
    }
    a:hover { text-decoration: underline; }
    .mobile-nav {
      display: none;
    }
    @media (max-width: 1080px) {
      .grid,
      .split {
        grid-template-columns: 1fr;
      }
      .topnav {
        grid-template-columns: 1fr;
      }
      .visual-grid {
        grid-template-columns: 1fr;
      }
      .stats {
        grid-template-columns: repeat(2, minmax(0, 1fr));
      }
      .run-entry-head {
        flex-direction: column;
        align-items: flex-start;
      }
      .run-entry-actions {
        width: 100%;
      }
      .run-entry-actions .status-banner {
        min-width: 0;
        width: 100%;
      }
      .search-grid {
        grid-template-columns: 1fr 1fr;
      }
      .dual {
        grid-template-columns: 1fr;
      }
      .search-wide {
        grid-column: span 2;
      }
    }
    @media (max-width: 640px) {
      body { padding-bottom: 86px; }
      .shell { width: min(100% - 18px, 100%); margin-top: 10px; }
      .hero, .panel { border-radius: 22px; }
      .title { max-width: none; }
      .stats,
      .control-grid,
      .search-grid {
        grid-template-columns: 1fr;
      }
      .search-wide {
        grid-column: span 1;
      }
      body[data-view="overview"] #papers {
        grid-template-columns: 1fr;
      }
      .mobile-nav {
        position: fixed;
        left: 12px;
        right: 12px;
        bottom: 12px;
        z-index: 20;
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 8px;
        padding: 10px;
        border-radius: 20px;
        background: rgba(20, 33, 61, 0.92);
        box-shadow: 0 20px 40px rgba(20, 33, 61, 0.2);
      }
      .mobile-nav a {
        display: block;
        text-align: center;
        color: rgba(255,255,255,0.84);
        padding: 10px 8px;
        border-radius: 14px;
        font-size: 12px;
        letter-spacing: 0.08em;
        text-transform: uppercase;
      }
      .mobile-nav a.active {
        background: rgba(255,255,255,0.14);
        color: #fff;
      }
    }
  </style>
</head>
<body data-view="__PAGE_VIEW__">
  <div class="shell">
    <section class="hero">
      <div class="eyebrow" id="hero-eyebrow">Daily Research Radar</div>
      <h1 class="title" id="hero-title">arXiv LLM Watch</h1>
      <p class="subtitle" id="hero-subtitle">本地 dashboard，支持实时查看热点、自动刷新 Markdown 报告预览，以及按关键词、状态、分类搜索论文。数据直接来自你的 SQLite 和日报文件。</p>
      <div class="hero-meta" id="hero-meta"></div>
      <nav class="topnav">
        <a href="/" data-nav="overview">
          <span class="nav-label" data-i18n="nav.overview.label">Overview</span>
          <span class="nav-copy" data-i18n="nav.overview.copy">看今日快照、热点和代表论文，适合先扫一眼。</span>
        </a>
        <a href="/papers" data-nav="papers">
          <span class="nav-label" data-i18n="nav.papers.label">Papers</span>
          <span class="nav-copy" data-i18n="nav.papers.copy">进入论文工作台，做搜索、筛选、排序和翻页。</span>
        </a>
        <a href="/reports" data-nav="reports">
          <span class="nav-label" data-i18n="nav.reports.label">Reports</span>
          <span class="nav-copy" data-i18n="nav.reports.copy">查看历史日报和完整 Markdown 预览。</span>
        </a>
      </nav>
    </section>

    <section class="panel" id="run-section">
      <div class="panel-head">
        <div>
          <h2 class="panel-title" data-i18n="run.panel.title">Run Batch</h2>
          <div class="panel-subtitle" data-i18n="run.panel.subtitle">首页入口。先拉取新论文，再刷新今天的热点、代表论文和报告。</div>
        </div>
        <a class="panel-action" href="/reports" data-i18n="run.archive_link">Open Run Archive</a>
      </div>
      <div class="panel-body">
        <div class="run-entry">
          <div class="run-entry-head">
            <div class="run-entry-copy">
              <div class="topic-label" data-i18n="run.entry.label">Start Today&apos;s Refresh</div>
              <p class="muted" data-i18n="run.entry.copy">Fetch the latest arXiv papers, analyze up to the configured cap with Ark, and update the dashboard in one step.</p>
            </div>
            <div class="run-entry-actions">
              <button id="run-button" data-i18n="run.button">Run Batch</button>
              <div class="status-banner muted" id="run-status" data-i18n="run.idle">Idle.</div>
            </div>
          </div>
        </div>
        <div class="control-grid">
          <div>
            <label for="query-keywords" data-i18n="run.keywords">Fetch Keywords</label>
            <input id="query-keywords" type="text" placeholder="llm, reasoning, alignment">
          </div>
          <div>
            <label for="max-results" data-i18n="run.max_results">Max Results</label>
            <input id="max-results" type="number" min="1">
          </div>
          <div>
            <label for="lookback-days" data-i18n="run.lookback">Lookback Days</label>
            <input id="lookback-days" type="number" min="1">
          </div>
          <div>
            <label for="analysis-limit" data-i18n="run.analysis_limit">Analysis Limit</label>
            <input id="analysis-limit" type="number" min="1">
          </div>
          <div>
            <label for="report-paper-limit" data-i18n="run.report_limit">Report Display Limit</label>
            <input id="report-paper-limit" type="number" min="1">
          </div>
        </div>
        <p class="muted" data-i18n="run.report_limit_help">报告展示篇数只控制日报里展示多少篇已分析论文，不影响抓取数量和分析数量。</p>
        <details class="fold-panel" id="run-history-fold">
          <summary>
            <div class="topic-label" data-i18n="run.recent_runs">Recent Runs</div>
          </summary>
          <div class="run-history" id="run-history"></div>
        </details>
      </div>
    </section>

    <div class="grid">
      <section class="panel" id="stats-section">
        <div class="panel-head">
          <div>
            <h2 class="panel-title" data-i18n="stats.panel.title">Snapshot</h2>
            <div class="panel-subtitle" data-i18n="stats.panel.subtitle">Current database view</div>
          </div>
        </div>
        <div class="panel-body">
          <div class="stats" id="stats"></div>
        </div>
      </section>
    </div>

    <div class="grid">
      <section class="panel" id="topics-section">
        <div class="panel-head">
          <div>
            <h2 class="panel-title" data-i18n="topics.panel.title">Hot Topics</h2>
            <div class="panel-subtitle" data-i18n="topics.panel.subtitle">Recent theme growth from analyzed papers, with clickable topic filters</div>
          </div>
        </div>
        <div class="panel-body">
          <div class="visual-grid">
            <div class="visual-card">
              <div class="topic-label" data-i18n="topics.category_share">Category Share</div>
              <div class="svg-wrap" id="category-chart"></div>
            </div>
            <div class="visual-card">
              <div class="topic-label" data-i18n="topics.topic_heat">Topic Heat</div>
              <div class="svg-wrap" id="topic-heat-chart"></div>
            </div>
          </div>
          <div id="topics"></div>
        </div>
      </section>

      <section class="panel" id="queue-section">
        <details class="fold-panel">
          <summary class="panel-head">
            <div>
              <h2 class="panel-title">Queue</h2>
              <div class="panel-subtitle">Internal processing queue</div>
            </div>
          </summary>
          <div class="panel-body">
            <div class="queue-stack" id="queue"></div>
          </div>
        </details>
      </section>
    </div>

    <section class="panel" style="margin-top: 18px;" id="reports-section">
      <div class="panel-head">
        <div>
          <h2 class="panel-title" id="reports-panel-title" data-i18n="reports.panel.title">Report Preview</h2>
          <div class="panel-subtitle" id="reports-panel-subtitle" data-i18n="reports.panel.subtitle">Auto-refreshes as new Markdown reports are generated</div>
        </div>
        <a class="panel-action" href="/reports" data-i18n="reports.open_reports">Open Reports</a>
      </div>
      <div class="panel-body">
        <div class="toolbar" id="reports-toolbar">
          <div class="grow">
            <label for="report-select" data-i18n="reports.selected_report">Selected Report</label>
            <select id="report-select"></select>
          </div>
          <button class="secondary" id="refresh-preview-button" data-i18n="reports.refresh_preview">Refresh Preview</button>
        </div>
        <div class="preview-meta" id="report-meta"></div>
        <div class="split" style="margin-top: 16px;">
          <div class="report-list" id="report-list"></div>
          <div class="preview-frame" id="report-preview" data-i18n="reports.loading_preview">Loading preview...</div>
        </div>
      </div>
    </section>

    <section class="panel" style="margin-top: 18px;" id="papers-section">
      <div class="panel-head">
        <div>
          <h2 class="panel-title" id="papers-panel-title" data-i18n="papers.panel.title">Keyword Search</h2>
          <div class="panel-subtitle" id="papers-panel-subtitle" data-i18n="papers.panel.subtitle">Search and filter papers across all statuses</div>
        </div>
        <a class="panel-action" href="/papers" data-i18n="papers.open_workspace">Open Papers Workspace</a>
      </div>
      <div class="panel-body">
        <div class="search-grid">
          <div class="search-wide">
            <label for="paper-query" data-i18n="papers.query.label">Keyword / Topic</label>
            <input id="paper-query" type="text" data-i18n-placeholder="papers.query.placeholder" placeholder="alignment, reasoning, memory-efficient ...">
          </div>
          <div>
            <label for="paper-status" data-i18n="papers.status.label">Status</label>
            <select id="paper-status">
              <option value="analyzed">Analyzed</option>
              <option value="pending">Pending</option>
              <option value="rejected">Rejected</option>
              <option value="error">Error</option>
              <option value="all">All</option>
            </select>
          </div>
          <div>
            <label for="paper-category" data-i18n="papers.category.label">Category</label>
            <select id="paper-category"></select>
          </div>
          <div>
            <label for="paper-tracked-topic" data-i18n="papers.tracked_topic.label">Tracked Topic</label>
            <select id="paper-tracked-topic"></select>
          </div>
          <div>
            <label for="paper-flag" data-i18n="papers.flag.label">Review Flag</label>
            <select id="paper-flag">
              <option value="all">All papers</option>
              <option value="starred">Starred only</option>
              <option value="ignored">Ignored only</option>
              <option value="manual">Manual edits only</option>
            </select>
          </div>
          <div>
            <label for="paper-days" data-i18n="papers.days.label">Time Window</label>
            <select id="paper-days">
              <option value="" data-i18n="papers.days.all">All time</option>
              <option value="1" data-i18n="papers.days.1">Last 24h</option>
              <option value="3" data-i18n="papers.days.3">Last 3d</option>
              <option value="7" data-i18n="papers.days.7">Last 7d</option>
              <option value="14" data-i18n="papers.days.14">Last 14d</option>
              <option value="30" data-i18n="papers.days.30">Last 30d</option>
            </select>
          </div>
          <div>
            <label for="paper-limit" data-i18n="papers.limit.label">Limit</label>
            <input id="paper-limit" type="number" min="1" value="12">
          </div>
          <div>
            <label for="paper-sort" data-i18n="papers.sort.label">Sort By</label>
            <select id="paper-sort">
              <option value="published_desc" data-i18n="papers.sort.published_desc">Latest first</option>
              <option value="published_asc" data-i18n="papers.sort.published_asc">Oldest first</option>
              <option value="score_desc" data-i18n="papers.sort.score_desc">Highest score</option>
              <option value="title_asc" data-i18n="papers.sort.title_asc">Title A-Z</option>
            </select>
          </div>
        </div>
        <div class="result-toolbar">
          <div class="filter-actions">
            <button class="secondary" id="paper-search-button" data-i18n="papers.search">Search Papers</button>
            <button class="ghost-button" id="paper-clear-button" type="button" data-i18n="papers.clear">Clear Filters</button>
            <button class="ghost-button" id="paper-copy-link-button" type="button" data-i18n="papers.copy_link">Copy Search Link</button>
          </div>
          <div class="muted" id="paper-search-meta" data-i18n="papers.meta.default">Showing latest analyzed papers.</div>
        </div>
        <div class="quick-filters" id="quick-topic-filters"></div>
        <div class="paper-stack" id="papers" style="margin-top: 18px;"></div>
        <div class="pagination" id="paper-pagination"></div>
      </div>
    </section>
  </div>
  <nav class="mobile-nav">
    <a href="/" data-nav="overview" data-i18n="nav.overview.label">Overview</a>
    <a href="/papers" data-nav="papers" data-i18n="nav.papers.label">Papers</a>
    <a href="/reports" data-nav="reports" data-i18n="nav.reports.label">Reports</a>
  </nav>

  <script>
    const PAGE_VIEW = "__PAGE_VIEW__";
    const state = {
      lang: "zh",
      initialized: false,
      selectedReport: "",
      reportCacheKey: "",
      paperQueryKey: "",
      refreshTimer: null,
      initialSearch: null,
      currentPaperPage: 1,
      lastPayload: null,
      lastPaperPayload: null,
    };

    const I18N = {
      zh: {
        "page.overview": "总览 - LLM Paper Radar",
        "page.papers": "论文 - LLM Paper Radar",
        "page.reports": "报告 - LLM Paper Radar",
        "nav.overview.label": "总览",
        "nav.overview.copy": "看今日快照、热点和代表论文，适合先扫一眼。",
        "nav.papers.label": "论文",
        "nav.papers.copy": "进入论文工作台，做搜索、筛选、排序和翻页。",
        "nav.reports.label": "报告",
        "nav.reports.copy": "查看历史日报和完整 Markdown 预览。",
        "hero.overview.eyebrow": "每日研究雷达",
        "hero.overview.title": "LLM Paper Radar",
        "hero.overview.subtitle": "每天先看最值得读的方向和代表论文，把首页压缩成一眼能扫完的研究总览。",
        "hero.papers.eyebrow": "论文工作台",
        "hero.papers.title": "LLM 论文工作台",
        "hero.papers.subtitle": "专门的论文工作台。按关键词、主题、状态、分类和排序方式筛选，聚焦找论文，不混入日报噪音。",
        "hero.reports.eyebrow": "报告归档",
        "hero.reports.title": "日报归档",
        "hero.reports.subtitle": "集中查看历史 Markdown 日报，切换版本、预览内容，并快速回到论文工作台继续深挖。",
        "hero.meta.model": "模型：{value}",
        "hero.meta.categories": "分类：{value}",
        "hero.meta.keywords": "抓取关键词：{value}",
        "hero.meta.analysis_cap": "分析上限：{value}",
        "hero.meta.report_archive": "报告数量：{value}",
        "run.panel.title": "运行批次",
        "run.panel.subtitle": "首页入口。先拉取新论文，再刷新今天的热点、代表论文和报告。",
        "run.archive_link": "打开运行归档",
        "run.entry.label": "启动今天的刷新",
        "run.entry.copy": "一键抓取最新 arXiv 论文，按设定上限调用 Ark 分析，并更新首页、搜索页和报告页。",
        "run.button": "运行批次",
        "run.idle": "空闲中。",
        "run.lookback": "回看天数",
        "run.max_results": "抓取上限",
        "run.keywords": "抓取关键词",
        "run.analysis_limit": "分析上限",
        "run.report_limit": "报告展示篇数",
        "run.report_limit_help": "报告展示篇数只控制日报里展示多少篇已分析论文，不影响抓取数量和分析数量。",
        "run.recent_runs": "最近运行",
        "run.history.none": "还没有完成的运行记录。",
        "run.history.fetched": "已抓取",
        "run.history.queued": "已入分析队列",
        "run.history.analyzed": "已分析",
        "run.history.in_progress": "正在抓取和分析，完成后会生成报告。",
        "run.history.completed_without_report": "已完成，但没有产出报告文件。",
        "run.status.running": "正在运行，开始于 {time}。",
        "run.status.failed": "上次运行失败于 {time}：{error}",
        "run.status.finished": "上次运行完成于 {time}。{suffix}",
        "run.status.report_suffix": "报告：{path}",
        "run.status.idle": "空闲中。",
        "run.status.success": "成功",
        "run.status.failed_label": "失败",
        "run.status.running_label": "运行中",
        "stats.panel.title": "快照",
        "stats.panel.subtitle": "当前数据库视图",
        "stats.analyzed": "已分析",
        "stats.starred": "已收藏",
        "stats.topics": "主题",
        "stats.reports": "报告",
        "topics.panel.title": "热点主题",
        "topics.panel.subtitle": "基于已分析论文计算的近期主题增速，可直接点击筛选。",
        "topics.category_share": "分类占比",
        "topics.topic_heat": "主题热度",
        "topics.recent": "近期",
        "topics.baseline": "基线",
        "topics.no_analyzed": "当前时间窗口还没有已分析论文。",
        "topics.no_trends": "当前还没有可用的主题趋势数据。",
        "topics.chart.analyzed": "已分析",
        "reports.panel.title": "报告预览",
        "reports.panel.subtitle": "Markdown 报告生成后会自动刷新。",
        "reports.open_reports": "打开报告页",
        "reports.selected_report": "当前报告",
        "reports.refresh_preview": "刷新预览",
        "reports.loading_preview": "正在加载报告预览...",
        "reports.no_reports": "还没有生成报告。",
        "reports.no_report_selected": "当前没有选中的报告。",
        "reports.no_report_available": "当前没有可用报告。",
        "reports.summary_title": "最新报告摘要",
        "reports.summary_subtitle": "仅显示最新 Markdown 报告的紧凑预览，完整内容请进入报告页。",
        "reports.archive_title": "报告归档",
        "reports.archive_subtitle": "浏览已生成的 Markdown 报告，切换版本并查看完整内容。",
        "reports.bytes": "{value} 字节",
        "reports.open_full_line": "- 打开 [/reports](/reports) 查看完整报告。",
        "papers.panel.title": "关键词搜索",
        "papers.panel.subtitle": "按关键词、分类、主题和状态筛选论文。",
        "papers.open_workspace": "打开论文工作台",
        "papers.query.label": "关键词 / 主题",
        "papers.query.placeholder": "如：对齐、reasoning、memory-efficient ...",
        "papers.status.label": "状态",
        "papers.category.label": "分类",
        "papers.tracked_topic.label": "受控主题",
        "papers.flag.label": "人工标记",
        "papers.days.label": "时间窗口",
        "papers.limit.label": "每页数量",
        "papers.sort.label": "排序方式",
        "papers.days.all": "全部时间",
        "papers.days.1": "最近 24 小时",
        "papers.days.3": "最近 3 天",
        "papers.days.7": "最近 7 天",
        "papers.days.14": "最近 14 天",
        "papers.days.30": "最近 30 天",
        "papers.sort.published_desc": "最新优先",
        "papers.sort.published_asc": "最早优先",
        "papers.sort.score_desc": "分数最高",
        "papers.sort.title_asc": "标题 A-Z",
        "papers.search": "搜索论文",
        "papers.clear": "清空筛选",
        "papers.copy_link": "复制搜索链接",
        "papers.meta.default": "展示最新已分析论文。",
        "papers.top_papers_title": "代表论文",
        "papers.top_papers_subtitle": "当前窗口中的精选论文。完整搜索请进入论文工作台。",
        "papers.top_papers_meta": "当前窗口展示 {count} 篇代表论文。",
        "papers.no_results": "当前筛选条件下没有论文。",
        "papers.meta.summary": "显示 {start}-{end} / 共 {total} 篇，第 {page}/{pages} 页，排序：{sort}，状态：{status}，分类：{category}，查询：\\\"{query}\\\"{extras}。",
        "papers.meta.extra": "，{label}：{value}",
        "papers.copy_success": "已复制当前搜索链接：{link}",
        "papers.copy_fail": "复制失败，当前链接：{link}",
        "papers.status.analyzed": "已分析",
        "papers.status.pending": "待分析",
        "papers.category.all": "全部分类 ({count})",
        "papers.tracked_topic.all": "全部受控主题 ({count})",
        "papers.flag.all": "全部论文 ({count})",
        "papers.flag.starred": "仅看收藏 ({count})",
        "papers.flag.ignored": "仅看忽略 ({count})",
        "papers.flag.manual": "仅看人工修改 ({count})",
        "pagination.prev": "上一页",
        "pagination.next": "下一页",
        "paper.flag.starred": "已收藏",
        "paper.flag.ignored": "已忽略",
        "paper.flag.manual": "人工标注",
        "paper.view_details": "查看详情",
        "paper.open_arxiv": "打开 arXiv",
        "paper.tracked_topics": "受控主题",
        "paper.model_topics": "模型主题",
        "paper.matched_in": "命中位置：{value}",
        "paper.manual_topics": "人工主题：{value}",
        "paper.open_detail": "打开详情",
        "paper.full_detail": "完整详情",
        "paper.structured_view": "结构化视图",
        "paper.summary": "摘要",
        "paper.problem": "问题",
        "paper.method": "方法",
        "paper.findings": "结果",
        "paper.limitations": "局限",
        "paper.status.pending": "等待分析。",
        "paper.status.rejected": "已拒绝。",
        "paper.status.error": "分析失败。",
        "paper.status.no_summary": "暂无摘要。",
        "common.na": "暂无",
        "common.copy_failed": "复制失败",
        "common.failed_to_start_run": "启动运行失败。",
      },
      en: {
        "page.overview": "Overview - LLM Paper Radar",
        "page.papers": "Papers - LLM Paper Radar",
        "page.reports": "Reports - LLM Paper Radar",
        "nav.overview.label": "Overview",
        "nav.overview.copy": "Scan today's snapshot, hot topics, and representative papers.",
        "nav.papers.label": "Papers",
        "nav.papers.copy": "Search, filter, sort, and page through the paper workspace.",
        "nav.reports.label": "Reports",
        "nav.reports.copy": "Browse the report archive and full Markdown previews.",
        "hero.overview.eyebrow": "Daily Research Radar",
        "hero.overview.title": "LLM Paper Radar",
        "hero.overview.subtitle": "Start with the most valuable directions and representative papers in a compressed daily overview.",
        "hero.papers.eyebrow": "Paper Explorer",
        "hero.papers.title": "LLM Paper Workspace",
        "hero.papers.subtitle": "A focused paper workspace for search, topic filters, status filters, categories, and ranking without report noise.",
        "hero.reports.eyebrow": "Report Archive",
        "hero.reports.title": "Daily Report Archive",
        "hero.reports.subtitle": "Browse generated Markdown reports, switch versions, and jump back into paper exploration.",
        "hero.meta.model": "Model: {value}",
        "hero.meta.categories": "Categories: {value}",
        "hero.meta.keywords": "Fetch keywords: {value}",
        "hero.meta.analysis_cap": "Analysis cap: {value}",
        "hero.meta.report_archive": "Report archive: {value}",
        "run.panel.title": "Run Batch",
        "run.panel.subtitle": "Primary homepage entry. Fetch fresh papers first, then refresh today's topics, representative papers, and reports.",
        "run.archive_link": "Open Run Archive",
        "run.entry.label": "Start Today's Refresh",
        "run.entry.copy": "Fetch the latest arXiv papers, analyze up to the configured cap with Ark, and update the dashboard in one step.",
        "run.button": "Run Batch",
        "run.idle": "Idle.",
        "run.lookback": "Lookback Days",
        "run.max_results": "Max Results",
        "run.keywords": "Fetch Keywords",
        "run.analysis_limit": "Analysis Limit",
        "run.report_limit": "Report Display Limit",
        "run.report_limit_help": "This only controls how many analyzed papers appear in the daily report, not how many papers are fetched or analyzed.",
        "run.recent_runs": "Recent Runs",
        "run.history.none": "No completed runs yet.",
        "run.history.fetched": "fetched",
        "run.history.queued": "queued",
        "run.history.analyzed": "analyzed",
        "run.history.in_progress": "Fetching and analyzing. A report will appear after the run finishes.",
        "run.history.completed_without_report": "Completed without report output.",
        "run.status.running": "Running since {time}.",
        "run.status.failed": "Last run failed at {time}: {error}",
        "run.status.finished": "Last run finished at {time}. {suffix}",
        "run.status.report_suffix": "Report: {path}",
        "run.status.idle": "Idle.",
        "run.status.success": "success",
        "run.status.failed_label": "failed",
        "run.status.running_label": "running",
        "stats.panel.title": "Snapshot",
        "stats.panel.subtitle": "Current database view",
        "stats.analyzed": "Analyzed",
        "stats.starred": "Starred",
        "stats.topics": "Topics",
        "stats.reports": "Reports",
        "topics.panel.title": "Hot Topics",
        "topics.panel.subtitle": "Recent theme growth from analyzed papers, with clickable topic filters.",
        "topics.category_share": "Category Share",
        "topics.topic_heat": "Topic Heat",
        "topics.recent": "recent",
        "topics.baseline": "baseline",
        "topics.no_analyzed": "No analyzed papers in the current window.",
        "topics.no_trends": "No topic trend data yet.",
        "topics.chart.analyzed": "Analyzed",
        "reports.panel.title": "Report Preview",
        "reports.panel.subtitle": "Auto-refreshes as new Markdown reports are generated.",
        "reports.open_reports": "Open Reports",
        "reports.selected_report": "Selected Report",
        "reports.refresh_preview": "Refresh Preview",
        "reports.loading_preview": "Loading preview...",
        "reports.no_reports": "No reports yet.",
        "reports.no_report_selected": "No report selected.",
        "reports.no_report_available": "No report available.",
        "reports.summary_title": "Latest Report Summary",
        "reports.summary_subtitle": "A compact preview of the newest Markdown report. Open Reports for the full archive.",
        "reports.archive_title": "Report Archive",
        "reports.archive_subtitle": "Browse generated Markdown reports, switch versions, and inspect the full content.",
        "reports.bytes": "{value} bytes",
        "reports.open_full_line": "- Open [/reports](/reports) for the full report.",
        "papers.panel.title": "Keyword Search",
        "papers.panel.subtitle": "Search and filter papers across keywords, categories, topics, and status.",
        "papers.open_workspace": "Open Papers Workspace",
        "papers.query.label": "Keyword / Topic",
        "papers.query.placeholder": "alignment, reasoning, memory-efficient ...",
        "papers.status.label": "Status",
        "papers.category.label": "Category",
        "papers.tracked_topic.label": "Tracked Topic",
        "papers.flag.label": "Review Flag",
        "papers.days.label": "Time Window",
        "papers.limit.label": "Limit",
        "papers.sort.label": "Sort By",
        "papers.days.all": "All time",
        "papers.days.1": "Last 24h",
        "papers.days.3": "Last 3d",
        "papers.days.7": "Last 7d",
        "papers.days.14": "Last 14d",
        "papers.days.30": "Last 30d",
        "papers.sort.published_desc": "Latest first",
        "papers.sort.published_asc": "Oldest first",
        "papers.sort.score_desc": "Highest score",
        "papers.sort.title_asc": "Title A-Z",
        "papers.search": "Search Papers",
        "papers.clear": "Clear Filters",
        "papers.copy_link": "Copy Search Link",
        "papers.meta.default": "Showing latest analyzed papers.",
        "papers.top_papers_title": "Top Papers",
        "papers.top_papers_subtitle": "A short list of recent analyzed papers. Open the Papers workspace for full search.",
        "papers.top_papers_meta": "Showing {count} representative paper(s) from the current report window.",
        "papers.no_results": "No papers match the current filters.",
        "papers.meta.summary": "Showing {start}-{end} of {total} paper(s), page {page}/{pages}, sort={sort}, status={status}, category={category}, query=\\\"{query}\\\"{extras}.",
        "papers.meta.extra": ", {label}={value}",
        "papers.copy_success": "Copied search link: {link}",
        "papers.copy_fail": "Copy failed. Current link: {link}",
        "papers.status.analyzed": "Analyzed",
        "papers.status.pending": "Pending",
        "papers.category.all": "All categories ({count})",
        "papers.tracked_topic.all": "All tracked topics ({count})",
        "papers.flag.all": "All papers ({count})",
        "papers.flag.starred": "Starred only ({count})",
        "papers.flag.ignored": "Ignored only ({count})",
        "papers.flag.manual": "Manual edits only ({count})",
        "pagination.prev": "Prev",
        "pagination.next": "Next",
        "paper.flag.starred": "Starred",
        "paper.flag.ignored": "Ignored",
        "paper.flag.manual": "Manual review",
        "paper.view_details": "View details",
        "paper.open_arxiv": "Open arXiv",
        "paper.tracked_topics": "Tracked Topics",
        "paper.model_topics": "Model Topics",
        "paper.matched_in": "Matched in: {value}",
        "paper.manual_topics": "Manual topics: {value}",
        "paper.open_detail": "Open Detail",
        "paper.full_detail": "Full Detail",
        "paper.structured_view": "Structured View",
        "paper.summary": "Summary",
        "paper.problem": "Problem",
        "paper.method": "Method",
        "paper.findings": "Findings",
        "paper.limitations": "Limitations",
        "paper.status.pending": "Pending analysis.",
        "paper.status.rejected": "Rejected.",
        "paper.status.error": "Unknown error.",
        "paper.status.no_summary": "No summary.",
        "common.na": "n/a",
        "common.copy_failed": "Copy failed",
        "common.failed_to_start_run": "Failed to start run.",
      },
    };

    const TRACKED_TOPIC_LABELS = {
      zh: {
        "reasoning": "推理",
        "agents & tool use": "智能体与工具使用",
        "rag & retrieval": "RAG 与检索",
        "alignment & safety": "对齐与安全",
        "evaluation & llm as a judge": "评测与 LLM 裁判",
        "hallucination & factuality": "幻觉与事实性",
        "post training & preference optimization": "后训练与偏好优化",
        "training efficiency": "训练效率",
        "inference efficiency": "推理效率",
        "long context & memory": "长上下文与记忆",
        "multimodal llm": "多模态 LLM",
        "mechanistic interpretability": "机制可解释性",
        "synthetic data & distillation": "合成数据与蒸馏",
        "coding & program synthesis": "编码与程序合成",
        "benchmarks & datasets": "基准与数据集",
      },
      en: {},
    };

    function currentLang() {
      return state.lang === "en" ? "en" : "zh";
    }

    function t(key, params = {}) {
      const lang = currentLang();
      const template = I18N[lang]?.[key] ?? I18N.en[key] ?? key;
      return String(template).replace(/\{(\w+)\}/g, (_, name) => String(params[name] ?? ""));
    }

    function displayTopicLabel(topic) {
      const value = String(topic || "");
      const normalized = value.trim().toLowerCase();
      if (currentLang() === "zh") {
        return TRACKED_TOPIC_LABELS.zh[normalized] || value;
      }
      return value;
    }

    function applyStaticI18n(root = document) {
      document.documentElement.lang = "zh-CN";
      document.title = t(`page.${PAGE_VIEW}`);
      root.querySelectorAll("[data-i18n]").forEach((node) => {
        node.textContent = t(node.dataset.i18n);
      });
      root.querySelectorAll("[data-i18n-placeholder]").forEach((node) => {
        node.placeholder = t(node.dataset.i18nPlaceholder);
      });
    }

    function escapeHtml(value) {
      return String(value ?? "")
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#39;");
    }

    function formatInline(text) {
      const escaped = escapeHtml(text);
      return escaped
        .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
        .replace(/\[([^\]]+)\]\((https?:\/\/[^\s)]+)\)/g, '<a href="$2" target="_blank" rel="noreferrer">$1</a>');
    }

    function formatTime(value) {
      if (!value) return t("common.na");
      return new Date(value).toLocaleString(currentLang() === "zh" ? "zh-CN" : "en-US");
    }

    function formatTopicMomentum(topic) {
      if (topic?.momentum_label) return topic.momentum_label;
      const current = Number(topic?.current_count || 0);
      const baseline = Number(topic?.baseline_count || 0);
      if (baseline <= 0) {
        return `New (0 -> ${current})`;
      }
      const growth = ((current - baseline) / baseline) * 100;
      const sign = growth >= 0 ? "+" : "";
      return `${sign}${Math.round(growth)}% (${baseline} -> ${current})`;
    }

    function truncateText(text, maxLength = 220) {
      const value = String(text ?? "").replace(/\s+/g, " ").trim();
      if (!value) return "";
      if (value.length <= maxLength) return value;
      return `${value.slice(0, maxLength - 1).trimEnd()}...`;
    }

    function escapeRegExp(value) {
      const slash = String.fromCharCode(92);
      const specials = new Set([slash, ".", "*", "+", "?", "^", "$", "{", "}", "(", ")", "|", "[", "]"]);
      return Array.from(String(value ?? ""))
        .map((char) => (specials.has(char) ? `${slash}${char}` : char))
        .join("");
    }

    function highlightText(text, query) {
      const safe = escapeHtml(text);
      const tokens = String(query || "").trim().split(/\s+/).filter((item) => item.length > 1).slice(0, 4);
      if (!tokens.length) return safe;
      return tokens.reduce((result, token) => {
        const pattern = new RegExp(`(${escapeRegExp(escapeHtml(token))})`, "ig");
        return result.replace(pattern, "<mark>$1</mark>");
      }, safe);
    }

    function paperAnchorId(entryId) {
      return `paper-${String(entryId || "").replace(/[^a-zA-Z0-9]+/g, "-")}`;
    }

    function paperDetailUrl(entryId) {
      const params = new URLSearchParams({
        entry_id: entryId,
        return_to: `${window.location.pathname}${window.location.search}#${paperAnchorId(entryId)}`,
      });
      return `/paper?${params.toString()}`;
    }

    function localizedPaperText(paper, field) {
      const zh = paper?.[`${field}_zh`];
      const en = paper?.[`${field}_en`];
      if (currentLang() === "zh") {
        return zh || en || "";
      }
      return en || zh || "";
    }

    function localizedSummary(paper) {
      return localizedPaperText(paper, "summary") || paper?.summary || t("paper.status.no_summary");
    }

    function formatRunStatus(value) {
      if (value === "success") return t("run.status.success");
      if (value === "failed") return t("run.status.failed_label");
      if (value === "running") return t("run.status.running_label");
      return value || t("common.na");
    }

    function searchBasePath() {
      return "/papers";
    }

    function parseSearchParams() {
      const params = new URLSearchParams(window.location.search);
      return {
        query: params.get("query") || "",
        status: params.get("status") || "analyzed",
        category: params.get("category") || "all",
        tracked_topic: params.get("tracked_topic") || "",
        flag: params.get("flag") || "all",
        days: params.get("days") || "",
        page: Math.max(1, Number(params.get("page") || 1)),
        sort: params.get("sort") || "published_desc",
        limit: Math.max(1, Number(params.get("limit") || 12)),
      };
    }

    function updateSearchUrl(query, status, category, trackedTopic, flag, days, page, sort, limit) {
      const params = new URLSearchParams();
      if (query) params.set("query", query);
      if (status && status !== "analyzed") params.set("status", status);
      if (category && category !== "all") params.set("category", category);
      if (trackedTopic) params.set("tracked_topic", trackedTopic);
      if (flag && flag !== "all") params.set("flag", flag);
      if (days) params.set("days", String(days));
      if (page && page > 1) params.set("page", String(page));
      if (sort && sort !== "published_desc") params.set("sort", sort);
      if (limit && Number(limit) !== 12) params.set("limit", String(limit));
      const base = searchBasePath();
      const next = params.toString() ? `${base}?${params.toString()}` : base;
      window.history.replaceState({}, "", next);
    }

    function setSearchFilters({ query, status, category, tracked_topic, flag, days, limit, sort }) {
      if (query !== undefined) document.getElementById("paper-query").value = query;
      if (status !== undefined) document.getElementById("paper-status").value = status;
      if (category !== undefined) document.getElementById("paper-category").value = category;
      if (tracked_topic !== undefined) document.getElementById("paper-tracked-topic").value = tracked_topic;
      if (flag !== undefined) document.getElementById("paper-flag").value = flag;
      if (days !== undefined) document.getElementById("paper-days").value = days;
      if (limit !== undefined) document.getElementById("paper-limit").value = limit;
      if (sort !== undefined) document.getElementById("paper-sort").value = sort;
    }

    async function applyTopicFilter(topic) {
      setSearchFilters({
        query: "",
        status: "analyzed",
        category: "all",
        tracked_topic: topic,
        flag: "all",
        days: "",
      });
      state.paperQueryKey = "";
      state.currentPaperPage = 1;
      if (PAGE_VIEW !== "papers") {
        const target = new URL(searchBasePath(), window.location.origin);
        target.searchParams.set("tracked_topic", topic);
        window.location.href = target.toString();
        return;
      }
      updateSearchUrl(
        "",
        "analyzed",
        "all",
        topic,
        "all",
        "",
        1,
        document.getElementById("paper-sort").value,
        document.getElementById("paper-limit").value,
      );
      await loadPaperResults(true, 1);
      document.getElementById("papers").scrollIntoView({ behavior: "smooth", block: "start" });
    }

    function applyActiveNav() {
      document.querySelectorAll("[data-nav]").forEach((node) => {
        if (node.dataset.nav === PAGE_VIEW) {
          node.classList.add("active");
        }
      });
    }

    function renderHero(config, reports) {
      const eyebrow = document.getElementById("hero-eyebrow");
      const title = document.getElementById("hero-title");
      const subtitle = document.getElementById("hero-subtitle");
      const viewCopy = {
        overview: {
          eyebrow: t("hero.overview.eyebrow"),
          title: t("hero.overview.title"),
          subtitle: t("hero.overview.subtitle"),
        },
        papers: {
          eyebrow: t("hero.papers.eyebrow"),
          title: t("hero.papers.title"),
          subtitle: t("hero.papers.subtitle"),
        },
        reports: {
          eyebrow: t("hero.reports.eyebrow"),
          title: t("hero.reports.title"),
          subtitle: t("hero.reports.subtitle"),
        },
      }[PAGE_VIEW] || {
        eyebrow: t("hero.overview.eyebrow"),
        title: t("hero.overview.title"),
        subtitle: t("hero.overview.subtitle"),
      };
      if (eyebrow) eyebrow.textContent = viewCopy.eyebrow;
      if (title) title.textContent = viewCopy.title;
      if (subtitle) subtitle.textContent = viewCopy.subtitle;
      const meta = document.getElementById("hero-meta");
      const keywordChip = (config.query_keywords || []).length
        ? `<span class="chip">${escapeHtml(t("hero.meta.keywords", { value: config.query_keywords.join(", ") }))}</span>`
        : "";
      meta.innerHTML = `
        <span class="chip">${escapeHtml(t("hero.meta.model", { value: config.model }))}</span>
        <span class="chip">${escapeHtml(t("hero.meta.categories", { value: config.categories.join(", ") }))}</span>
        ${keywordChip}
        <span class="chip">${escapeHtml(t("hero.meta.analysis_cap", { value: config.analysis_limit_per_run }))}</span>
        <span class="chip">${escapeHtml(t("hero.meta.report_archive", { value: reports.length || 0 }))}</span>
      `;
    }

    function renderQuickTopicFilters(topics) {
      const container = document.getElementById("quick-topic-filters");
      if (!topics?.length) {
        container.innerHTML = "";
        return;
      }
      container.innerHTML = topics.slice(0, 8).map((topic) => `
        <button type="button" class="quick-chip" data-topic-filter="${escapeHtml(topic.name)}">
          ${escapeHtml(displayTopicLabel(topic.name))} (${escapeHtml(topic.current_count)})
        </button>
      `).join("");
    }

    function configurePapersSectionForOverview(papers) {
      const title = document.getElementById("papers-panel-title");
      const subtitle = document.getElementById("papers-panel-subtitle");
      if (title) title.textContent = t("papers.top_papers_title");
      if (subtitle) subtitle.textContent = t("papers.top_papers_subtitle");
      renderPapers((papers || []).slice(0, 4), { compact: true, topicLimit: 3 });
      const meta = document.getElementById("paper-search-meta");
      if (meta) meta.textContent = t("papers.top_papers_meta", { count: Math.min((papers || []).length, 4) });
    }

    function renderRunHistory(history) {
      const container = document.getElementById("run-history");
      if (!container) return;
      if (!history?.length) {
        container.innerHTML = `<p class="muted">${escapeHtml(t("run.history.none"))}</p>`;
        return;
      }
      container.innerHTML = history.map((run) => `
        <article class="run-card">
          <div class="queue-meta" style="justify-content: space-between; margin-top: 0;">
            <strong>${escapeHtml(run.source)}</strong>
            <span>${escapeHtml(formatRunStatus(run.status))}</span>
          </div>
          <div class="queue-meta">
            <span>${escapeHtml(formatTime(run.started_at))}</span>
            <span>${escapeHtml(run.fetched_count)} ${escapeHtml(t("run.history.fetched"))}</span>
            <span>${escapeHtml(run.analysis_batch_count)} ${escapeHtml(t("run.history.queued"))}</span>
            <span>${escapeHtml(run.analyzed_count)} ${escapeHtml(t("run.history.analyzed"))}</span>
          </div>
          <div class="muted">${escapeHtml(formatRunHistoryDetail(run))}</div>
        </article>
      `).join("");
    }

    function syncRunHistoryFold(runState, history) {
      const fold = document.getElementById("run-history-fold");
      if (!fold) return;
      if (runState?.running || runState?.last_error) {
        fold.open = true;
      }
    }

    function formatRunHistoryDetail(run) {
      if (run.status === "running") {
        return t("run.history.in_progress");
      }
      if (run.error_message) {
        return run.error_message;
      }
      if (run.report_path) {
        return run.report_path;
      }
      return t("run.history.completed_without_report");
    }

    function configureReportsSection() {
      const title = document.getElementById("reports-panel-title");
      const subtitle = document.getElementById("reports-panel-subtitle");
      if (PAGE_VIEW === "overview") {
        if (title) title.textContent = t("reports.summary_title");
        if (subtitle) subtitle.textContent = t("reports.summary_subtitle");
        return;
      }
      if (title) title.textContent = t("reports.archive_title");
      if (subtitle) subtitle.textContent = t("reports.archive_subtitle");
    }

    function summarizeMarkdown(content, maxLines = 48) {
      const lines = content.split(/\\r?\\n/);
      const clipped = lines.slice(0, maxLines);
      if (lines.length > maxLines) {
        clipped.push("", t("reports.open_full_line"));
      }
      return clipped.join("\\n");
    }

    function renderStats(recent, facets, reports, topicTrends) {
      const stats = document.getElementById("stats");
      const items = [
        [t("stats.analyzed"), recent.analyzed || 0],
        [t("stats.starred"), facets?.flags?.starred || 0],
        [t("stats.topics"), topicTrends?.length || 0],
        [t("stats.reports"), reports?.length || 0],
      ];
      stats.innerHTML = items.map(([label, value]) => `
        <div class="stat">
          <div class="stat-label">${escapeHtml(label)}</div>
          <div class="stat-value">${escapeHtml(value)}</div>
        </div>
      `).join("");
    }

    function renderCategoryChart(categoryMix) {
      const container = document.getElementById("category-chart");
      if (!categoryMix.length) {
        container.innerHTML = `<p class="muted">${escapeHtml(t("topics.no_analyzed"))}</p>`;
        return;
      }

      const colors = ["#b23a48", "#0d5c63", "#c58b2a", "#5c6b73", "#7b5715", "#782629"];
      const total = categoryMix.reduce((sum, item) => sum + Number(item.count || 0), 0);
      const radius = 54;
      const circumference = 2 * Math.PI * radius;
      let offset = 0;
      const segments = categoryMix.map((item, index) => {
        const count = Number(item.count || 0);
        const length = total > 0 ? (count / total) * circumference : 0;
        const segment = `
          <circle
            cx="80"
            cy="80"
            r="${radius}"
            fill="none"
            stroke="${colors[index % colors.length]}"
            stroke-width="18"
            stroke-linecap="round"
            stroke-dasharray="${length} ${Math.max(circumference - length, 0)}"
            stroke-dashoffset="${-offset}"
            transform="rotate(-90 80 80)"
          />
        `;
        offset += length;
        return segment;
      }).join("");

      const legend = categoryMix.map((item, index) => {
        const count = Number(item.count || 0);
        const share = total > 0 ? Math.round((count / total) * 100) : 0;
        return `
          <div class="queue-meta" style="justify-content: space-between; margin: 6px 0 0;">
            <span><span style="display:inline-block;width:10px;height:10px;border-radius:999px;background:${colors[index % colors.length]};margin-right:8px;"></span>${escapeHtml(item.category)}</span>
            <span>${count} / ${share}%</span>
          </div>
        `;
      }).join("");

      container.innerHTML = `
        <svg viewBox="0 0 160 160" width="100%" height="180" aria-label="Category share chart">
          <circle cx="80" cy="80" r="${radius}" fill="none" stroke="rgba(20, 33, 61, 0.08)" stroke-width="18" />
          ${segments}
          <text x="80" y="74" text-anchor="middle" font-size="13" fill="#5c6b73">${escapeHtml(t("topics.chart.analyzed"))}</text>
          <text x="80" y="95" text-anchor="middle" font-size="26" font-weight="700" fill="#14213d">${total}</text>
        </svg>
        <div>${legend}</div>
      `;
    }

    function renderTopicHeatChart(topics) {
      const container = document.getElementById("topic-heat-chart");
      if (!topics.length) {
        container.innerHTML = `<p class="muted">${escapeHtml(t("topics.no_trends"))}</p>`;
        return;
      }

      const visible = topics.slice(0, 6);
      const maxCurrent = Math.max(...visible.map((topic) => Number(topic.current_count || 0)), 1);
      const width = 520;
      const rowHeight = 32;
      const height = 26 + (visible.length * rowHeight);
      const bars = visible.map((topic, index) => {
        const y = 8 + (index * rowHeight);
        const displayName = displayTopicLabel(topic.name);
        const label = displayName.length > 24 ? `${displayName.slice(0, 24)}...` : displayName;
        const value = Number(topic.current_count || 0);
        const barWidth = Math.max(24, Math.round((value / maxCurrent) * 210));
        const intensity = Math.min(0.95, 0.32 + (value / maxCurrent) * 0.5);
        return `
          <text x="8" y="${y + 15}" font-size="11" fill="#5c6b73">${escapeHtml(label)}</text>
          <rect x="220" y="${y}" width="${barWidth}" height="16" rx="8" fill="rgba(178, 58, 72, ${intensity})"></rect>
          <text x="${220 + barWidth + 10}" y="${y + 13}" font-size="11" fill="#14213d">${escapeHtml(formatTopicMomentum(topic))}</text>
        `;
      }).join("");

      container.innerHTML = `
        <svg viewBox="0 0 ${width} ${height}" width="100%" height="${height}" aria-label="Topic heat chart">
          ${bars}
        </svg>
      `;
    }

    function renderTopics(topics) {
      const container = document.getElementById("topics");
      if (!topics.length) {
        container.innerHTML = `<p class="muted">${escapeHtml(t("topics.no_trends"))}</p>`;
        return;
      }
      const maxCurrent = Math.max(...topics.map((topic) => Number(topic.current_count || 0)), 1);
      container.innerHTML = `
        <div class="topic-bars">
          ${topics.map((topic) => `
            <button type="button" class="topic-row topic-filter-button" data-topic-filter="${escapeHtml(topic.name)}">
              <div class="topic-headline">
                <strong>${escapeHtml(displayTopicLabel(topic.name))}</strong>
                <span>${escapeHtml(formatTopicMomentum(topic))}</span>
              </div>
              <div class="topic-meter">
                <div class="topic-meter-fill" style="width:${Math.max(8, Math.round((Number(topic.current_count || 0) / maxCurrent) * 100))}%"></div>
              </div>
            </button>
          `).join("")}
        </div>
      `;
    }

    function renderQueue(pending, rejected, errors) {
      const container = document.getElementById("queue");
      const sections = [
        ["Pending", pending, (item) => "Waiting to be analyzed."],
        ["Rejected", rejected, (item) => item.rejected_reason || item.relevance_reason || "Rejected."],
        ["Error", errors, (item) => item.analysis_error || "Unknown error."],
      ];
      container.innerHTML = sections.map(([label, items, detailFn]) => `
        <div>
          <p class="panel-subtitle" style="margin: 0 0 10px;">${escapeHtml(label)}</p>
          ${items.length ? items.map((item) => `
            <article class="queue-card">
              <h3 class="queue-title">${escapeHtml(item.title)}</h3>
              <div class="queue-meta">
                <span>${escapeHtml(item.primary_category || t("common.na"))}</span>
                <span>${escapeHtml(formatTime(item.published))}</span>
              </div>
              <p class="muted">${escapeHtml(detailFn(item))}</p>
            </article>
          `).join("") : `<p class="muted">No items.</p>`}
        </div>
      `).join("");
    }

    function renderRunState(runState) {
      const status = document.getElementById("run-status");
      const button = document.getElementById("run-button");
      button.disabled = Boolean(runState.running);
      if (runState.running) {
        status.textContent = t("run.status.running", { time: formatTime(runState.last_started_at) });
        return;
      }
      if (runState.last_error) {
        status.textContent = t("run.status.failed", { time: formatTime(runState.last_finished_at), error: runState.last_error });
        return;
      }
      if (runState.last_finished_at) {
        const result = runState.last_result || {};
        const suffix = result.report_path ? t("run.status.report_suffix", { path: result.report_path }) : "";
        status.textContent = t("run.status.finished", { time: formatTime(runState.last_finished_at), suffix });
        return;
      }
      status.textContent = t("run.status.idle");
    }

    function renderStatusOptions(facets) {
      const select = document.getElementById("paper-status");
      const current = select.value || state.initialSearch?.status || "analyzed";
      const counts = Object.fromEntries((facets?.statuses || []).map((item) => [item.value, item.count]));
      const options = [
        ["analyzed", t("papers.status.analyzed")],
        ["pending", t("papers.status.pending")],
      ];
      select.innerHTML = options.map(([value, label]) => {
        const count = counts[value] || 0;
        return `<option value="${escapeHtml(value)}">${escapeHtml(`${label} (${count})`)}</option>`;
      }).join("");
      select.value = options.some(([value]) => value === current) ? current : "analyzed";
    }

    function renderCategoryOptions(categories, facets) {
      const select = document.getElementById("paper-category");
      const current = select.value || state.initialSearch?.category || "all";
      const countMap = Object.fromEntries((facets?.categories || []).map((item) => [item.value, item.count]));
      const options = ["all", ...categories];
      select.innerHTML = options.map((category) => {
        const label = category === "all"
          ? t("papers.category.all", { count: facets?.total || 0 })
          : `${category} (${countMap[category] || 0})`;
        return `<option value="${escapeHtml(category)}">${escapeHtml(label)}</option>`;
      }).join("");
      select.value = options.includes(current) ? current : "all";
    }

    function renderTrackedTopicOptions(facets) {
      const select = document.getElementById("paper-tracked-topic");
      const current = select.value || state.initialSearch?.tracked_topic || "";
      const options = [{ name: "", count: facets?.total || 0 }, ...(facets?.tracked_topics || [])];
      select.innerHTML = options.map((item) => {
        const label = item.name
          ? `${displayTopicLabel(item.name)} (${item.count})`
          : t("papers.tracked_topic.all", { count: facets?.total || 0 });
        return `<option value="${escapeHtml(item.name || "")}">${escapeHtml(label)}</option>`;
      }).join("");
      select.value = options.some((item) => item.name === current) ? current : "";
    }

    function renderFlagOptions(facets) {
      const select = document.getElementById("paper-flag");
      const current = select.value || state.initialSearch?.flag || "all";
      const counts = facets?.flags || {};
      const options = [
        ["all", t("papers.flag.all", { count: facets?.total || 0 })],
        ["starred", t("papers.flag.starred", { count: counts.starred || 0 })],
        ["ignored", t("papers.flag.ignored", { count: counts.ignored || 0 })],
        ["manual", t("papers.flag.manual", { count: counts.manual || 0 })],
      ];
      select.innerHTML = options.map(([value, label]) => `<option value="${escapeHtml(value)}">${escapeHtml(label)}</option>`).join("");
      select.value = options.some(([value]) => value === current) ? current : "all";
    }

    function renderReportList(reports) {
      const select = document.getElementById("report-select");
      const list = document.getElementById("report-list");
      const currentSelection = state.selectedReport;
      const nextSelection = reports.some((report) => report.name === currentSelection)
        ? currentSelection
        : (reports[0]?.name || "");
      state.selectedReport = nextSelection;

      select.innerHTML = reports.map((report) => `
        <option value="${escapeHtml(report.name)}">${escapeHtml(report.name)}</option>
      `).join("");
      select.value = nextSelection;

      list.innerHTML = reports.length ? reports.map((report) => `
        <div class="report-item ${report.name === nextSelection ? "active" : ""}" data-report-name="${escapeHtml(report.name)}">
          <strong>${escapeHtml(report.name)}</strong>
          <div class="queue-meta">
            <span>${escapeHtml(formatTime(report.modified_at))}</span>
            <span>${escapeHtml(report.size_bytes)} bytes</span>
          </div>
        </div>
      `).join("") : `<p class="muted">${escapeHtml(t("reports.no_reports"))}</p>`;

      list.querySelectorAll(".report-item").forEach((node) => {
        node.addEventListener("click", () => {
          state.selectedReport = node.dataset.reportName || "";
          renderReportList(reports);
          loadReportPreview(true);
        });
      });

      const selected = reports.find((report) => report.name === nextSelection);
      document.getElementById("report-meta").innerHTML = selected
        ? `<span>${escapeHtml(formatTime(selected.modified_at))}</span><span>${escapeHtml(t("reports.bytes", { value: selected.size_bytes }))}</span>`
        : `<span>${escapeHtml(t("reports.no_report_selected"))}</span>`;
    }

    function renderMarkdown(content) {
      const lines = content.split(/\\r?\\n/);
      const blocks = [];
      let index = 0;
      let listItems = [];

      function flushList() {
        if (!listItems.length) return;
        blocks.push(`<ul>${listItems.join("")}</ul>`);
        listItems = [];
      }

      while (index < lines.length) {
        const line = lines[index];
        const trimmed = line.trim();

        if (!trimmed) {
          flushList();
          index += 1;
          continue;
        }

        if (trimmed.startsWith("|")) {
          flushList();
          const tableLines = [];
          while (index < lines.length && lines[index].trim().startsWith("|")) {
            tableLines.push(lines[index].trim());
            index += 1;
          }
          const rows = tableLines.map((row) => row.split("|").slice(1, -1).map((cell) => cell.trim()));
          const header = rows[0] || [];
          const bodyRows = rows.slice(2);
          blocks.push(`
            <table>
              <thead><tr>${header.map((cell) => `<th>${formatInline(cell)}</th>`).join("")}</tr></thead>
              <tbody>${bodyRows.map((row) => `<tr>${row.map((cell) => `<td>${formatInline(cell)}</td>`).join("")}</tr>`).join("")}</tbody>
            </table>
          `);
          continue;
        }

        if (trimmed.startsWith("### ")) {
          flushList();
          blocks.push(`<h3>${formatInline(trimmed.slice(4))}</h3>`);
          index += 1;
          continue;
        }
        if (trimmed.startsWith("## ")) {
          flushList();
          blocks.push(`<h2>${formatInline(trimmed.slice(3))}</h2>`);
          index += 1;
          continue;
        }
        if (trimmed.startsWith("# ")) {
          flushList();
          blocks.push(`<h1>${formatInline(trimmed.slice(2))}</h1>`);
          index += 1;
          continue;
        }
        if (/^\\*\\*(.+)\\*\\*$/.test(trimmed)) {
          flushList();
          blocks.push(`<h4>${formatInline(trimmed.slice(2, -2))}</h4>`);
          index += 1;
          continue;
        }
        if (trimmed.startsWith("- ")) {
          listItems.push(`<li>${formatInline(trimmed.slice(2))}</li>`);
          index += 1;
          continue;
        }

        flushList();
        blocks.push(`<p>${formatInline(trimmed)}</p>`);
        index += 1;
      }

      flushList();
      return blocks.join("");
    }

    async function loadReportPreview(force = false) {
      const reportName = state.selectedReport;
      const preview = document.getElementById("report-preview");
      if (!reportName) {
        preview.innerHTML = `<p class="muted">${escapeHtml(t("reports.no_report_available"))}</p>`;
        return;
      }
      const cacheKey = `${reportName}`;
      if (!force && cacheKey === state.reportCacheKey) {
        return;
      }
      const response = await fetch(`/reports/${encodeURIComponent(reportName)}`);
      const content = await response.text();
      state.reportCacheKey = cacheKey;
      const rendered = PAGE_VIEW === "overview" ? summarizeMarkdown(content, 24) : content;
      preview.innerHTML = renderMarkdown(rendered);
    }

    function statusDetail(paper) {
      if (paper.analysis_status === "pending") return paper.summary || t("paper.status.pending");
      if (paper.analysis_status === "rejected") return paper.rejected_reason || paper.relevance_reason || t("paper.status.rejected");
      if (paper.analysis_status === "error") return paper.analysis_error || t("paper.status.error");
      return localizedSummary(paper);
    }

    function renderPaperSearchMeta(payload) {
      const meta = document.getElementById("paper-search-meta");
      const start = payload.total ? payload.offset + 1 : 0;
      const end = payload.offset + payload.papers.length;
      const extras = [];
      if (payload.tracked_topic) {
        extras.push(t("papers.meta.extra", {
          label: document.getElementById("paper-tracked-topic").labels?.[0]?.textContent || t("papers.tracked_topic.label"),
          value: document.getElementById("paper-tracked-topic").selectedOptions[0]?.textContent || displayTopicLabel(payload.tracked_topic),
        }));
      }
      if (payload.flag && payload.flag !== "all") {
        extras.push(t("papers.meta.extra", {
          label: document.getElementById("paper-flag").labels?.[0]?.textContent || t("papers.flag.label"),
          value: document.getElementById("paper-flag").selectedOptions[0]?.textContent || payload.flag,
        }));
      }
      if (payload.days) {
        extras.push(t("papers.meta.extra", {
          label: document.getElementById("paper-days").labels?.[0]?.textContent || t("papers.days.label"),
          value: document.getElementById("paper-days").selectedOptions[0]?.textContent || payload.days,
        }));
      }
      meta.textContent = t("papers.meta.summary", {
        start,
        end,
        total: payload.total,
        page: payload.page,
        pages: payload.total_pages,
        sort: document.getElementById("paper-sort").selectedOptions[0]?.textContent || payload.sort,
        status: document.getElementById("paper-status").selectedOptions[0]?.textContent || payload.status,
        category: document.getElementById("paper-category").selectedOptions[0]?.textContent || payload.category,
        query: payload.query || "",
        extras: extras.join(""),
      });
    }

    function buildPageList(page, totalPages) {
      const picked = new Set([1, totalPages, page, page - 1, page + 1, page - 2, page + 2]);
      return [...picked]
        .filter((value) => value >= 1 && value <= totalPages)
        .sort((a, b) => a - b);
    }

    function renderPagination(payload) {
      const container = document.getElementById("paper-pagination");
      if (!payload.total || payload.total_pages <= 1) {
        container.innerHTML = "";
        return;
      }

      const pages = buildPageList(payload.page, payload.total_pages);
      const buttons = [];
      buttons.push(`
        <button type="button" class="page-ghost" data-page-nav="${payload.page - 1}" ${payload.page <= 1 ? "disabled" : ""}>${escapeHtml(t("pagination.prev"))}</button>
      `);

      for (let index = 0; index < pages.length; index += 1) {
        const page = pages[index];
        const previous = pages[index - 1];
        if (previous && page - previous > 1) {
          buttons.push(`<span class="muted">...</span>`);
        }
        buttons.push(`
          <button type="button" class="${page === payload.page ? "page-current" : "page-ghost"}" data-page-nav="${page}">
            ${page}
          </button>
        `);
      }

      buttons.push(`
        <button type="button" class="page-ghost" data-page-nav="${payload.page + 1}" ${payload.page >= payload.total_pages ? "disabled" : ""}>${escapeHtml(t("pagination.next"))}</button>
      `);
      container.innerHTML = buttons.join("");
    }

    function renderPapers(papers, options = {}) {
      const compact = Boolean(options.compact);
      const rawTopicLimit = Number(options.topicLimit || 0);
      const topicLimit = rawTopicLimit > 0 ? rawTopicLimit : null;
      const highlightQuery = options.query || options.trackedTopic || "";
      const container = document.getElementById("papers");
      if (!papers.length) {
        container.innerHTML = `<p class="muted">${escapeHtml(t("papers.no_results"))}</p>`;
        return;
      }
      container.innerHTML = papers.map((paper) => `
        <article id="${paperAnchorId(paper.entry_id)}" class="paper-card ${compact ? "compact" : ""}">
          ${paper.is_starred ? `<span class="flag-pill starred">${escapeHtml(t("paper.flag.starred"))}</span>` : ""}
          ${paper.is_ignored ? `<span class="flag-pill ignored">${escapeHtml(t("paper.flag.ignored"))}</span>` : ""}
          ${(paper.manual_topics?.length || paper.manual_note) ? `<span class="flag-pill manual">${escapeHtml(t("paper.flag.manual"))}</span>` : ""}
          <h3 class="paper-title"><a href="${paperDetailUrl(paper.entry_id)}">${highlightText(paper.title, highlightQuery)}</a></h3>
          <div class="paper-meta">
            <span>${escapeHtml(paper.primary_category || t("common.na"))}</span>
            <span>${escapeHtml(formatTime(paper.published))}</span>
            <a href="${paperDetailUrl(paper.entry_id)}">${escapeHtml(t("paper.view_details"))}</a>
            <a href="${escapeHtml(paper.entry_id)}" target="_blank" rel="noreferrer">${escapeHtml(t("paper.open_arxiv"))}</a>
          </div>
          ${paper.tracked_topics?.length ? `
            <div class="topic-label">${escapeHtml(t("paper.tracked_topics"))}</div>
            <div>${(topicLimit ? paper.tracked_topics.slice(0, topicLimit) : paper.tracked_topics).map((topic) => `<button type="button" class="topic-pill tracked topic-filter" data-topic-filter="${escapeHtml(topic)}">${escapeHtml(displayTopicLabel(topic))}</button>`).join("")}</div>
          ` : ""}
          ${(paper.topics || []).length ? `
            <div class="topic-label">${escapeHtml(t("paper.model_topics"))}</div>
            <div>${(topicLimit ? paper.topics.slice(0, topicLimit) : paper.topics).map((topic) => `<button type="button" class="topic-pill topic-filter" data-topic-filter="${escapeHtml(topic)}">${escapeHtml(topic)}</button>`).join("")}</div>
          ` : ""}
          <p class="muted" style="margin-top: 12px;">${highlightText(truncateText(statusDetail(paper), compact ? 190 : 360), highlightQuery)}</p>
          ${paper.match_reasons?.length ? `<div class="match-reasons">${escapeHtml(t("paper.matched_in", { value: paper.match_reasons.join(", ") }))}</div>` : ""}
          ${paper.manual_topics?.length ? `<div class="match-reasons">${escapeHtml(t("paper.manual_topics", { value: paper.manual_topics.map((topic) => displayTopicLabel(topic)).join(", ") }))}</div>` : ""}
          <div class="paper-actions">
            <a class="text-link" href="${paperDetailUrl(paper.entry_id)}">${compact ? escapeHtml(t("paper.open_detail")) : escapeHtml(t("paper.full_detail"))}</a>
          </div>
          ${!compact && paper.analysis_status === "analyzed" ? `
            <details>
              <summary>${escapeHtml(t("paper.structured_view"))}</summary>
              <div class="dual">
                <section class="dual-section">
                  <strong>${escapeHtml(t("paper.summary"))}</strong>
                  <div>${escapeHtml(localizedSummary(paper) || t("common.na"))}</div>
                </section>
                <section class="dual-section">
                  <strong>${escapeHtml(t("paper.problem"))}</strong>
                  <div>${escapeHtml(localizedPaperText(paper, "problem") || t("common.na"))}</div>
                </section>
                <section class="dual-section">
                  <strong>${escapeHtml(t("paper.method"))}</strong>
                  <div>${escapeHtml(localizedPaperText(paper, "method") || t("common.na"))}</div>
                </section>
                <section class="dual-section">
                  <strong>${escapeHtml(t("paper.findings"))}</strong>
                  <div>${escapeHtml(localizedPaperText(paper, "findings") || t("common.na"))}</div>
                </section>
                <section class="dual-section">
                  <strong>${escapeHtml(t("paper.limitations"))}</strong>
                  <div>${escapeHtml(localizedPaperText(paper, "limitations") || t("common.na"))}</div>
                </section>
              </div>
            </details>
          ` : ""}
        </article>
      `).join("");
    }

    async function loadPaperResults(force = false, requestedPage = null) {
      const query = document.getElementById("paper-query").value.trim();
      const status = document.getElementById("paper-status").value;
      const category = document.getElementById("paper-category").value;
      const trackedTopic = document.getElementById("paper-tracked-topic").value;
      const flag = document.getElementById("paper-flag").value;
      const days = document.getElementById("paper-days").value;
      const limit = Number(document.getElementById("paper-limit").value || 12);
      const sort = document.getElementById("paper-sort").value;
      const page = Math.max(1, Number(requestedPage || state.currentPaperPage || 1));
      const key = JSON.stringify({ query, status, category, trackedTopic, flag, days, limit, page, sort });
      if (!force && key === state.paperQueryKey) {
        return;
      }
      state.paperQueryKey = key;
      state.currentPaperPage = page;
      const params = new URLSearchParams({
        query,
        status,
        category,
        tracked_topic: trackedTopic,
        flag,
        days,
        limit: String(limit),
        page: String(page),
        sort,
      });
      const response = await fetch(`/api/papers?${params.toString()}`);
      const payload = await response.json();
      state.lastPaperPayload = payload;
      state.currentPaperPage = payload.page;
      updateSearchUrl(query, status, category, trackedTopic, flag, days, payload.page, sort, limit);
      renderPaperSearchMeta(payload);
      renderPapers(payload.papers, { query, trackedTopic });
      renderPagination(payload);
    }

    async function renderDashboardFromPayload(payload, forcePaperReload = false) {
      applyStaticI18n();
      applyActiveNav();
      renderHero(payload.config, payload.reports);
      renderRunState(payload.run_state);
      renderRunHistory(payload.run_history || []);
      syncRunHistoryFold(payload.run_state, payload.run_history || []);
      configureReportsSection();
      if (PAGE_VIEW !== "reports") {
        renderStatusOptions(payload.filter_facets);
        renderCategoryOptions(payload.available_categories, payload.filter_facets);
        renderTrackedTopicOptions(payload.filter_facets);
        renderFlagOptions(payload.filter_facets);
      }
      if (PAGE_VIEW === "overview") {
        renderStats(payload.status_counts.recent, payload.filter_facets, payload.reports, payload.topic_trends);
        renderCategoryChart(payload.category_mix || []);
        renderTopicHeatChart(payload.topic_trends);
        renderTopics(payload.topic_trends);
        configurePapersSectionForOverview(payload.analyzed_papers || []);
        return;
      }

      if (PAGE_VIEW === "reports") {
        renderReportList(payload.reports);
        state.reportCacheKey = "";
        await loadReportPreview(true);
        return;
      }

      renderQuickTopicFilters(payload.topic_trends);
      if (forcePaperReload || !state.lastPaperPayload) {
        await loadPaperResults(true, state.currentPaperPage);
        return;
      }
      renderPaperSearchMeta(state.lastPaperPayload);
      renderPapers(state.lastPaperPayload.papers, {
        query: state.lastPaperPayload.query,
        trackedTopic: state.lastPaperPayload.tracked_topic,
      });
      renderPagination(state.lastPaperPayload);
    }

    async function triggerRun() {
      const body = {
        lookback_days: Number(document.getElementById("lookback-days").value),
        max_results: Number(document.getElementById("max-results").value),
        query_keywords: document.getElementById("query-keywords").value,
        analysis_limit: Number(document.getElementById("analysis-limit").value),
        report_paper_limit: Number(document.getElementById("report-paper-limit").value),
      };
      const response = await fetch("/api/run", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      if (!response.ok) {
        const payload = await response.json();
        alert(payload.error || t("common.failed_to_start_run"));
        return;
      }
      state.paperQueryKey = "";
      await loadState(true);
    }

    async function loadState(forcePaperReload = false) {
      try {
        const response = await fetch("/api/state");
        const payload = await response.json();
        state.lastPayload = payload;

        if (!state.initialized) {
          state.initialSearch = parseSearchParams();
          state.currentPaperPage = state.initialSearch.page;
          document.getElementById("lookback-days").value = payload.config.lookback_days;
          document.getElementById("max-results").value = payload.config.max_results;
          document.getElementById("query-keywords").value = (payload.config.query_keywords || []).join(", ");
          document.getElementById("analysis-limit").value = payload.config.analysis_limit_per_run;
          document.getElementById("report-paper-limit").value = payload.config.report_paper_limit;
          document.getElementById("paper-limit").value = state.initialSearch.limit || payload.config.report_paper_limit;
          setSearchFilters(state.initialSearch);
          state.initialized = true;
        }
        await renderDashboardFromPayload(payload, forcePaperReload);
        scheduleStateRefresh(payload);
      } catch (error) {
        scheduleStateRefresh(null, 10000);
      }
    }

    function scheduleStateRefresh(payload = null, fallbackDelay = null) {
      if (state.refreshTimer) {
        clearTimeout(state.refreshTimer);
      }
      const hasRunningHistory = Boolean(payload?.run_history?.some((item) => item.status === "running"));
      const delay = fallbackDelay ?? ((payload?.run_state?.running || hasRunningHistory) ? 2500 : 8000);
      state.refreshTimer = window.setTimeout(() => {
        loadState(false);
      }, delay);
    }

    if (document.getElementById("run-button")) {
      document.getElementById("run-button").addEventListener("click", triggerRun);
    }
    if (document.getElementById("paper-search-button")) {
      document.getElementById("paper-search-button").addEventListener("click", () => {
        state.currentPaperPage = 1;
        loadPaperResults(true, 1);
      });
    }
    if (document.getElementById("refresh-preview-button")) {
      document.getElementById("refresh-preview-button").addEventListener("click", () => {
        state.reportCacheKey = "";
        loadReportPreview(true);
      });
    }
    if (document.getElementById("report-select")) {
      document.getElementById("report-select").addEventListener("change", (event) => {
        state.selectedReport = event.target.value;
        state.reportCacheKey = "";
        loadReportPreview(true);
      });
    }
    if (document.getElementById("paper-query")) {
      document.getElementById("paper-query").addEventListener("keydown", (event) => {
        if (event.key === "Enter") {
          state.currentPaperPage = 1;
          loadPaperResults(true, 1);
        }
      });
    }
    if (document.getElementById("paper-status")) {
      document.getElementById("paper-status").addEventListener("change", () => {
        state.currentPaperPage = 1;
        loadPaperResults(true, 1);
      });
    }
    if (document.getElementById("paper-category")) {
      document.getElementById("paper-category").addEventListener("change", () => {
        state.currentPaperPage = 1;
        loadPaperResults(true, 1);
      });
    }
    if (document.getElementById("paper-tracked-topic")) {
      document.getElementById("paper-tracked-topic").addEventListener("change", () => {
        state.currentPaperPage = 1;
        loadPaperResults(true, 1);
      });
    }
    if (document.getElementById("paper-flag")) {
      document.getElementById("paper-flag").addEventListener("change", () => {
        state.currentPaperPage = 1;
        loadPaperResults(true, 1);
      });
    }
    if (document.getElementById("paper-days")) {
      document.getElementById("paper-days").addEventListener("change", () => {
        state.currentPaperPage = 1;
        loadPaperResults(true, 1);
      });
    }
    if (document.getElementById("paper-limit")) {
      document.getElementById("paper-limit").addEventListener("change", () => {
        state.currentPaperPage = 1;
        loadPaperResults(true, 1);
      });
    }
    if (document.getElementById("paper-sort")) {
      document.getElementById("paper-sort").addEventListener("change", () => {
        state.currentPaperPage = 1;
        loadPaperResults(true, 1);
      });
    }
    if (document.getElementById("paper-clear-button")) {
      document.getElementById("paper-clear-button").addEventListener("click", () => {
        setSearchFilters({
          query: "",
          status: "analyzed",
          category: "all",
          tracked_topic: "",
          flag: "all",
          days: "",
          limit: 12,
          sort: "published_desc",
        });
        state.paperQueryKey = "";
        state.currentPaperPage = 1;
        loadPaperResults(true, 1);
      });
    }
    if (document.getElementById("paper-copy-link-button")) {
      document.getElementById("paper-copy-link-button").addEventListener("click", async () => {
        const link = window.location.href;
        try {
          await navigator.clipboard.writeText(link);
          document.getElementById("paper-search-meta").textContent = t("papers.copy_success", { link });
        } catch (error) {
          document.getElementById("paper-search-meta").textContent = t("papers.copy_fail", { link });
        }
      });
    }
    if (document.getElementById("paper-pagination")) {
      document.getElementById("paper-pagination").addEventListener("click", (event) => {
        const trigger = event.target.closest("[data-page-nav]");
        if (!trigger || trigger.disabled) return;
        const targetPage = Number(trigger.dataset.pageNav || 1);
        loadPaperResults(true, targetPage);
      });
    }
    document.addEventListener("click", (event) => {
      const trigger = event.target.closest("[data-topic-filter]");
      if (!trigger) return;
      event.preventDefault();
      applyTopicFilter(trigger.dataset.topicFilter || "");
    });

    applyStaticI18n();
    loadState(true);
  </script>
</body>
</html>
"""


PAPER_HTML = """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Paper Detail - arXiv LLM Watch</title>
  <style>
    :root {
      --ink: #14213d;
      --paper: #f6f1e8;
      --sand: #e8dcc8;
      --accent: #b23a48;
      --teal: #0d5c63;
      --muted: #5c6b73;
      --gold: #c58b2a;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: "Avenir Next", "PingFang SC", "Noto Sans SC", sans-serif;
      background:
        radial-gradient(circle at top left, rgba(197, 139, 42, 0.14), transparent 34%),
        linear-gradient(180deg, #fbf8f2 0%, #efe4d1 100%);
      color: var(--ink);
    }
    .shell {
      width: min(1120px, calc(100% - 28px));
      margin: 24px auto 40px;
    }
    .hero,
    .panel {
      border-radius: 26px;
      border: 1px solid rgba(20, 33, 61, 0.08);
      background: rgba(255,255,255,0.74);
      backdrop-filter: blur(14px);
      box-shadow: 0 24px 60px rgba(20, 33, 61, 0.08);
    }
    .hero {
      padding: 24px;
      margin-bottom: 18px;
    }
    .eyebrow,
    .section-label {
      font-size: 12px;
      letter-spacing: 0.16em;
      text-transform: uppercase;
      color: var(--accent);
    }
    .title {
      margin: 12px 0 10px;
      font-size: clamp(28px, 4vw, 46px);
      line-height: 1.02;
    }
    .subtitle,
    .muted {
      color: var(--muted);
      line-height: 1.7;
      font-size: 14px;
    }
    .meta,
    .actions,
    .topic-group,
    .topnav {
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      margin-top: 14px;
    }
    .topnav {
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 12px;
      margin-top: 18px;
    }
    .chip,
    .topic-pill {
      display: inline-block;
      padding: 7px 12px;
      border-radius: 999px;
      font-size: 12px;
      text-decoration: none;
    }
    .chip {
      background: rgba(20, 33, 61, 0.06);
      color: var(--ink);
    }
    .topic-pill {
      background: rgba(197, 139, 42, 0.12);
      color: #7b5715;
    }
    .topic-pill.tracked {
      background: rgba(13, 92, 99, 0.12);
      color: var(--teal);
    }
    .topnav a {
      display: block;
      padding: 14px 16px;
      border-radius: 18px;
      background: linear-gradient(180deg, rgba(20, 33, 61, 0.07), rgba(20, 33, 61, 0.03));
      border: 1px solid rgba(20, 33, 61, 0.08);
      font-size: 13px;
      transition: transform 150ms ease, box-shadow 150ms ease, border-color 150ms ease;
    }
    .topnav a:hover {
      text-decoration: none;
      transform: translateY(-1px);
      box-shadow: 0 12px 24px rgba(20, 33, 61, 0.08);
    }
    .topnav a.active {
      border-color: rgba(13, 92, 99, 0.32);
      background: linear-gradient(180deg, rgba(13, 92, 99, 0.10), rgba(13, 92, 99, 0.06));
      box-shadow: 0 16px 28px rgba(13, 92, 99, 0.12);
    }
    .nav-label {
      display: block;
      font-size: 14px;
      font-weight: 700;
      letter-spacing: 0.08em;
      text-transform: uppercase;
    }
    .nav-copy {
      display: block;
      margin-top: 6px;
      font-size: 12px;
      line-height: 1.5;
      color: var(--muted);
    }
    .panel {
      margin-top: 18px;
      padding: 22px;
    }
    .section-title {
      margin: 10px 0 0;
      font-size: 24px;
      line-height: 1.15;
    }
    .grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 14px;
    }
    .card {
      border-radius: 18px;
      border: 1px solid rgba(20, 33, 61, 0.08);
      background: rgba(246, 241, 232, 0.7);
      padding: 16px;
    }
    .field-label {
      display: block;
      margin-bottom: 8px;
      font-size: 12px;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      color: var(--accent);
    }
    input,
    textarea {
      width: 100%;
      border: 1px solid rgba(20, 33, 61, 0.12);
      border-radius: 14px;
      background: rgba(255,255,255,0.9);
      padding: 12px 14px;
      font: inherit;
      color: var(--ink);
    }
    textarea {
      min-height: 120px;
      resize: vertical;
    }
    button {
      appearance: none;
      border: 0;
      border-radius: 14px;
      background: linear-gradient(135deg, var(--teal), #2b7a78);
      color: #fff;
      padding: 12px 16px;
      font: inherit;
      cursor: pointer;
      box-shadow: 0 12px 24px rgba(13, 92, 99, 0.14);
    }
    button.secondary {
      background: linear-gradient(135deg, var(--accent), #d96b3d);
      box-shadow: 0 12px 24px rgba(178, 58, 72, 0.14);
    }
    .button-row {
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      margin-top: 14px;
    }
    .card strong {
      display: block;
      margin-bottom: 10px;
      font-size: 13px;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      color: var(--accent);
    }
    .card strong.inline {
      display: inline;
      margin: 0;
    }
    .related-list {
      display: grid;
      gap: 12px;
      margin-top: 14px;
    }
    .related-item {
      border-radius: 18px;
      border: 1px solid rgba(20, 33, 61, 0.08);
      background: rgba(246, 241, 232, 0.7);
      padding: 16px;
    }
    .related-title {
      margin: 0;
      font-size: 18px;
      line-height: 1.2;
    }
    a {
      color: var(--teal);
      text-decoration: none;
    }
    a:hover { text-decoration: underline; }
    .empty {
      padding: 40px 24px;
      text-align: center;
      color: var(--muted);
    }
    .mobile-nav {
      display: none;
    }
    @media (max-width: 840px) {
      body {
        padding-bottom: 84px;
      }
      .grid {
        grid-template-columns: 1fr;
      }
      .topnav {
        grid-template-columns: 1fr;
      }
      .mobile-nav {
        position: fixed;
        left: 12px;
        right: 12px;
        bottom: 12px;
        z-index: 20;
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 8px;
        padding: 10px;
        border-radius: 18px;
        background: rgba(20, 33, 61, 0.92);
        box-shadow: 0 20px 40px rgba(20, 33, 61, 0.18);
      }
      .mobile-nav a {
        display: block;
        text-align: center;
        color: rgba(255,255,255,0.84);
        padding: 10px 8px;
        border-radius: 12px;
        font-size: 12px;
        letter-spacing: 0.08em;
        text-transform: uppercase;
      }
      .mobile-nav a.active {
        background: rgba(255,255,255,0.14);
        color: #fff;
      }
    }
  </style>
</head>
<body>
  <div class="shell">
    <div id="paper-root" class="empty">Loading paper detail...</div>
  </div>
  <nav class="mobile-nav" id="paper-mobile-nav"></nav>
  <script>
    const state = {
      lang: "zh",
      paper: null,
    };

    const I18N = {
      zh: {
        "page.title": "论文详情 - LLM Paper Radar",
        "loading": "正在加载论文详情...",
        "missing_entry": "缺少 entry_id。",
        "paper_not_found": "没有找到这篇论文。",
        "nav.overview.label": "总览",
        "nav.overview.copy": "回到今日总览。",
        "nav.papers.label": "论文",
        "nav.papers.copy": "继续筛选或浏览更多论文。",
        "nav.reports.label": "报告",
        "nav.reports.copy": "查看完整日报和历史版本。",
        "hero.eyebrow": "单篇论文视图",
        "paper.open_arxiv": "打开 arXiv",
        "paper.open_pdf": "打开 PDF",
        "paper.back": "返回上一页",
        "paper.copy_link": "复制页面链接",
        "paper.link_copied": "页面链接已复制",
        "paper.copy_failed": "复制失败",
        "paper.published": "发布时间：{value}",
        "paper.updated": "更新时间：{value}",
        "paper.score": "相关度：{value}",
        "paper.starred": "已收藏",
        "paper.ignored": "已忽略",
        "meta.section": "元数据",
        "meta.title": "上下文",
        "meta.authors": "作者",
        "meta.categories": "分类",
        "meta.tracked_topics": "受控主题",
        "meta.model_topics": "模型主题",
        "meta.relevance_reason": "相关性判断",
        "meta.raw_abstract": "原始摘要",
        "workflow.section": "工作流",
        "workflow.title": "人工校正",
        "workflow.quick_actions": "快捷操作",
        "workflow.quick_actions.copy": "用人工校正把高价值论文留在前面，把低价值论文排除在后续阅读之外。",
        "workflow.star": "收藏",
        "workflow.unstar": "取消收藏",
        "workflow.ignore": "忽略",
        "workflow.unignore": "取消忽略",
        "workflow.reanalyze": "重新分析",
        "workflow.last_review": "最近一次人工更新：{value}",
        "workflow.manual_topics": "人工主题",
        "workflow.manual_topics_label": "逗号分隔的主题",
        "workflow.manual_topics_placeholder": "如：推理、对齐、评测",
        "workflow.analyst_note": "分析备注",
        "workflow.analyst_note_placeholder": "记录这篇论文为什么重要、要回看什么，或为什么忽略。",
        "workflow.save_review": "保存校正",
        "analysis.section": "结构化摘要",
        "analysis.title": "单语分析视图",
        "analysis.summary": "摘要",
        "analysis.background": "背景",
        "analysis.problem": "问题",
        "analysis.method": "方法",
        "analysis.findings": "结果",
        "analysis.limitations": "局限",
        "discovery.section": "发现",
        "discovery.title": "相关论文",
        "discovery.score": "相似度：{value}",
        "status.section": "状态",
        "status.title": "尚未完成结构化分析",
        "status.detail": "详细信息",
        "common.na": "暂无",
      },
      en: {
        "page.title": "Paper Detail - LLM Paper Radar",
        "loading": "Loading paper detail...",
        "missing_entry": "Missing entry_id.",
        "paper_not_found": "Paper not found.",
        "nav.overview.label": "Overview",
        "nav.overview.copy": "Back to today's overview.",
        "nav.papers.label": "Papers",
        "nav.papers.copy": "Continue filtering or browse more papers.",
        "nav.reports.label": "Reports",
        "nav.reports.copy": "Open the full report archive.",
        "hero.eyebrow": "Single Paper View",
        "paper.open_arxiv": "Open arXiv",
        "paper.open_pdf": "Open PDF",
        "paper.back": "Back to previous view",
        "paper.copy_link": "Copy page link",
        "paper.link_copied": "Page link copied",
        "paper.copy_failed": "Copy failed",
        "paper.published": "Published: {value}",
        "paper.updated": "Updated: {value}",
        "paper.score": "Score: {value}",
        "paper.starred": "Starred",
        "paper.ignored": "Ignored",
        "meta.section": "Metadata",
        "meta.title": "Context",
        "meta.authors": "Authors",
        "meta.categories": "Categories",
        "meta.tracked_topics": "Tracked Topics",
        "meta.model_topics": "Model Topics",
        "meta.relevance_reason": "Relevance Reason",
        "meta.raw_abstract": "Raw Abstract",
        "workflow.section": "Workflow",
        "workflow.title": "Review Controls",
        "workflow.quick_actions": "Quick Actions",
        "workflow.quick_actions.copy": "Use manual review to keep high-value papers surfaced and low-value ones out of future batches.",
        "workflow.star": "Star",
        "workflow.unstar": "Unstar",
        "workflow.ignore": "Ignore",
        "workflow.unignore": "Unignore",
        "workflow.reanalyze": "Re-analyze",
        "workflow.last_review": "Last review update: {value}",
        "workflow.manual_topics": "Manual Topics",
        "workflow.manual_topics_label": "Comma-separated topics",
        "workflow.manual_topics_placeholder": "reasoning, alignment, evaluation",
        "workflow.analyst_note": "Analyst Note",
        "workflow.analyst_note_placeholder": "Why this paper matters, what to revisit, or why to ignore it.",
        "workflow.save_review": "Save Review",
        "analysis.section": "Structured Summary",
        "analysis.title": "Single-Language Analysis",
        "analysis.summary": "Summary",
        "analysis.background": "Background",
        "analysis.problem": "Problem",
        "analysis.method": "Method",
        "analysis.findings": "Findings",
        "analysis.limitations": "Limitations",
        "discovery.section": "Discovery",
        "discovery.title": "Related Papers",
        "discovery.score": "Score: {value}",
        "status.section": "Status",
        "status.title": "Not Fully Analyzed",
        "status.detail": "Detail",
        "common.na": "n/a",
      },
    };

    const TRACKED_TOPIC_LABELS = {
      zh: {
        "reasoning": "推理",
        "agents & tool use": "智能体与工具使用",
        "rag & retrieval": "RAG 与检索",
        "alignment & safety": "对齐与安全",
        "evaluation & llm as a judge": "评测与 LLM 裁判",
        "hallucination & factuality": "幻觉与事实性",
        "post training & preference optimization": "后训练与偏好优化",
        "training efficiency": "训练效率",
        "inference efficiency": "推理效率",
        "long context & memory": "长上下文与记忆",
        "multimodal llm": "多模态 LLM",
        "mechanistic interpretability": "机制可解释性",
        "synthetic data & distillation": "合成数据与蒸馏",
        "coding & program synthesis": "编码与程序合成",
        "benchmarks & datasets": "基准与数据集",
      },
      en: {},
    };

    function currentLang() {
      return state.lang === "en" ? "en" : "zh";
    }

    function t(key, params = {}) {
      const lang = currentLang();
      const template = I18N[lang]?.[key] ?? I18N.en[key] ?? key;
      return String(template).replace(/\{(\w+)\}/g, (_, name) => String(params[name] ?? ""));
    }

    function displayTopicLabel(topic) {
      const value = String(topic || "");
      const normalized = value.trim().toLowerCase();
      if (currentLang() === "zh") {
        return TRACKED_TOPIC_LABELS.zh[normalized] || value;
      }
      return value;
    }

    function localizedPaperText(paper, field) {
      const zh = paper?.[`${field}_zh`];
      const en = paper?.[`${field}_en`];
      if (currentLang() === "zh") {
        return zh || en || "";
      }
      return en || zh || "";
    }

    function localizedSummary(paper) {
      return localizedPaperText(paper, "summary") || paper?.summary || t("common.na");
    }

    function applyLanguageState() {
      document.documentElement.lang = "zh-CN";
      document.title = t("page.title");
    }

    function escapeHtml(value) {
      return String(value ?? "")
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#39;");
    }

    function formatTime(value) {
      if (!value) return t("common.na");
      return new Date(value).toLocaleString(currentLang() === "zh" ? "zh-CN" : "en-US");
    }

    function currentParams() {
      return new URLSearchParams(window.location.search);
    }

    function resolveReturnTo() {
      const candidate = currentParams().get("return_to") || "/papers";
      return candidate.startsWith("/") ? candidate : "/papers";
    }

    function originView() {
      const target = resolveReturnTo();
      if (target.startsWith("/reports")) return "reports";
      if (target.startsWith("/papers")) return "papers";
      return "overview";
    }

    function homeFilterUrl(topic) {
      const params = new URLSearchParams({ tracked_topic: topic });
      return `/papers?${params.toString()}`;
    }

    function renderTopicLinks(topics, tracked = false) {
      if (!topics?.length) return `<span class="muted">${escapeHtml(t("common.na"))}</span>`;
      return topics.map((topic) => `
        <a class="topic-pill ${tracked ? "tracked" : ""}" href="${homeFilterUrl(topic)}">${escapeHtml(tracked ? displayTopicLabel(topic) : topic)}</a>
      `).join("");
    }

    async function postPaperAction(body) {
      const response = await fetch("/api/paper/action", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      const payload = await response.json();
      if (!response.ok) {
        throw new Error(payload.error || "Action failed.");
      }
      return payload.paper;
    }

    function renderAnalyzedSections(paper) {
      return `
        <section class="panel">
          <div class="section-label">${escapeHtml(t("analysis.section"))}</div>
          <h2 class="section-title">${escapeHtml(t("analysis.title"))}</h2>
          <div class="grid" style="margin-top: 14px;">
            <div class="card">
              <strong>${escapeHtml(t("analysis.summary"))}</strong>
              <div>${escapeHtml(localizedSummary(paper))}</div>
            </div>
            <div class="card">
              <strong>${escapeHtml(t("analysis.background"))}</strong>
              <div>${escapeHtml(localizedPaperText(paper, "background") || t("common.na"))}</div>
            </div>
            <div class="card">
              <strong>${escapeHtml(t("analysis.problem"))}</strong>
              <div>${escapeHtml(localizedPaperText(paper, "problem") || t("common.na"))}</div>
            </div>
            <div class="card">
              <strong>${escapeHtml(t("analysis.method"))}</strong>
              <div>${escapeHtml(localizedPaperText(paper, "method") || t("common.na"))}</div>
            </div>
            <div class="card">
              <strong>${escapeHtml(t("analysis.findings"))}</strong>
              <div>${escapeHtml(localizedPaperText(paper, "findings") || t("common.na"))}</div>
            </div>
            <div class="card">
              <strong>${escapeHtml(t("analysis.limitations"))}</strong>
              <div>${escapeHtml(localizedPaperText(paper, "limitations") || t("common.na"))}</div>
            </div>
          </div>
        </section>
      `;
    }

    function renderRelatedPapers(paper) {
      if (!paper.related_papers?.length) return "";
      return `
        <section class="panel">
          <div class="section-label">${escapeHtml(t("discovery.section"))}</div>
          <h2 class="section-title">${escapeHtml(t("discovery.title"))}</h2>
          <div class="related-list">
            ${paper.related_papers.map((item) => `
              <article class="related-item">
                <h3 class="related-title"><a href="/paper?entry_id=${encodeURIComponent(item.entry_id)}">${escapeHtml(item.title)}</a></h3>
                <div class="meta" style="margin-top: 10px;">
                  <span class="chip">${escapeHtml(item.primary_category || t("common.na"))}</span>
                  <span class="chip">${escapeHtml(t("discovery.score", { value: item.similarity_score || 0 }))}</span>
                  <span class="chip">${escapeHtml(formatTime(item.published))}</span>
                </div>
                <div class="topic-group">${renderTopicLinks(item.shared_tracked_topics || [], true)}</div>
                <div class="muted" style="margin-top: 10px;">${escapeHtml(localizedSummary(item))}</div>
              </article>
            `).join("")}
          </div>
        </section>
      `;
    }

    function renderPaper(paper) {
      state.paper = paper;
      applyLanguageState();
      const root = document.getElementById("paper-root");
      const returnTo = resolveReturnTo();
      const activeView = originView();
      const mobileNav = document.getElementById("paper-mobile-nav");
      root.className = "";
      root.innerHTML = `
        <section class="hero">
          <div class="eyebrow">${escapeHtml(t("hero.eyebrow"))}</div>
          <h1 class="title">${escapeHtml(paper.title)}</h1>
          <div class="subtitle">${escapeHtml(localizedSummary(paper))}</div>
          <nav class="topnav">
            <a href="/" class="${activeView === "overview" ? "active" : ""}">
              <span class="nav-label">${escapeHtml(t("nav.overview.label"))}</span>
              <span class="nav-copy">${escapeHtml(t("nav.overview.copy"))}</span>
            </a>
            <a href="/papers" class="${activeView === "papers" ? "active" : ""}">
              <span class="nav-label">${escapeHtml(t("nav.papers.label"))}</span>
              <span class="nav-copy">${escapeHtml(t("nav.papers.copy"))}</span>
            </a>
            <a href="/reports" class="${activeView === "reports" ? "active" : ""}">
              <span class="nav-label">${escapeHtml(t("nav.reports.label"))}</span>
              <span class="nav-copy">${escapeHtml(t("nav.reports.copy"))}</span>
            </a>
          </nav>
          <div class="meta">
            <span class="chip">${escapeHtml(paper.primary_category || t("common.na"))}</span>
            <span class="chip">${escapeHtml(t("paper.published", { value: formatTime(paper.published) }))}</span>
            <span class="chip">${escapeHtml(t("paper.updated", { value: formatTime(paper.updated) }))}</span>
            <span class="chip">${escapeHtml(t("paper.score", { value: Number(paper.llm_score || 0).toFixed(2) }))}</span>
            ${paper.is_starred ? `<span class="chip">${escapeHtml(t("paper.starred"))}</span>` : ""}
            ${paper.is_ignored ? `<span class="chip">${escapeHtml(t("paper.ignored"))}</span>` : ""}
          </div>
          <div class="actions">
            <a href="${escapeHtml(returnTo)}">${escapeHtml(t("paper.back"))}</a>
            <a href="${escapeHtml(paper.entry_id)}" target="_blank" rel="noreferrer">${escapeHtml(t("paper.open_arxiv"))}</a>
            ${paper.pdf_url ? `<a href="${escapeHtml(paper.pdf_url)}" target="_blank" rel="noreferrer">${escapeHtml(t("paper.open_pdf"))}</a>` : ""}
            <a href="#" id="copy-paper-link">${escapeHtml(t("paper.copy_link"))}</a>
          </div>
        </section>

        <section class="panel">
          <div class="section-label">${escapeHtml(t("meta.section"))}</div>
          <h2 class="section-title">${escapeHtml(t("meta.title"))}</h2>
          <div class="grid" style="margin-top: 14px;">
            <div class="card">
              <strong>${escapeHtml(t("meta.authors"))}</strong>
              <div>${escapeHtml((paper.authors || []).join(", ") || t("common.na"))}</div>
            </div>
            <div class="card">
              <strong>${escapeHtml(t("meta.categories"))}</strong>
              <div>${escapeHtml((paper.categories || []).join(", ") || t("common.na"))}</div>
            </div>
            <div class="card">
              <strong>${escapeHtml(t("meta.tracked_topics"))}</strong>
              <div class="topic-group">${renderTopicLinks(paper.tracked_topics || [], true)}</div>
            </div>
            <div class="card">
              <strong>${escapeHtml(t("meta.model_topics"))}</strong>
              <div class="topic-group">${renderTopicLinks(paper.topics || [], false)}</div>
            </div>
            <div class="card">
              <strong>${escapeHtml(t("meta.relevance_reason"))}</strong>
              <div>${escapeHtml(paper.relevance_reason || t("common.na"))}</div>
            </div>
            <div class="card">
              <strong>${escapeHtml(t("meta.raw_abstract"))}</strong>
              <div>${escapeHtml(paper.summary || t("common.na"))}</div>
            </div>
          </div>
        </section>

        <section class="panel">
          <div class="section-label">${escapeHtml(t("workflow.section"))}</div>
          <h2 class="section-title">${escapeHtml(t("workflow.title"))}</h2>
          <div class="grid" style="margin-top: 14px;">
            <div class="card">
              <strong>${escapeHtml(t("workflow.quick_actions"))}</strong>
              <div class="muted">${escapeHtml(t("workflow.quick_actions.copy"))}</div>
              <div class="button-row">
                <button type="button" id="toggle-star-button">${escapeHtml(paper.is_starred ? t("workflow.unstar") : t("workflow.star"))}</button>
                <button type="button" id="toggle-ignore-button" class="secondary">${escapeHtml(paper.is_ignored ? t("workflow.unignore") : t("workflow.ignore"))}</button>
                <button type="button" id="requeue-button">${escapeHtml(t("workflow.reanalyze"))}</button>
              </div>
              <div class="muted" style="margin-top: 12px;">${escapeHtml(t("workflow.last_review", { value: formatTime(paper.user_action_updated_at || "") }))}</div>
            </div>
            <div class="card">
              <strong>${escapeHtml(t("workflow.manual_topics"))}</strong>
              <label class="field-label" for="manual-topics-input">${escapeHtml(t("workflow.manual_topics_label"))}</label>
              <input id="manual-topics-input" type="text" value="${escapeHtml((paper.manual_topics || []).join(", "))}" placeholder="${escapeHtml(t("workflow.manual_topics_placeholder"))}">
              <label class="field-label" for="manual-note-input" style="margin-top: 14px;">${escapeHtml(t("workflow.analyst_note"))}</label>
              <textarea id="manual-note-input" placeholder="${escapeHtml(t("workflow.analyst_note_placeholder"))}">${escapeHtml(paper.manual_note || "")}</textarea>
              <div class="button-row">
                <button type="button" id="save-review-button">${escapeHtml(t("workflow.save_review"))}</button>
              </div>
            </div>
          </div>
        </section>

        ${paper.analysis_status === "analyzed" ? renderAnalyzedSections(paper) : `
          <section class="panel">
            <div class="section-label">${escapeHtml(t("status.section"))}</div>
            <h2 class="section-title">${escapeHtml(t("status.title"))}</h2>
            <div class="card" style="margin-top: 14px;">
              <strong>${escapeHtml(t("status.detail"))}</strong>
              <div>${escapeHtml(paper.analysis_error || paper.rejected_reason || paper.relevance_reason || paper.summary || t("common.na"))}</div>
            </div>
          </section>
        `}
        ${renderRelatedPapers(paper)}
      `;

      if (mobileNav) {
        mobileNav.innerHTML = `
          <a href="/" class="${activeView === "overview" ? "active" : ""}">${escapeHtml(t("nav.overview.label"))}</a>
          <a href="/papers" class="${activeView === "papers" ? "active" : ""}">${escapeHtml(t("nav.papers.label"))}</a>
          <a href="/reports" class="${activeView === "reports" ? "active" : ""}">${escapeHtml(t("nav.reports.label"))}</a>
        `;
      }

      const copy = document.getElementById("copy-paper-link");
      if (copy) {
        copy.addEventListener("click", async (event) => {
          event.preventDefault();
          try {
            await navigator.clipboard.writeText(window.location.href);
            copy.textContent = t("paper.link_copied");
          } catch (error) {
            copy.textContent = t("paper.copy_failed");
          }
        });
      }

      const entryId = paper.entry_id;
      const saveButton = document.getElementById("save-review-button");
      if (saveButton) {
        saveButton.addEventListener("click", async () => {
          const topics = document.getElementById("manual-topics-input").value
            .split(",")
            .map((item) => item.trim())
            .filter(Boolean);
          const note = document.getElementById("manual-note-input").value;
          const updated = await postPaperAction({
            entry_id: entryId,
            manual_topics: topics,
            manual_note: note,
          });
          renderPaper(updated);
        });
      }

      const starButton = document.getElementById("toggle-star-button");
      if (starButton) {
        starButton.addEventListener("click", async () => {
          const updated = await postPaperAction({
            entry_id: entryId,
            starred: !paper.is_starred,
          });
          renderPaper(updated);
        });
      }

      const ignoreButton = document.getElementById("toggle-ignore-button");
      if (ignoreButton) {
        ignoreButton.addEventListener("click", async () => {
          const updated = await postPaperAction({
            entry_id: entryId,
            ignored: !paper.is_ignored,
          });
          renderPaper(updated);
        });
      }

      const requeueButton = document.getElementById("requeue-button");
      if (requeueButton) {
        requeueButton.addEventListener("click", async () => {
          const updated = await postPaperAction({
            entry_id: entryId,
            requeue: true,
          });
          renderPaper(updated);
        });
      }
    }

    async function main() {
      applyLanguageState();
      const params = new URLSearchParams(window.location.search);
      const entryId = params.get("entry_id");
      const root = document.getElementById("paper-root");
      root.textContent = t("loading");
      if (!entryId) {
        root.textContent = t("missing_entry");
        return;
      }

      const response = await fetch(`/api/paper?entry_id=${encodeURIComponent(entryId)}`);
      if (!response.ok) {
        root.textContent = t("paper_not_found");
        return;
      }
      const payload = await response.json();
      renderPaper(payload);
    }

    main();
  </script>
</body>
</html>
"""

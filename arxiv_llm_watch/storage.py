from __future__ import annotations

import json
import sqlite3
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, List

from .models import Paper, PaperAnalysis
from .topics import extract_tracked_topics


class Storage:
    def __init__(self, db_path: Path) -> None:
        self.connection = sqlite3.connect(str(db_path))
        self.connection.row_factory = sqlite3.Row

    def initialize(self) -> None:
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS papers (
                entry_id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                summary TEXT NOT NULL,
                published TEXT NOT NULL,
                updated TEXT NOT NULL,
                primary_category TEXT NOT NULL,
                categories_json TEXT NOT NULL,
                authors_json TEXT NOT NULL,
                pdf_url TEXT,
                fetched_at TEXT NOT NULL,
                analysis_status TEXT NOT NULL DEFAULT 'pending',
                rejected_reason TEXT,
                llm_score REAL,
                relevance_reason TEXT,
                llm_topics_json TEXT,
                summary_zh TEXT,
                summary_en TEXT,
                background_zh TEXT,
                background_en TEXT,
                problem_zh TEXT,
                problem_en TEXT,
                method_zh TEXT,
                method_en TEXT,
                findings_zh TEXT,
                findings_en TEXT,
                limitations_zh TEXT,
                limitations_en TEXT,
                analysis_error TEXT
            );

            CREATE TABLE IF NOT EXISTS run_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'running',
                started_at TEXT NOT NULL,
                finished_at TEXT,
                overrides_json TEXT,
                fetched_count INTEGER NOT NULL DEFAULT 0,
                analysis_batch_count INTEGER NOT NULL DEFAULT 0,
                analyzed_count INTEGER NOT NULL DEFAULT 0,
                filtered_count INTEGER NOT NULL DEFAULT 0,
                rejected_count INTEGER NOT NULL DEFAULT 0,
                error_count INTEGER NOT NULL DEFAULT 0,
                report_path TEXT,
                error_message TEXT
            );

            CREATE INDEX IF NOT EXISTS idx_papers_published ON papers(published);
            CREATE INDEX IF NOT EXISTS idx_papers_status ON papers(analysis_status);
            CREATE INDEX IF NOT EXISTS idx_run_history_started_at ON run_history(started_at);
            """
        )
        self._ensure_column("papers", "is_starred", "INTEGER NOT NULL DEFAULT 0")
        self._ensure_column("papers", "is_ignored", "INTEGER NOT NULL DEFAULT 0")
        self._ensure_column("papers", "manual_topics_json", "TEXT")
        self._ensure_column("papers", "manual_note", "TEXT")
        self._ensure_column("papers", "user_action_updated_at", "TEXT")
        self.connection.commit()

    def save_papers(self, papers: Iterable[Paper], fetched_at_iso: str) -> None:
        rows = [
            (
                paper.entry_id,
                paper.title,
                paper.summary,
                paper.published.isoformat(),
                paper.updated.isoformat(),
                paper.primary_category,
                json.dumps(paper.categories, ensure_ascii=True),
                json.dumps(paper.authors, ensure_ascii=True),
                paper.pdf_url,
                fetched_at_iso,
                "pending",
            )
            for paper in papers
        ]
        self.connection.executemany(
            """
            INSERT INTO papers (
                entry_id, title, summary, published, updated, primary_category,
                categories_json, authors_json, pdf_url, fetched_at, analysis_status
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(entry_id) DO UPDATE SET
                title = excluded.title,
                summary = excluded.summary,
                updated = excluded.updated,
                primary_category = excluded.primary_category,
                categories_json = excluded.categories_json,
                authors_json = excluded.authors_json,
                pdf_url = excluded.pdf_url,
                fetched_at = excluded.fetched_at
            """,
            rows,
        )
        self.connection.commit()

    def list_pending_papers(self, limit: int | None = None) -> List[Paper]:
        query = """
            SELECT entry_id, title, summary, published, updated,
                   primary_category, categories_json, authors_json, pdf_url
            FROM papers
            WHERE analysis_status = 'pending'
              AND COALESCE(is_ignored, 0) = 0
            ORDER BY published DESC
        """
        params = ()
        if limit is not None and limit > 0:
            query += "\nLIMIT ?"
            params = (limit,)
        cursor = self.connection.execute(query, params)
        rows = cursor.fetchall()
        return [self._row_to_paper(row) for row in rows]

    def mark_filtered(self, entry_id: str, reason: str) -> None:
        self._mark_status(entry_id, "filtered", reason=reason)

    def mark_rejected(self, entry_id: str, reason: str, score: float) -> None:
        self.connection.execute(
            """
            UPDATE papers
            SET analysis_status = 'rejected',
                rejected_reason = ?,
                llm_score = ?,
                relevance_reason = ?
            WHERE entry_id = ?
            """,
            (reason, score, reason, entry_id),
        )
        self.connection.commit()

    def mark_error(self, entry_id: str, message: str) -> None:
        self.connection.execute(
            """
            UPDATE papers
            SET analysis_status = 'error',
                analysis_error = ?
            WHERE entry_id = ?
            """,
            (message, entry_id),
        )
        self.connection.commit()

    def save_analysis(self, entry_id: str, analysis: PaperAnalysis) -> None:
        self.connection.execute(
            """
            UPDATE papers
            SET analysis_status = 'analyzed',
                rejected_reason = NULL,
                analysis_error = NULL,
                llm_score = ?,
                relevance_reason = ?,
                llm_topics_json = ?,
                summary_zh = ?,
                summary_en = ?,
                background_zh = ?,
                background_en = ?,
                problem_zh = ?,
                problem_en = ?,
                method_zh = ?,
                method_en = ?,
                findings_zh = ?,
                findings_en = ?,
                limitations_zh = ?,
                limitations_en = ?
            WHERE entry_id = ?
            """,
            (
                analysis.llm_score,
                analysis.relevance_reason,
                json.dumps(analysis.topics, ensure_ascii=True),
                analysis.summary.zh,
                analysis.summary.en,
                analysis.background.zh,
                analysis.background.en,
                analysis.problem.zh,
                analysis.problem.en,
                analysis.method.zh,
                analysis.method.en,
                analysis.findings.zh,
                analysis.findings.en,
                analysis.limitations.zh,
                analysis.limitations.en,
                entry_id,
            ),
        )
        self.connection.commit()

    def list_topic_window(self, days: int) -> List[dict]:
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
        cursor = self.connection.execute(
            """
            SELECT entry_id, title, summary, categories_json, published, llm_topics_json, manual_topics_json
            FROM papers
            WHERE analysis_status = 'analyzed'
              AND published >= ?
            ORDER BY published DESC
            """,
            (cutoff,),
        )
        rows = cursor.fetchall()
        return [
            {
                "entry_id": row["entry_id"],
                "title": row["title"],
                "summary": row["summary"],
                "published": row["published"],
                "topics": json.loads(row["llm_topics_json"] or "[]"),
                "tracked_topics": self._merge_topics(
                    json.loads(row["manual_topics_json"] or "[]"),
                    extract_tracked_topics(
                        title=row["title"],
                        summary=row["summary"] or "",
                        categories=json.loads(row["categories_json"] or "[]"),
                        raw_topics=json.loads(row["llm_topics_json"] or "[]"),
                    ),
                ),
            }
            for row in rows
        ]

    def list_report_papers(self, days: int, limit: int) -> List[dict]:
        return self.list_report_papers_window(days=days, offset_days=0, limit=limit)

    def list_report_papers_between(
        self,
        start: datetime,
        end: datetime,
        limit: int | None = None,
    ) -> List[dict]:
        return self._list_analyzed_papers_between(start=start, end=end, limit=limit)

    def list_report_papers_window(
        self,
        days: int,
        offset_days: int = 0,
        limit: int | None = None,
    ) -> List[dict]:
        now = datetime.now(timezone.utc)
        window_end = now - timedelta(days=max(0, offset_days))
        window_start = window_end - timedelta(days=max(1, days))
        return self._list_analyzed_papers_between(start=window_start, end=window_end, limit=limit)

    def count_status_between(
        self,
        start: datetime,
        end: datetime,
    ) -> dict:
        cursor = self.connection.execute(
            """
            SELECT analysis_status, COUNT(*) AS count
            FROM papers
            WHERE published >= ?
              AND published < ?
            GROUP BY analysis_status
            """,
            (start.isoformat(), end.isoformat()),
        )
        counts = {row["analysis_status"]: int(row["count"]) for row in cursor.fetchall()}
        counts["total"] = sum(counts.values())
        return counts

    def _list_analyzed_papers_between(
        self,
        start: datetime,
        end: datetime,
        limit: int | None = None,
    ) -> List[dict]:
        limit_clause = "LIMIT ?" if limit is not None and limit > 0 else ""
        params: list[object] = [start.isoformat(), end.isoformat()]
        if limit_clause:
            params.append(limit)
        cursor = self.connection.execute(
            f"""
            SELECT *
            FROM papers
            WHERE analysis_status = 'analyzed'
              AND published >= ?
              AND published < ?
            ORDER BY published DESC
            {limit_clause}
            """,
            tuple(params),
        )
        rows = cursor.fetchall()
        results = []
        for row in rows:
            raw_topics = json.loads(row["llm_topics_json"] or "[]")
            tracked_topics = self._merge_topics(
                json.loads(row["manual_topics_json"] or "[]"),
                extract_tracked_topics(
                    title=row["title"],
                    summary=row["summary"] or "",
                    categories=json.loads(row["categories_json"] or "[]"),
                    raw_topics=raw_topics,
                ),
            )
            results.append(
                {
                    "entry_id": row["entry_id"],
                    "title": row["title"],
                    "summary": row["summary"] or "",
                    "summary_zh": row["summary_zh"] or "",
                    "summary_en": row["summary_en"] or "",
                    "background_zh": row["background_zh"] or "",
                    "background_en": row["background_en"] or "",
                    "problem_zh": row["problem_zh"] or "",
                    "problem_en": row["problem_en"] or "",
                    "method_zh": row["method_zh"] or "",
                    "method_en": row["method_en"] or "",
                    "findings_zh": row["findings_zh"] or "",
                    "findings_en": row["findings_en"] or "",
                    "limitations_zh": row["limitations_zh"] or "",
                    "limitations_en": row["limitations_en"] or "",
                    "manual_topics": json.loads(row["manual_topics_json"] or "[]"),
                    "manual_note": row["manual_note"] or "",
                    "is_starred": bool(row["is_starred"]),
                    "is_ignored": bool(row["is_ignored"]),
                    "published": row["published"],
                    "primary_category": row["primary_category"],
                    "categories": json.loads(row["categories_json"] or "[]"),
                    "topics": raw_topics,
                    "tracked_topics": tracked_topics,
                    "relevance_reason": row["relevance_reason"] or "",
                }
            )
        return results

    def list_papers_by_status(self, status: str, limit: int) -> List[dict]:
        cursor = self.connection.execute(
            """
            SELECT entry_id, title, published, primary_category, analysis_status, llm_score,
                   relevance_reason, rejected_reason, analysis_error,
                   llm_topics_json, categories_json, summary, summary_zh, summary_en,
                   problem_zh, problem_en, method_zh, method_en,
                   findings_zh, findings_en, limitations_zh, limitations_en,
                   is_starred, is_ignored, manual_topics_json, manual_note
            FROM papers
            WHERE analysis_status = ?
            ORDER BY published DESC
            LIMIT ?
            """,
            (status, limit),
        )
        rows = cursor.fetchall()
        return [self._row_to_paper_payload(row) for row in rows]

    def get_paper(self, entry_id: str) -> dict | None:
        cursor = self.connection.execute(
            """
            SELECT entry_id, title, published, updated, primary_category, analysis_status, llm_score,
                   relevance_reason, rejected_reason, analysis_error, pdf_url,
                   llm_topics_json, categories_json, authors_json, summary, summary_zh, summary_en,
                   background_zh, background_en, problem_zh, problem_en,
                   method_zh, method_en, findings_zh, findings_en, limitations_zh, limitations_en,
                   is_starred, is_ignored, manual_topics_json, manual_note, user_action_updated_at
            FROM papers
            WHERE entry_id = ?
            LIMIT 1
            """,
            (entry_id,),
        )
        row = cursor.fetchone()
        if row is None:
            return None

        payload = self._row_to_paper_payload(row)
        payload["updated"] = row["updated"]
        payload["pdf_url"] = row["pdf_url"] or ""
        payload["authors"] = json.loads(row["authors_json"] or "[]")
        payload["user_action_updated_at"] = row["user_action_updated_at"] or ""
        return payload

    def find_related_papers(self, entry_id: str, limit: int = 6) -> List[dict]:
        source = self.get_paper(entry_id)
        if source is None:
            return []

        source_tracked = set(source.get("tracked_topics", []))
        source_topics = set(source.get("topics", []))
        source_category = source.get("primary_category")

        cursor = self.connection.execute(
            """
            SELECT entry_id, title, published, primary_category, analysis_status, llm_score,
                   relevance_reason, rejected_reason, analysis_error,
                   llm_topics_json, categories_json, summary, summary_zh, summary_en,
                   background_zh, background_en, problem_zh, problem_en,
                   method_zh, method_en, findings_zh, findings_en, limitations_zh, limitations_en
            FROM papers
            WHERE analysis_status = 'analyzed'
              AND entry_id != ?
            ORDER BY published DESC
            """,
            (entry_id,),
        )
        rows = cursor.fetchall()
        candidates = []
        for row in rows:
            payload = self._row_to_paper_payload(row)
            payload_tracked = set(payload.get("tracked_topics", []))
            payload_topics = set(payload.get("topics", []))
            shared_tracked = sorted(source_tracked & payload_tracked)
            shared_topics = sorted(source_topics & payload_topics)
            score = (len(shared_tracked) * 10) + (len(shared_topics) * 2)
            if source_category and payload.get("primary_category") == source_category:
                score += 3
            if score <= 0:
                continue
            payload["similarity_score"] = score
            payload["shared_tracked_topics"] = shared_tracked
            payload["shared_model_topics"] = shared_topics
            candidates.append(payload)

        candidates.sort(
            key=lambda item: (
                item.get("similarity_score", 0),
                item.get("llm_score") or 0,
                item.get("published", ""),
                item.get("title", ""),
            ),
            reverse=True,
        )
        return candidates[: max(1, int(limit or 6))]

    def search_papers(
        self,
        query: str = "",
        status: str | None = None,
        category: str | None = None,
        tracked_topic: str = "",
        flag: str = "all",
        days: int | None = None,
        limit: int = 20,
        offset: int = 0,
        sort: str = "published_desc",
    ) -> List[dict]:
        offset = max(0, int(offset or 0))
        limit = max(1, int(limit or 20))
        matched = self._search_matching_papers(
            query=query,
            status=status,
            category=category,
            tracked_topic=tracked_topic,
            flag=flag,
            days=days,
            sort=sort,
        )
        return matched[offset : offset + limit]

    def count_search_papers(
        self,
        query: str = "",
        status: str | None = None,
        category: str | None = None,
        tracked_topic: str = "",
        flag: str = "all",
        days: int | None = None,
        sort: str = "published_desc",
    ) -> int:
        return len(
            self._search_matching_papers(
                query=query,
                status=status,
                category=category,
                tracked_topic=tracked_topic,
                flag=flag,
                days=days,
                sort=sort,
            )
        )

    def _search_matching_papers(
        self,
        query: str = "",
        status: str | None = None,
        category: str | None = None,
        tracked_topic: str = "",
        flag: str = "all",
        days: int | None = None,
        sort: str = "published_desc",
    ) -> List[dict]:
        conditions: List[str] = []
        params: List[object] = []

        if status and status != "all":
            conditions.append("analysis_status = ?")
            params.append(status)
        if category and category != "all":
            conditions.append("primary_category = ?")
            params.append(category)
        if days is not None and int(days) > 0:
            cutoff = (datetime.now(timezone.utc) - timedelta(days=int(days))).isoformat()
            conditions.append("published >= ?")
            params.append(cutoff)

        where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        cursor = self.connection.execute(
            f"""
            SELECT entry_id, title, published, primary_category, analysis_status, llm_score,
                   relevance_reason, rejected_reason, analysis_error,
                   llm_topics_json, categories_json, summary, summary_zh, summary_en,
                   problem_zh, problem_en, method_zh, method_en,
                   findings_zh, findings_en, limitations_zh, limitations_en,
                   is_starred, is_ignored, manual_topics_json, manual_note
            FROM papers
            {where_clause}
            ORDER BY published DESC
            """,
            tuple(params),
        )
        rows = cursor.fetchall()
        normalized_query = query.strip().lower()
        normalized_topic = tracked_topic.strip().lower()
        results: List[dict] = []
        for row in rows:
            payload = self._row_to_paper_payload(row)
            if not self._matches_flag(payload, flag):
                continue
            if normalized_topic and normalized_topic not in " ".join(topic.lower() for topic in payload.get("tracked_topics", [])):
                continue
            if normalized_query and normalized_query not in self._paper_search_blob(payload):
                continue
            payload["match_reasons"] = self._paper_match_reasons(payload, normalized_query)
            results.append(payload)
        self._sort_paper_payloads(results, sort)
        return results

    def build_search_facets(self, days: int | None = None) -> dict:
        papers = self._search_matching_papers(
            query="",
            status="all",
            category="all",
            tracked_topic="",
            flag="all",
            days=days,
            sort="published_desc",
        )
        status_counts: Counter[str] = Counter()
        category_counts: Counter[str] = Counter()
        topic_counts: Counter[str] = Counter()
        flags = {"starred": 0, "ignored": 0, "manual": 0}
        for paper in papers:
            status_counts.update([paper.get("analysis_status") or "unknown"])
            if paper.get("primary_category"):
                category_counts.update([paper["primary_category"]])
            topic_counts.update(set(paper.get("tracked_topics", [])))
            if paper.get("is_starred"):
                flags["starred"] += 1
            if paper.get("is_ignored"):
                flags["ignored"] += 1
            if paper.get("manual_topics") or paper.get("manual_note"):
                flags["manual"] += 1
        return {
            "statuses": [{"value": key, "count": status_counts[key]} for key in sorted(status_counts.keys())],
            "categories": [{"value": key, "count": category_counts[key]} for key in sorted(category_counts.keys())],
            "tracked_topics": [
                {"name": topic, "count": count}
                for topic, count in topic_counts.most_common(24)
            ],
            "flags": flags,
            "total": len(papers),
        }

    def update_paper_feedback(
        self,
        entry_id: str,
        *,
        starred: bool | None = None,
        ignored: bool | None = None,
        manual_topics: List[str] | None = None,
        manual_note: str | None = None,
    ) -> None:
        assignments: list[str] = ["user_action_updated_at = ?"]
        params: list[object] = [datetime.now(timezone.utc).isoformat()]
        if starred is not None:
            assignments.append("is_starred = ?")
            params.append(1 if starred else 0)
        if ignored is not None:
            assignments.append("is_ignored = ?")
            params.append(1 if ignored else 0)
        if manual_topics is not None:
            cleaned_topics = [topic.strip() for topic in manual_topics if topic.strip()]
            assignments.append("manual_topics_json = ?")
            params.append(json.dumps(cleaned_topics, ensure_ascii=True))
        if manual_note is not None:
            assignments.append("manual_note = ?")
            params.append((manual_note or "").strip())
        params.append(entry_id)
        self.connection.execute(
            f"""
            UPDATE papers
            SET {", ".join(assignments)}
            WHERE entry_id = ?
            """,
            tuple(params),
        )
        self.connection.commit()

    def requeue_paper(self, entry_id: str) -> None:
        self.connection.execute(
            """
            UPDATE papers
            SET analysis_status = 'pending',
                rejected_reason = NULL,
                llm_score = NULL,
                relevance_reason = NULL,
                llm_topics_json = NULL,
                summary_zh = NULL,
                summary_en = NULL,
                background_zh = NULL,
                background_en = NULL,
                problem_zh = NULL,
                problem_en = NULL,
                method_zh = NULL,
                method_en = NULL,
                findings_zh = NULL,
                findings_en = NULL,
                limitations_zh = NULL,
                limitations_en = NULL,
                analysis_error = NULL,
                is_ignored = 0,
                user_action_updated_at = ?
            WHERE entry_id = ?
            """,
            (datetime.now(timezone.utc).isoformat(), entry_id),
        )
        self.connection.commit()

    def start_run(self, source: str, overrides: dict | None = None) -> int:
        cursor = self.connection.execute(
            """
            INSERT INTO run_history (source, status, started_at, overrides_json)
            VALUES (?, 'running', ?, ?)
            """,
            (
                source,
                datetime.now(timezone.utc).isoformat(),
                json.dumps(overrides or {}, ensure_ascii=True),
            ),
        )
        self.connection.commit()
        return int(cursor.lastrowid)

    def mark_interrupted_runs(self, message: str = "Dashboard 进程重启，上一轮运行未完成。") -> int:
        cursor = self.connection.execute(
            """
            UPDATE run_history
            SET status = 'failed',
                finished_at = ?,
                error_message = CASE
                    WHEN COALESCE(error_message, '') = '' THEN ?
                    ELSE error_message
                END
            WHERE status = 'running'
            """,
            (datetime.now(timezone.utc).isoformat(), message),
        )
        self.connection.commit()
        return int(cursor.rowcount or 0)

    def finish_run(
        self,
        run_id: int,
        *,
        status: str,
        fetched_count: int = 0,
        analysis_batch_count: int = 0,
        analyzed_count: int = 0,
        filtered_count: int = 0,
        rejected_count: int = 0,
        error_count: int = 0,
        report_path: str = "",
        error_message: str = "",
    ) -> None:
        self.connection.execute(
            """
            UPDATE run_history
            SET status = ?,
                finished_at = ?,
                fetched_count = ?,
                analysis_batch_count = ?,
                analyzed_count = ?,
                filtered_count = ?,
                rejected_count = ?,
                error_count = ?,
                report_path = ?,
                error_message = ?
            WHERE id = ?
            """,
            (
                status,
                datetime.now(timezone.utc).isoformat(),
                fetched_count,
                analysis_batch_count,
                analyzed_count,
                filtered_count,
                rejected_count,
                error_count,
                report_path,
                error_message,
                run_id,
            ),
        )
        self.connection.commit()

    def list_run_history(self, limit: int = 10) -> List[dict]:
        cursor = self.connection.execute(
            """
            SELECT id, source, status, started_at, finished_at, overrides_json,
                   fetched_count, analysis_batch_count, analyzed_count, filtered_count,
                   rejected_count, error_count, report_path, error_message
            FROM run_history
            ORDER BY started_at DESC, id DESC
            LIMIT ?
            """,
            (limit,),
        )
        rows = cursor.fetchall()
        return [
            {
                "id": row["id"],
                "source": row["source"],
                "status": row["status"],
                "started_at": row["started_at"],
                "finished_at": row["finished_at"] or "",
                "overrides": json.loads(row["overrides_json"] or "{}"),
                "fetched_count": row["fetched_count"],
                "analysis_batch_count": row["analysis_batch_count"],
                "analyzed_count": row["analyzed_count"],
                "filtered_count": row["filtered_count"],
                "rejected_count": row["rejected_count"],
                "error_count": row["error_count"],
                "report_path": row["report_path"] or "",
                "error_message": row["error_message"] or "",
            }
            for row in rows
        ]

    def list_primary_categories(self) -> List[str]:
        cursor = self.connection.execute(
            """
            SELECT DISTINCT primary_category
            FROM papers
            WHERE primary_category IS NOT NULL
              AND primary_category != ''
            ORDER BY primary_category ASC
            """
        )
        return [row["primary_category"] for row in cursor.fetchall()]

    def count_recent_categories(self, days: int, status: str = "analyzed") -> List[dict]:
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
        cursor = self.connection.execute(
            """
            SELECT primary_category, COUNT(*) AS total
            FROM papers
            WHERE published >= ?
              AND analysis_status = ?
              AND primary_category IS NOT NULL
              AND primary_category != ''
            GROUP BY primary_category
            ORDER BY total DESC, primary_category ASC
            """,
            (cutoff, status),
        )
        return [
            {"category": row["primary_category"], "count": row["total"]}
            for row in cursor.fetchall()
        ]

    def count_by_status(self) -> dict:
        cursor = self.connection.execute(
            """
            SELECT analysis_status, COUNT(*) AS total
            FROM papers
            GROUP BY analysis_status
            """
        )
        return {row["analysis_status"]: row["total"] for row in cursor.fetchall()}

    def count_recent_by_status(self, days: int) -> dict:
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
        cursor = self.connection.execute(
            """
            SELECT analysis_status, COUNT(*) AS total
            FROM papers
            WHERE published >= ?
            GROUP BY analysis_status
            """,
            (cutoff,),
        )
        return {row["analysis_status"]: row["total"] for row in cursor.fetchall()}

    def close(self) -> None:
        self.connection.close()

    def _mark_status(self, entry_id: str, status: str, reason: str) -> None:
        self.connection.execute(
            """
            UPDATE papers
            SET analysis_status = ?,
                rejected_reason = ?
            WHERE entry_id = ?
            """,
            (status, reason, entry_id),
        )
        self.connection.commit()

    @staticmethod
    def _row_to_paper(row: sqlite3.Row) -> Paper:
        from datetime import datetime

        return Paper(
            entry_id=row["entry_id"],
            title=row["title"],
            summary=row["summary"],
            published=datetime.fromisoformat(row["published"]),
            updated=datetime.fromisoformat(row["updated"]),
            primary_category=row["primary_category"],
            categories=json.loads(row["categories_json"] or "[]"),
            authors=json.loads(row["authors_json"] or "[]"),
            pdf_url=row["pdf_url"],
        )

    def _row_to_paper_payload(self, row: sqlite3.Row) -> dict:
        raw_topics = json.loads(self._row_value(row, "llm_topics_json") or "[]")
        categories = json.loads(self._row_value(row, "categories_json") or "[]")
        manual_topics = json.loads(self._row_value(row, "manual_topics_json") or "[]")
        tracked_topics = self._merge_topics(
            manual_topics,
            extract_tracked_topics(
                title=self._row_value(row, "title") or "",
                summary=self._row_value(row, "summary") or "",
                categories=categories,
                raw_topics=raw_topics,
            ),
        )
        return {
            "entry_id": row["entry_id"],
            "title": row["title"],
            "published": row["published"],
            "primary_category": row["primary_category"],
            "analysis_status": row["analysis_status"],
            "llm_score": row["llm_score"],
            "relevance_reason": self._row_value(row, "relevance_reason") or "",
            "rejected_reason": self._row_value(row, "rejected_reason") or "",
            "analysis_error": self._row_value(row, "analysis_error") or "",
            "topics": raw_topics,
            "tracked_topics": tracked_topics,
            "manual_topics": manual_topics,
            "manual_note": self._row_value(row, "manual_note") or "",
            "is_starred": bool(self._row_value(row, "is_starred") or 0),
            "is_ignored": bool(self._row_value(row, "is_ignored") or 0),
            "categories": categories,
            "summary": self._row_value(row, "summary") or "",
            "summary_zh": self._row_value(row, "summary_zh") or "",
            "summary_en": self._row_value(row, "summary_en") or "",
            "background_zh": self._row_value(row, "background_zh") or "",
            "background_en": self._row_value(row, "background_en") or "",
            "problem_zh": self._row_value(row, "problem_zh") or "",
            "problem_en": self._row_value(row, "problem_en") or "",
            "method_zh": self._row_value(row, "method_zh") or "",
            "method_en": self._row_value(row, "method_en") or "",
            "findings_zh": self._row_value(row, "findings_zh") or "",
            "findings_en": self._row_value(row, "findings_en") or "",
            "limitations_zh": self._row_value(row, "limitations_zh") or "",
            "limitations_en": self._row_value(row, "limitations_en") or "",
        }

    @staticmethod
    def _paper_search_blob(paper: dict) -> str:
        fields = [
            paper.get("title", ""),
            paper.get("summary", ""),
            paper.get("summary_zh", ""),
            paper.get("summary_en", ""),
            paper.get("relevance_reason", ""),
            paper.get("manual_note", ""),
            " ".join(paper.get("tracked_topics", [])),
            " ".join(paper.get("topics", [])),
            " ".join(paper.get("manual_topics", [])),
        ]
        return " ".join(str(field).lower() for field in fields if field)

    @staticmethod
    def _row_value(row: sqlite3.Row, key: str) -> object | None:
        return row[key] if key in row.keys() else None

    @staticmethod
    def _merge_topics(manual_topics: List[str], auto_topics: List[str]) -> List[str]:
        merged: list[str] = []
        for topic in [*(manual_topics or []), *(auto_topics or [])]:
            cleaned = str(topic or "").strip()
            if cleaned and cleaned not in merged:
                merged.append(cleaned)
        return merged

    @staticmethod
    def _matches_flag(paper: dict, flag: str) -> bool:
        normalized = (flag or "all").strip().lower()
        if normalized == "starred":
            return bool(paper.get("is_starred"))
        if normalized == "ignored":
            return bool(paper.get("is_ignored"))
        if normalized == "manual":
            return bool(paper.get("manual_topics") or paper.get("manual_note"))
        return True

    @staticmethod
    def _paper_match_reasons(paper: dict, query: str) -> List[str]:
        normalized = (query or "").strip().lower()
        if not normalized:
            return []
        candidates = [
            ("title", paper.get("title", "")),
            ("abstract", paper.get("summary", "")),
            ("summary", f"{paper.get('summary_zh', '')} {paper.get('summary_en', '')}"),
            ("tracked topic", " ".join(paper.get("tracked_topics", []))),
            ("model topic", " ".join(paper.get("topics", []))),
            ("manual note", paper.get("manual_note", "")),
        ]
        reasons = [label for label, value in candidates if normalized in str(value).lower()]
        return reasons[:4]

    def _ensure_column(self, table: str, column: str, definition: str) -> None:
        cursor = self.connection.execute(f"PRAGMA table_info({table})")
        columns = {row["name"] for row in cursor.fetchall()}
        if column in columns:
            return
        self.connection.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")

    @staticmethod
    def _sort_paper_payloads(papers: List[dict], sort: str) -> None:
        sort = (sort or "published_desc").strip().lower()
        if sort == "published_asc":
            papers.sort(key=lambda item: (item.get("published", ""), item.get("title", "")))
            return
        if sort == "score_desc":
            papers.sort(
                key=lambda item: (
                    item.get("llm_score") or 0,
                    item.get("published", ""),
                    item.get("title", ""),
                ),
                reverse=True,
            )
            return
        if sort == "title_asc":
            papers.sort(key=lambda item: ((item.get("title") or "").lower(), item.get("published", "")))
            return
        papers.sort(
            key=lambda item: (
                item.get("published", ""),
                item.get("llm_score") or 0,
                item.get("title", ""),
            ),
            reverse=True,
        )

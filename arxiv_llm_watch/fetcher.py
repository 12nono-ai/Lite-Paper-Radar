from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import List, Set

from .models import Paper

try:
    import arxiv
except ImportError:  # pragma: no cover
    arxiv = None


def _normalize_datetime(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _escape_query_term(value: str) -> str:
    return value.replace('"', '\\"')


class ArxivFetcher:
    def __init__(self, page_size: int = 100, delay_seconds: float = 3.0) -> None:
        if arxiv is None:
            raise RuntimeError("The `arxiv` package is not installed. Run `pip3 install -r requirements.txt` first.")
        self.client = arxiv.Client(page_size=page_size, delay_seconds=delay_seconds, num_retries=3)

    @staticmethod
    def build_query(categories: List[str], keywords: List[str] | None = None) -> str:
        category_query = " OR ".join(f"cat:{category}" for category in categories)
        normalized_keywords = [keyword.strip() for keyword in (keywords or []) if keyword.strip()]
        if not normalized_keywords:
            return category_query
        keyword_query = " OR ".join(f'all:"{_escape_query_term(keyword)}"' for keyword in normalized_keywords)
        return f"({category_query}) AND ({keyword_query})"

    def fetch_recent_papers(
        self,
        categories: List[str],
        max_results: int,
        lookback_days: int,
        keywords: List[str] | None = None,
    ) -> List[Paper]:
        search = arxiv.Search(
            query=self.build_query(categories, keywords=keywords),
            max_results=max_results,
            sort_by=arxiv.SortCriterion.SubmittedDate,
        )
        cutoff = datetime.now(timezone.utc) - timedelta(days=lookback_days)
        seen_ids: Set[str] = set()
        papers: List[Paper] = []

        for result in self.client.results(search):
            published = _normalize_datetime(result.published)
            if published < cutoff:
                break

            entry_id = result.entry_id
            if entry_id in seen_ids:
                continue
            seen_ids.add(entry_id)

            updated = _normalize_datetime(result.updated)
            papers.append(
                Paper(
                    entry_id=entry_id,
                    title=result.title.strip(),
                    summary=result.summary.strip(),
                    published=published,
                    updated=updated,
                    primary_category=(result.primary_category or "").strip(),
                    categories=[category.strip() for category in result.categories],
                    authors=[author.name.strip() for author in result.authors],
                    pdf_url=result.pdf_url,
                )
            )

        return papers

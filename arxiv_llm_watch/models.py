from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional


@dataclass
class Paper:
    entry_id: str
    title: str
    summary: str
    published: datetime
    updated: datetime
    primary_category: str
    categories: List[str]
    authors: List[str]
    pdf_url: Optional[str] = None


@dataclass
class SectionText:
    zh: str = ""
    en: str = ""


@dataclass
class PaperAnalysis:
    is_llm_related: bool
    relevance_reason: str
    llm_score: float
    topics: List[str] = field(default_factory=list)
    summary: SectionText = field(default_factory=SectionText)
    background: SectionText = field(default_factory=SectionText)
    problem: SectionText = field(default_factory=SectionText)
    method: SectionText = field(default_factory=SectionText)
    findings: SectionText = field(default_factory=SectionText)
    limitations: SectionText = field(default_factory=SectionText)


@dataclass
class TopicTrend:
    name: str
    current_count: int
    baseline_count: int
    growth: float


@dataclass
class ComparisonPoint:
    label: str
    zh: str = ""
    en: str = ""


@dataclass
class ReportComparison:
    overview: SectionText = field(default_factory=SectionText)
    common_ground: SectionText = field(default_factory=SectionText)
    method_landscape: SectionText = field(default_factory=SectionText)
    differences: List[ComparisonPoint] = field(default_factory=list)
    takeaways: List[ComparisonPoint] = field(default_factory=list)

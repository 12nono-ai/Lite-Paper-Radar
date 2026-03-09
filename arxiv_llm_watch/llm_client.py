from __future__ import annotations

import json
from typing import Any, Dict, List

from .config import AppConfig
from .models import ComparisonPoint, Paper, PaperAnalysis, ReportComparison, SectionText
from .topics import TRACKED_TOPIC_NAMES

try:
    from volcenginesdkarkruntime import Ark
except ImportError:  # pragma: no cover
    Ark = None


class LLMClient:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        if Ark is None:
            raise RuntimeError(
                "The `volcenginesdkarkruntime` package is not installed. "
                "Run `.venv/bin/pip install -r requirements.txt` first."
            )
        self.client = Ark(
            base_url=self.config.ark_base_url,
            api_key=self.config.ark_api_key,
        )

    def classify_and_summarize(self, paper: Paper) -> PaperAnalysis:
        completion = self.client.chat.completions.create(
            model=self.config.ark_model,
            temperature=self.config.llm_temperature,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a rigorous research analyst. "
                        "You must decide whether a paper is truly about LLMs or a closely related LLM system topic. "
                        "Return valid JSON only."
                    ),
                },
                {
                    "role": "user",
                    "content": self._build_prompt(paper),
                },
            ],
        )
        content = self._extract_content(completion)
        parsed = self._parse_content_json(content)
        return self._to_analysis(parsed)

    def compare_report_papers(self, papers: List[dict]) -> ReportComparison:
        if len(papers) < 2:
            return ReportComparison()

        completion = self.client.chat.completions.create(
            model=self.config.ark_model,
            temperature=0.1,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a rigorous research editor. "
                        "Synthesize cross-paper comparisons for an LLM paper report. "
                        "Return valid JSON only."
                    ),
                },
                {
                    "role": "user",
                    "content": self._build_comparison_prompt(papers),
                },
            ],
        )
        content = self._extract_content(completion)
        parsed = self._parse_content_json(content)
        return self._to_report_comparison(parsed)

    def _build_prompt(self, paper: Paper) -> str:
        schema = {
            "is_llm_related": True,
            "llm_score": 0.0,
            "relevance_reason": "short explanation",
            "topics": ["topic a", "topic b"],
            "summary_zh": "Chinese summary",
            "summary_en": "English summary",
            "background": {"zh": "", "en": ""},
            "problem": {"zh": "", "en": ""},
            "method": {"zh": "", "en": ""},
            "findings": {"zh": "", "en": ""},
            "limitations": {"zh": "", "en": ""},
        }
        return (
            "Analyze the following arXiv paper using only the provided metadata.\n"
            "Decide whether it is clearly about LLMs, LLM-based systems, LLM evaluation, "
            "LLM alignment, LLM inference, LLM agents, or closely related LLM applications.\n"
            "If it is not clearly LLM-related, set is_llm_related to false and still provide relevance_reason.\n\n"
            "For topics, prefer exact labels from this controlled vocabulary when they apply:\n"
            f"{json.dumps(TRACKED_TOPIC_NAMES, ensure_ascii=True)}\n"
            "You may add at most one extra custom topic if the vocabulary misses an important angle.\n\n"
            "Return compact JSON with exactly these keys:\n"
            f"{json.dumps(schema, ensure_ascii=True)}\n\n"
            f"Title: {paper.title}\n"
            f"Categories: {', '.join(paper.categories)}\n"
            f"Published: {paper.published.isoformat()}\n"
            f"Abstract: {paper.summary}\n"
        )

    def _build_comparison_prompt(self, papers: List[dict]) -> str:
        schema = {
            "overview": {"zh": "", "en": ""},
            "common_ground": {"zh": "", "en": ""},
            "method_landscape": {"zh": "", "en": ""},
            "differences": [{"label": "focus", "zh": "", "en": ""}],
            "takeaways": [{"label": "takeaway_1", "zh": "", "en": ""}],
        }
        digest = [
            {
                "title": paper["title"],
                "category": paper["primary_category"],
                "topics": paper.get("topics", [])[:6],
                "summary_en": paper.get("summary_en", ""),
                "problem_en": paper.get("problem_en", ""),
                "method_en": paper.get("method_en", ""),
                "findings_en": paper.get("findings_en", ""),
                "limitations_en": paper.get("limitations_en", ""),
            }
            for paper in papers
        ]
        return (
            "You are writing the final comparison section for a daily report that contains several LLM-related papers.\n"
            "Compare the papers along common research background, problem framing, method styles, and the most important differences.\n"
            "Return compact JSON with exactly these keys:\n"
            f"{json.dumps(schema, ensure_ascii=True)}\n\n"
            "Requirements:\n"
            "- overview: 2-3 sentence high-level synthesis\n"
            "- common_ground: what these papers are collectively responding to\n"
            "- method_landscape: how the approaches differ in style or level\n"
            "- differences: 2-4 concrete comparison bullets with short labels\n"
            "- takeaways: 2-4 actionable report takeaways\n"
            "- use concise Chinese and English text\n\n"
            f"Papers:\n{json.dumps(digest, ensure_ascii=False)}"
        )

    @staticmethod
    def _extract_content(completion: Any) -> str:
        choices = getattr(completion, "choices", None)
        if not choices:
            raise RuntimeError(f"Invalid Ark response, missing choices: {completion}")

        message = choices[0].message
        content = getattr(message, "content", "")
        if isinstance(content, str):
            return content.strip()
        raise RuntimeError(f"Unsupported Ark content format: {content}")

    @staticmethod
    def _parse_content_json(content: str) -> Dict[str, Any]:
        cleaned = content.strip()
        if cleaned.startswith("```"):
            parts = [part for part in cleaned.split("```") if part.strip()]
            for part in parts:
                candidate = part.strip()
                if candidate.startswith("json"):
                    candidate = candidate[4:].strip()
                if candidate.startswith("{"):
                    cleaned = candidate
                    break

        if not cleaned.startswith("{"):
            start = cleaned.find("{")
            end = cleaned.rfind("}")
            if start != -1 and end != -1:
                cleaned = cleaned[start : end + 1]

        try:
            return json.loads(cleaned)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Model did not return valid JSON: {content}") from exc

    @staticmethod
    def _to_analysis(payload: Dict[str, Any]) -> PaperAnalysis:
        def parse_section(name: str) -> SectionText:
            section = payload.get(name, {}) or {}
            if isinstance(section, str):
                return SectionText(zh=section, en=section)
            return SectionText(
                zh=str(section.get("zh", "")).strip(),
                en=str(section.get("en", "")).strip(),
            )

        raw_topics = payload.get("topics", []) or []
        topics = [str(item).strip() for item in raw_topics if str(item).strip()]

        return PaperAnalysis(
            is_llm_related=bool(payload.get("is_llm_related", False)),
            relevance_reason=str(payload.get("relevance_reason", "")).strip(),
            llm_score=float(payload.get("llm_score", 0.0) or 0.0),
            topics=topics,
            summary=SectionText(
                zh=str(payload.get("summary_zh", "")).strip(),
                en=str(payload.get("summary_en", "")).strip(),
            ),
            background=parse_section("background"),
            problem=parse_section("problem"),
            method=parse_section("method"),
            findings=parse_section("findings"),
            limitations=parse_section("limitations"),
        )

    @staticmethod
    def _to_report_comparison(payload: Dict[str, Any]) -> ReportComparison:
        def parse_section(name: str) -> SectionText:
            section = payload.get(name, {}) or {}
            if isinstance(section, str):
                return SectionText(zh=section, en=section)
            return SectionText(
                zh=str(section.get("zh", "")).strip(),
                en=str(section.get("en", "")).strip(),
            )

        def parse_points(name: str) -> List[ComparisonPoint]:
            points = []
            for item in payload.get(name, []) or []:
                if not isinstance(item, dict):
                    continue
                points.append(
                    ComparisonPoint(
                        label=str(item.get("label", "")).strip(),
                        zh=str(item.get("zh", "")).strip(),
                        en=str(item.get("en", "")).strip(),
                    )
                )
            return points

        return ReportComparison(
            overview=parse_section("overview"),
            common_ground=parse_section("common_ground"),
            method_landscape=parse_section("method_landscape"),
            differences=parse_points("differences"),
            takeaways=parse_points("takeaways"),
        )

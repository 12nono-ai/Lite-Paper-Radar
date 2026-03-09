from __future__ import annotations

from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, List

from .models import ComparisonPoint, ReportComparison, SectionText, TopicTrend
from .topics import display_topics_for_paper, format_topic_momentum


def _ascii_bar(value: int, maximum: int, width: int = 16) -> str:
    if maximum <= 0:
        return ""
    filled = max(1, round((value / maximum) * width)) if value > 0 else 0
    return "#" * filled


def build_fallback_report_comparison(report_papers: List[dict], topic_trends: List[TopicTrend]) -> ReportComparison:
    if not report_papers:
        return ReportComparison()

    category_counter = Counter(paper["primary_category"] for paper in report_papers if paper.get("primary_category"))
    top_categories = ", ".join(f"{name} ({count})" for name, count in category_counter.most_common(3)) or "n/a"

    topic_counter = Counter()
    for paper in report_papers:
        topic_counter.update(display_topics_for_paper(paper))
    top_topics = ", ".join(name for name, _ in topic_counter.most_common(5)) or "n/a"

    differences = []
    for paper in report_papers[:3]:
        differences.append(
            ComparisonPoint(
                label=paper["title"],
                zh=f"重点主题：{', '.join(display_topics_for_paper(paper)[:3]) or paper.get('primary_category', 'n/a')}",
                en=f"Primary focus: {', '.join(display_topics_for_paper(paper)[:3]) or paper.get('primary_category', 'n/a')}",
            )
        )

    takeaways = []
    for trend in topic_trends[:3]:
        takeaways.append(
            ComparisonPoint(
                label=trend.name,
                zh=f"{trend.name} 在近期窗口中出现 {trend.current_count} 次，说明它是当前值得优先追踪的方向。",
                en=f"{trend.name} appears {trend.current_count} times in the recent window, making it a priority direction to watch.",
            )
        )

    return ReportComparison(
        overview=SectionText(
            zh=f"本期纳入 {len(report_papers)} 篇论文，主要分布在 {top_categories}，集中围绕 {top_topics} 等方向展开。",
            en=f"This report covers {len(report_papers)} papers, mainly distributed across {top_categories}, with recurring themes such as {top_topics}.",
        ),
        common_ground=SectionText(
            zh="这些论文共同回应了 LLM 系统在评测、训练效率、内部机制和可靠性方面的核心压力。",
            en="These papers collectively respond to core pressure points in LLM systems around evaluation, training efficiency, internal mechanisms, and reliability.",
        ),
        method_landscape=SectionText(
            zh="方法上既有针对训练与架构机制的底层改进，也有面向评测、对齐和行为分析的上层方法；整体呈现出系统优化与能力诊断并行推进的格局。",
            en="Methodologically, the set spans lower-level training and architecture improvements as well as higher-level evaluation, alignment, and behavior analysis, showing parallel progress in system optimization and capability diagnosis.",
        ),
        differences=differences or [
            ComparisonPoint(
                label="focus",
                zh="各论文在训练优化、机制解释、评测基准和安全分析上关注点不同。",
                en="The papers differ in emphasis across training optimization, mechanistic interpretation, evaluation benchmarks, and safety analysis.",
            )
        ],
        takeaways=takeaways or [
            ComparisonPoint(
                label="follow_up",
                zh="优先跟踪近期反复出现的主题标签，并观察它们是否持续跨多篇论文出现。",
                en="Prioritize topic labels that recur across recent papers and track whether they persist across multiple batches.",
            )
        ],
    )


def build_window_delta(current_window_papers: List[dict], previous_window_papers: List[dict]) -> dict:
    current_ids = {paper["entry_id"] for paper in current_window_papers}
    previous_ids = {paper["entry_id"] for paper in previous_window_papers}
    new_papers = [paper for paper in current_window_papers if paper["entry_id"] not in previous_ids][:5]
    dropped_papers = [paper for paper in previous_window_papers if paper["entry_id"] not in current_ids][:5]

    current_topics: Counter[str] = Counter()
    previous_topics: Counter[str] = Counter()
    for paper in current_window_papers:
        current_topics.update(set(display_topics_for_paper(paper)))
    for paper in previous_window_papers:
        previous_topics.update(set(display_topics_for_paper(paper)))

    topic_names = set(current_topics) | set(previous_topics)
    deltas = []
    for name in topic_names:
        current_count = current_topics.get(name, 0)
        previous_count = previous_topics.get(name, 0)
        diff = current_count - previous_count
        if diff == 0:
            continue
        deltas.append(
            {
                "name": name,
                "diff": diff,
                "current_count": current_count,
                "previous_count": previous_count,
            }
        )
    rising_topics = sorted(
        [item for item in deltas if item["diff"] > 0],
        key=lambda item: (item["diff"], item["current_count"], item["name"]),
        reverse=True,
    )[:5]
    cooling_topics = sorted(
        [item for item in deltas if item["diff"] < 0],
        key=lambda item: (item["diff"], item["current_count"], item["name"]),
    )[:5]

    return {
        "current_count": len(current_window_papers),
        "previous_count": len(previous_window_papers),
        "new_papers": new_papers,
        "dropped_papers": dropped_papers,
        "rising_topics": rising_topics,
        "cooling_topics": cooling_topics,
    }


def render_markdown_report(
    generated_at: datetime,
    report_papers: List[dict],
    topic_trends: List[TopicTrend],
    status_counts: Dict[str, int],
    comparison: ReportComparison | None = None,
    window_delta: dict | None = None,
) -> str:
    lines = [
        f"# arXiv LLM Watch Report - {generated_at.strftime('%Y-%m-%d')}",
        "",
        "## Snapshot",
        "",
        f"- analyzed: {status_counts.get('analyzed', 0)}",
        "",
    ]

    if window_delta:
        lines.extend(
            [
                "## Since Last Window",
                "",
                f"- current window papers: {window_delta.get('current_count', 0)}",
                f"- previous window papers: {window_delta.get('previous_count', 0)}",
                f"- new papers: {len(window_delta.get('new_papers', []))}",
                f"- dropped papers: {len(window_delta.get('dropped_papers', []))}",
                "",
            ]
        )
        if window_delta.get("rising_topics"):
            lines.append("**Rising Topics**")
            lines.append("")
            for item in window_delta["rising_topics"]:
                lines.append(
                    f"- {item['name']}: {item['previous_count']} -> {item['current_count']} ({item['diff']:+d})"
                )
            lines.append("")
        if window_delta.get("cooling_topics"):
            lines.append("**Cooling Topics**")
            lines.append("")
            for item in window_delta["cooling_topics"]:
                lines.append(
                    f"- {item['name']}: {item['previous_count']} -> {item['current_count']} ({item['diff']:+d})"
                )
            lines.append("")
        if window_delta.get("new_papers"):
            lines.append("**New In This Window**")
            lines.append("")
            for paper in window_delta["new_papers"]:
                lines.append(f"- [{paper['title']}]({paper['entry_id']})")
            lines.append("")
        if window_delta.get("dropped_papers"):
            lines.append("**No Longer In This Window**")
            lines.append("")
            for paper in window_delta["dropped_papers"]:
                lines.append(f"- [{paper['title']}]({paper['entry_id']})")
            lines.append("")

    lines.extend(["## Hot Topics", ""])

    if topic_trends:
        lines.extend(
            [
                "| topic | recent | baseline | momentum |",
                "| --- | ---: | ---: | ---: |",
            ]
        )
        for trend in topic_trends:
            lines.append(
                f"| {trend.name} | {trend.current_count} | {trend.baseline_count} | {format_topic_momentum(trend.current_count, trend.baseline_count)} |"
            )
    else:
        lines.append("No topic trend data yet.")

    lines.extend(["", "## Visual Snapshot", ""])
    category_counter = Counter(paper["primary_category"] for paper in report_papers if paper.get("primary_category"))
    max_category = max(category_counter.values(), default=0)
    if category_counter:
        lines.append("**Category Mix**")
        lines.append("")
        for category, count in category_counter.most_common():
            lines.append(f"- {category:8} | {_ascii_bar(count, max_category)} ({count})")
        lines.append("")

    if topic_trends:
        lines.append("**Topic Momentum**")
        lines.append("")
        max_topic = max(trend.current_count for trend in topic_trends)
        for trend in topic_trends[:6]:
            lines.append(
                f"- {trend.name} | {_ascii_bar(trend.current_count, max_topic)} ({format_topic_momentum(trend.current_count, trend.baseline_count)})"
            )
        lines.append("")

    lines.extend(["## Papers", ""])
    if not report_papers:
        lines.append("No analyzed papers available for this report.")
        return "\n".join(lines)

    for paper in report_papers:
        lines.extend(
            [
                f"### [{paper['title']}]({paper['entry_id']})",
                "",
                f"- arXiv: [paper link]({paper['entry_id']})",
                f"- published: {paper['published']}",
                f"- primary category: {paper['primary_category']}",
                f"- tracked topics: {', '.join(paper.get('tracked_topics', [])) if paper.get('tracked_topics') else 'n/a'}",
                f"- model topics: {', '.join(paper['topics']) if paper['topics'] else 'n/a'}",
                "",
                "**中文摘要**",
                "",
                paper["summary_zh"] or "n/a",
                "",
                "**English Summary**",
                "",
                paper["summary_en"] or "n/a",
                "",
                "**背景 / Background**",
                "",
                f"- zh: {paper['background_zh'] or 'n/a'}",
                f"- en: {paper['background_en'] or 'n/a'}",
                "",
                "**问题 / Problem**",
                "",
                f"- zh: {paper['problem_zh'] or 'n/a'}",
                f"- en: {paper['problem_en'] or 'n/a'}",
                "",
                "**方法 / Method**",
                "",
                f"- zh: {paper['method_zh'] or 'n/a'}",
                f"- en: {paper['method_en'] or 'n/a'}",
                "",
                "**结果 / Findings**",
                "",
                f"- zh: {paper['findings_zh'] or 'n/a'}",
                f"- en: {paper['findings_en'] or 'n/a'}",
                "",
                "**局限 / Limitations**",
                "",
                f"- zh: {paper['limitations_zh'] or 'n/a'}",
                f"- en: {paper['limitations_en'] or 'n/a'}",
                "",
            ]
        )

    lines.extend(["## Paper Comparison Matrix", ""])
    lines.extend(
        [
            "| paper | category | tracked focus | model topics |",
            "| --- | --- | --- | --- |",
        ]
    )
    for paper in report_papers:
        focus = ", ".join(display_topics_for_paper(paper)[:3]) or "n/a"
        model_topics = ", ".join(paper.get("topics", [])[:3]) or "n/a"
        lines.append(
            f"| [{paper['title']}]({paper['entry_id']}) | {paper['primary_category']} | {focus} | {model_topics} |"
        )

    if comparison:
        lines.extend(["", "## Cross-Paper Comparison", ""])
        lines.extend(
            [
                "**Overview**",
                "",
                f"- zh: {comparison.overview.zh or 'n/a'}",
                f"- en: {comparison.overview.en or 'n/a'}",
                "",
                "**Common Ground**",
                "",
                f"- zh: {comparison.common_ground.zh or 'n/a'}",
                f"- en: {comparison.common_ground.en or 'n/a'}",
                "",
                "**Method Landscape**",
                "",
                f"- zh: {comparison.method_landscape.zh or 'n/a'}",
                f"- en: {comparison.method_landscape.en or 'n/a'}",
                "",
                "**Key Differences**",
                "",
            ]
        )
        if comparison.differences:
            for point in comparison.differences:
                label = point.label or "difference"
                lines.append(f"- {label}: zh: {point.zh or 'n/a'} | en: {point.en or 'n/a'}")
        else:
            lines.append("- n/a")

        lines.extend(["", "**Takeaways**", ""])
        if comparison.takeaways:
            for point in comparison.takeaways:
                label = point.label or "takeaway"
                lines.append(f"- {label}: zh: {point.zh or 'n/a'} | en: {point.en or 'n/a'}")
        else:
            lines.append("- n/a")

    return "\n".join(lines)


def write_report(content: str, reports_dir: Path, generated_at: datetime) -> Path:
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_path = reports_dir / f"daily_{generated_at.strftime('%Y%m%d_%H%M%S')}.md"
    report_path.write_text(content, encoding="utf-8")
    return report_path

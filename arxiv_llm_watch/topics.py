from __future__ import annotations

from collections import Counter
from datetime import datetime, timedelta, timezone
import re
from typing import Iterable, List, Sequence

from .models import TopicTrend

LLM_KEYWORDS = (
    "large language model",
    "large language models",
    "language model",
    "language models",
    "llm",
    "foundation model",
    "foundation models",
    "instruction tuning",
    "instruction-tuning",
    "fine-tuning",
    "post-training",
    "pretraining",
    "alignment",
    "rlhf",
    "dpo",
    "preference optimization",
    "prompting",
    "prompt engineering",
    "reasoning",
    "agent",
    "agents",
    "rag",
    "retrieval-augmented",
    "retrieval augmented",
    "context window",
    "multimodal llm",
    "sft",
    "inference scaling",
    "test-time scaling",
    "llm as a judge",
    "llm-as-a-judge",
    "hallucination",
    "factuality",
    "truthfulness",
    "jailbreak",
    "speculative decoding",
    "kv cache",
    "vision-language model",
    "mechanistic interpretability",
)

TRACKED_TOPIC_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "reasoning",
        (
            "reasoning",
            "chain of thought",
            "chain-of-thought",
            "test time scaling",
            "test-time scaling",
            "inference scaling",
            "deliberate decoding",
            "self consistency",
            "self-consistency",
        ),
    ),
    (
        "agents & tool use",
        (
            "agent",
            "agents",
            "tool use",
            "tool-use",
            "function calling",
            "browser agent",
            "computer use",
            "web agent",
            "multi agent",
            "multi-agent",
        ),
    ),
    (
        "rag & retrieval",
        (
            "rag",
            "retrieval augmented",
            "retrieval-augmented",
            "retrieval",
            "retriever",
            "reranker",
            "knowledge base",
            "search augmented",
        ),
    ),
    (
        "alignment & safety",
        (
            "llm alignment",
            "preference alignment",
            "safety",
            "harmlessness",
            "constitutional ai",
            "guardrail",
            "guardrails",
            "jailbreak",
            "red teaming",
            "red-teaming",
            "honesty",
            "deception",
            "lie detection",
            "bias mitigation",
        ),
    ),
    (
        "evaluation & llm as a judge",
        (
            "llm evaluation",
            "evaluator",
            "llm as a judge",
            "llm-as-a-judge",
            "judge model",
            "judge",
            "grading",
        ),
    ),
    (
        "hallucination & factuality",
        (
            "hallucination",
            "hallucinations",
            "factuality",
            "truthfulness",
            "fact checking",
            "fact-checking",
            "fact verification",
            "faithfulness",
            "groundedness",
            "factual",
        ),
    ),
    (
        "post training & preference optimization",
        (
            "instruction tuning",
            "instruction-tuning",
            "supervised fine tuning",
            "supervised fine-tuning",
            "fine tuning",
            "fine-tuning",
            "finetuning",
            "post-training",
            "post training",
            "rlhf",
            "dpo",
            "orpo",
            "ipo",
            "preference optimization",
            "preference tuning",
        ),
    ),
    (
        "training efficiency",
        (
            "pretraining",
            "pre-training",
            "training optimization",
            "memory efficient training",
            "memory-efficient training",
            "optimizer",
            "optimizer state",
            "gradient checkpointing",
            "parallelism",
            "orthogonal transformation",
        ),
    ),
    (
        "inference efficiency",
        (
            "inference optimization",
            "inference efficiency",
            "speculative decoding",
            "kv cache",
            "latency",
            "acceleration",
            "serving",
            "kernel optimization",
            "quantization",
            "pruning",
            "sparsity",
            "attention kernel",
        ),
    ),
    (
        "long context & memory",
        (
            "long context",
            "long-context",
            "context window",
            "memory augmented",
            "memory-augmented",
            "context compression",
            "persistent memory",
            "long horizon",
        ),
    ),
    (
        "multimodal llm",
        (
            "multimodal",
            "multi modal",
            "multi-modal",
            "vision language",
            "vision-language",
            "vision-language model",
            "vlm",
            "image text",
            "image-text",
            "video language",
            "cross modal",
            "cross-modal",
            "audio language",
        ),
    ),
    (
        "mechanistic interpretability",
        (
            "mechanistic interpretability",
            "interpretability",
            "probing",
            "activation",
            "attention sink",
            "representation",
            "internal mechanism",
            "internal belief",
            "neuron",
            "circuit",
            "feature attribution",
        ),
    ),
    (
        "synthetic data & distillation",
        (
            "synthetic data",
            "distillation",
            "self training",
            "self-training",
            "bootstrapping",
            "teacher model",
            "student model",
            "data augmentation",
        ),
    ),
    (
        "coding & program synthesis",
        (
            "code generation",
            "coding",
            "program synthesis",
            "software engineering",
            "repository level",
            "repository-level",
            "code completion",
            "bug fixing",
        ),
    ),
    (
        "benchmarks & datasets",
        (
            "benchmark",
            "benchmarking",
            "evaluation suite",
            "dataset release",
            "leaderboard",
            "shared task",
        ),
    ),
)
TRACKED_TOPIC_NAMES = tuple(name for name, _ in TRACKED_TOPIC_RULES)


def passes_keyword_gate(title: str, summary: str, categories: Sequence[str]) -> bool:
    haystack = normalize_topic_name(" ".join([title, summary, " ".join(categories)]))
    return any(_matches_keyword(haystack, keyword) for keyword in LLM_KEYWORDS)


def normalize_topic_name(topic: str) -> str:
    normalized = " ".join(topic.strip().lower().replace("-", " ").split())
    return normalized


def _matches_keyword(haystack: str, keyword: str) -> bool:
    normalized_keyword = normalize_topic_name(keyword)
    pattern = rf"(?<![a-z0-9]){re.escape(normalized_keyword)}(?![a-z0-9])"
    return re.search(pattern, haystack) is not None


def extract_tracked_topics(
    title: str = "",
    summary: str = "",
    categories: Sequence[str] | None = None,
    raw_topics: Sequence[str] | None = None,
) -> List[str]:
    haystack = normalize_topic_name(
        " ".join(
            [
                title or "",
                summary or "",
                " ".join(categories or []),
                " ".join(raw_topics or []),
            ]
        )
    )
    if not haystack:
        return []

    matched: List[str] = []
    for name, keywords in TRACKED_TOPIC_RULES:
        if any(_matches_keyword(haystack, keyword) for keyword in keywords):
            matched.append(name)
    return matched


def display_topics_for_paper(paper: dict) -> List[str]:
    tracked = [normalize_topic_name(topic) for topic in paper.get("tracked_topics", []) if normalize_topic_name(topic)]
    if tracked:
        return tracked
    return [
        normalize_topic_name(topic)
        for topic in paper.get("topics", [])
        if normalize_topic_name(topic)
    ]


def _parse_datetime(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def format_topic_momentum(current_count: int, baseline_count: int) -> str:
    if baseline_count <= 0:
        return f"New (0 -> {current_count})"
    growth = (current_count - baseline_count) / baseline_count
    sign = "+" if growth >= 0 else ""
    return f"{sign}{growth * 100:.0f}% ({baseline_count} -> {current_count})"


def compute_topic_trends(
    papers: Iterable[dict],
    recent_days: int,
    baseline_days: int,
    top_n: int,
    now: datetime | None = None,
) -> List[TopicTrend]:
    current_time = now or datetime.now(timezone.utc)
    recent_start = current_time - timedelta(days=recent_days)
    baseline_start = recent_start - timedelta(days=baseline_days)

    recent_counter: Counter[str] = Counter()
    baseline_counter: Counter[str] = Counter()

    for paper in papers:
        published = _parse_datetime(paper["published"])
        topics = set(display_topics_for_paper(paper))
        if not topics:
            continue
        if published >= recent_start:
            recent_counter.update(topics)
            continue
        if published >= baseline_start:
            baseline_counter.update(topics)

    topic_names = set(recent_counter) | set(baseline_counter)
    trends: List[TopicTrend] = []
    for name in topic_names:
        current_count = recent_counter.get(name, 0)
        baseline_count = baseline_counter.get(name, 0)
        if current_count == 0:
            continue
        growth = (current_count - baseline_count) / max(baseline_count, 1)
        trends.append(
            TopicTrend(
                name=name,
                current_count=current_count,
                baseline_count=baseline_count,
                growth=growth,
            )
        )

    trends.sort(key=lambda item: (item.current_count, item.growth, -item.baseline_count, item.name), reverse=True)
    return trends[:top_n]

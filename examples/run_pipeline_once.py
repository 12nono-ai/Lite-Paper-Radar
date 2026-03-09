from __future__ import annotations

import json
from pathlib import Path

from arxiv_llm_watch.config import AppConfig
from arxiv_llm_watch.pipeline import ArxivLLMWatchPipeline


def main() -> None:
    config = AppConfig.from_env(Path(".env"))
    pipeline = ArxivLLMWatchPipeline(config)
    summary = pipeline.run(
        lookback_days=3,
        max_results=60,
        analysis_limit=4,
        report_paper_limit=6,
        query_keywords=["llm", "reasoning"],
        run_source="example",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

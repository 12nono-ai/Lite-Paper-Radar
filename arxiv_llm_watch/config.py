from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional


def load_env_file(path: Path) -> None:
    if not path.exists():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def _read_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return int(value) if value else default


def _read_float(name: str, default: float) -> float:
    value = os.getenv(name)
    return float(value) if value else default


def _read_csv(name: str, default: str) -> List[str]:
    raw = os.getenv(name, default)
    return [item.strip() for item in raw.split(",") if item.strip()]


@dataclass
class AppConfig:
    ark_api_key: str
    ark_base_url: str
    ark_model: str
    arxiv_categories: List[str]
    arxiv_keywords: List[str]
    arxiv_max_results: int
    lookback_days: int
    topic_recent_days: int
    topic_baseline_days: int
    topic_limit: int
    report_paper_limit: int
    analysis_limit_per_run: int
    data_dir: Path
    reports_dir: Path
    db_path: Path
    llm_temperature: float

    @classmethod
    def from_env(cls, env_path: Optional[Path] = None) -> "AppConfig":
        load_env_file(env_path or Path(".env"))
        data_dir = Path(os.getenv("DATA_DIR", "data"))
        reports_dir = Path(os.getenv("REPORTS_DIR", "reports"))
        db_path = Path(os.getenv("DB_PATH", str(data_dir / "arxiv_llm_watch.db")))

        return cls(
            ark_api_key=os.getenv("ARK_API_KEY", ""),
            ark_base_url=os.getenv("ARK_BASE_URL", "https://ark.cn-beijing.volces.com/api/v3"),
            ark_model=os.getenv("ARK_MODEL", "doubao-seed-2-0-pro-260215"),
            arxiv_categories=_read_csv("ARXIV_CATEGORIES", "cs.CL,cs.AI,cs.LG,stat.ML"),
            arxiv_keywords=_read_csv("ARXIV_KEYWORDS", ""),
            arxiv_max_results=_read_int("ARXIV_MAX_RESULTS", 250),
            lookback_days=_read_int("LOOKBACK_DAYS", 2),
            topic_recent_days=_read_int("TOPIC_RECENT_DAYS", 7),
            topic_baseline_days=_read_int("TOPIC_BASELINE_DAYS", 7),
            topic_limit=_read_int("TOPIC_LIMIT", 8),
            report_paper_limit=_read_int("REPORT_PAPER_LIMIT", 12),
            analysis_limit_per_run=_read_int("ANALYSIS_LIMIT_PER_RUN", 6),
            data_dir=data_dir,
            reports_dir=reports_dir,
            db_path=db_path,
            llm_temperature=_read_float("LLM_TEMPERATURE", 0.2),
        )

    def ensure_directories(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

    def validate(self) -> None:
        missing: Dict[str, str] = {}
        if not self.ark_api_key:
            missing["ARK_API_KEY"] = "missing API key"
        if not self.ark_base_url:
            missing["ARK_BASE_URL"] = "missing base URL"
        if not self.ark_model:
            missing["ARK_MODEL"] = "missing model or endpoint ID"
        if missing:
            details = ", ".join(f"{key}: {reason}" for key, reason in missing.items())
            raise ValueError(f"Invalid configuration: {details}")

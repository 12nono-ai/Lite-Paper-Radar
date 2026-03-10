from __future__ import annotations

import os
import json
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


def _read_headers_json(name: str) -> Dict[str, str]:
    raw = os.getenv(name, "").strip()
    if not raw:
        return {}
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{name} must be valid JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{name} must be a JSON object")
    return {str(key): str(value) for key, value in payload.items()}


def normalize_llm_provider(value: str) -> str:
    normalized = (value or "").strip().lower().replace("-", "_")
    aliases = {
        "ark_sdk": "ark",
        "arkruntime": "ark",
        "openai": "openai_compatible",
        "openai_compat": "openai_compatible",
        "compat": "openai_compatible",
    }
    return aliases.get(normalized, normalized or "ark")


@dataclass
class AppConfig:
    llm_provider: str
    llm_api_key: str
    llm_base_url: str
    llm_model: str
    llm_api_path: str
    llm_headers: Dict[str, str]
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
        llm_provider = normalize_llm_provider(os.getenv("LLM_PROVIDER", "ark"))
        data_dir = Path(os.getenv("DATA_DIR", "data"))
        reports_dir = Path(os.getenv("REPORTS_DIR", "reports"))
        db_path = Path(os.getenv("DB_PATH", str(data_dir / "arxiv_llm_watch.db")))
        legacy_ark_base_url = os.getenv(
            "ARK_BASE_URL",
            "https://ark.cn-beijing.volces.com/api/v3" if llm_provider == "ark" else "",
        )
        legacy_ark_model = os.getenv("ARK_MODEL", "doubao-seed-2-0-pro-260215" if llm_provider == "ark" else "")

        return cls(
            llm_provider=llm_provider,
            llm_api_key=os.getenv("LLM_API_KEY", "") or os.getenv("ARK_API_KEY", ""),
            llm_base_url=os.getenv("LLM_BASE_URL", "") or legacy_ark_base_url,
            llm_model=os.getenv("LLM_MODEL", "") or legacy_ark_model,
            llm_api_path=os.getenv("LLM_API_PATH", "/chat/completions"),
            llm_headers=_read_headers_json("LLM_HEADERS_JSON"),
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
        if self.llm_provider not in {"ark", "openai_compatible"}:
            missing["LLM_PROVIDER"] = "must be one of: ark, openai_compatible"
        if not self.llm_api_key:
            missing["LLM_API_KEY"] = "missing API key (or legacy ARK_API_KEY)"
        if not self.llm_base_url:
            missing["LLM_BASE_URL"] = "missing base URL (or legacy ARK_BASE_URL)"
        if not self.llm_model:
            missing["LLM_MODEL"] = "missing model name (or legacy ARK_MODEL)"
        if missing:
            details = ", ".join(f"{key}: {reason}" for key, reason in missing.items())
            raise ValueError(f"Invalid configuration: {details}")

    @property
    def llm_provider_label(self) -> str:
        return "Ark" if self.llm_provider == "ark" else "OpenAI Compatible"

    @property
    def llm_endpoint(self) -> str:
        if not self.llm_api_path:
            return self.llm_base_url
        return f"{self.llm_base_url.rstrip('/')}/{self.llm_api_path.lstrip('/')}"

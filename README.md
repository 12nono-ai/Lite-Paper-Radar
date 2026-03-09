![LLM Paper Radar cover](assets/brand/cover.svg)

# LLM Paper Radar

[English](README.md) | [简体中文](README.zh-CN.md)

`LLM Paper Radar` is the public-facing name for the `arxiv-llm-watch` package. It is a lightweight daily paper monitor for LLM researchers: fetch fresh arXiv papers, generate bilingual Ark summaries, track short-term topic momentum, and browse everything in a local dashboard.

Useful repository metadata lives here:

- launch copy and GitHub topics: [docs/github-launch-kit.md](docs/github-launch-kit.md)
- architecture overview: [docs/architecture.md](docs/architecture.md)
- examples: [examples/README.md](examples/README.md)
- release checklist: [RELEASE_CHECKLIST.md](RELEASE_CHECKLIST.md)
- project logo: [assets/brand/logo.svg](assets/brand/logo.svg)
- contribution guide: [CONTRIBUTING.md](CONTRIBUTING.md)
- code of conduct: [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md)
- security policy: [SECURITY.md](SECURITY.md)
- changelog: [CHANGELOG.md](CHANGELOG.md)

## What it does

- fetches recent papers from configurable arXiv categories
- applies a low-cost keyword gate before hitting the model API
- asks Ark for bilingual structured analysis
- maps papers into a controlled LLM topic taxonomy
- computes hot topics over rolling time windows
- stores state in SQLite and renders daily reports
- serves a local dashboard for browsing, filtering, and manual review

The current implementation works on arXiv metadata only: `title`, `abstract`, `categories`, dates, and authors. It does not parse PDFs.

## Features

### Pipeline

- configurable arXiv category tracking
- per-run analysis cap to control API cost
- bilingual summary fields for `summary`, `background`, `problem`, `method`, `findings`, and `limitations`
- tracked-topic extraction for areas such as reasoning, agents, RAG, safety, efficiency, multimodality, and interpretability
- daily Markdown reports with hot topics, representative papers, and a cross-paper comparison section
- run history tracking for basic observability

### Dashboard

- dedicated `Overview`, `Papers`, and `Reports` views
- hot-topic overview optimized for daily reading
- paper explorer with keyword search, topic filters, sorting, pagination, and time-window filtering
- per-paper detail pages with related papers
- manual review controls such as star, ignore, manual topics, notes, and re-analyze
- report archive with run controls and execution history
- SVG charts for category share and topic heat

## Preview

The repository includes a social preview banner and a square logo under `assets/brand/`.

![Dashboard preview](assets/screenshots/dashboard-overview.svg)

You can use either of these for the GitHub social preview:

- vector banner: `assets/brand/cover.svg`
- generated banner: `assets/brand/cover-ark.jpg`

## Project layout

```text
arxiv_llm_watch/
  cli.py
  config.py
  dashboard.py
  fetcher.py
  llm_client.py
  models.py
  pipeline.py
  reporter.py
  storage.py
  topics.py
tests/
```

## Installation

Create a virtual environment and install the project:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
```

If you prefer not to install in editable mode, `pip install -r requirements.txt` also works for local use.

## Configuration

Copy `.env.example` to `.env` and fill in your Ark settings.

### Environment variables

- `ARK_API_KEY`: Ark API key
- `ARK_BASE_URL`: Ark base URL
- `ARK_MODEL`: Ark model name or endpoint ID
- `ARXIV_CATEGORIES`: comma-separated arXiv categories
- `ARXIV_KEYWORDS`: optional comma-separated fetch keywords, combined with categories at query time
- `ARXIV_MAX_RESULTS`: recent results to pull before date filtering
- `LOOKBACK_DAYS`: only keep papers published in this window
- `TOPIC_RECENT_DAYS`: recent trend window
- `TOPIC_BASELINE_DAYS`: baseline trend window immediately before the recent window
- `TOPIC_LIMIT`: number of hot topics in the report
- `REPORT_PAPER_LIMIT`: number of analyzed papers rendered in the daily report; this does not change fetch volume or analysis volume
- `ANALYSIS_LIMIT_PER_RUN`: maximum number of pending papers analyzed in one run
- `DATA_DIR`: output directory for local state
- `REPORTS_DIR`: output directory for generated reports
- `DB_PATH`: SQLite database path
- `LLM_TEMPERATURE`: model temperature for structured analysis

## Usage

Run one pipeline batch:

```bash
python3 -m arxiv_llm_watch.cli run
```

Override runtime settings when needed:

```bash
python3 -m arxiv_llm_watch.cli run --lookback-days 4 --max-results 200 --query-keywords "reasoning,agent" --analysis-limit 6
```

Start the local dashboard:

```bash
python3 -m arxiv_llm_watch.cli dashboard
```

Then open [http://127.0.0.1:8765](http://127.0.0.1:8765) in your browser.

If you install the package, you can also use the console entry point:

```bash
arxiv-llm-watch run
arxiv-llm-watch dashboard
llm-paper-radar run
llm-paper-radar dashboard
```

## Outputs

The pipeline writes local artifacts to ignored directories by default:

- SQLite state to `data/arxiv_llm_watch.db`
- timestamped daily reports to `reports/daily_YYYYMMDD_HHMMSS.md`

These files are meant for local runtime state and should not be committed.

## How it works

1. Fetch recent papers from configured arXiv categories.
2. Apply a lightweight keyword gate to skip obvious non-LLM papers.
3. Send remaining papers to Ark for structured JSON analysis.
4. Extract model topics and normalize them into tracked LLM topics.
5. Compute topic momentum across recent and baseline windows.
6. Generate a Markdown report and refresh dashboard state.

## Scheduling

Example cron entry for a daily run at 09:00:

```cron
0 9 * * * cd /path/to/arxiv-llm-watch && /path/to/arxiv-llm-watch/.venv/bin/python -m arxiv_llm_watch.cli run
```

## Open-source notes

- `.env`, local databases, and generated reports are ignored by default.
- `pyproject.toml` includes package metadata and a console entry point.
- GitHub Actions CI runs the unit test suite on push and pull request.
- issue templates, a PR template, and Dependabot config are included under `.github/`.
- `LICENSE`, `CODE_OF_CONDUCT.md`, `CONTRIBUTING.md`, `SECURITY.md`, and `CHANGELOG.md` are included for public repository use.

## Limitations

- The keyword gate is intentionally conservative on API cost, not authoritative on relevance.
- Topic trends become more reliable after several days of accumulated history.
- The current version does not parse full PDFs or citations.
- Ark text generation is wired through `chat.completions.create(...)`, which matches the current text-generation API surface used by this project.

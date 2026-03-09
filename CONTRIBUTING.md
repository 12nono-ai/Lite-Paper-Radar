# Contributing

Thanks for contributing to LLM Paper Radar.

## Development setup

1. Create and activate a virtual environment.
2. Install the project in editable mode:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
```

3. Copy `.env.example` to `.env` and fill in your Ark credentials if you want to run the live pipeline.

## Local checks

Run the test suite:

```bash
python3 -m unittest discover -s tests
```

Start the pipeline:

```bash
python3 -m arxiv_llm_watch.cli run
```

Start the dashboard:

```bash
python3 -m arxiv_llm_watch.cli dashboard
```

## Contribution guidelines

- Keep changes small and scoped.
- Add or update tests for behavior changes.
- Do not commit `.env`, local SQLite databases, generated reports, or API secrets.
- Prefer configuration over hardcoded local paths.
- Update `README.md` when behavior or setup changes.
- Update `CHANGELOG.md` for user-visible changes.
- Use the issue templates for bugs and feature requests when possible.

## Pull requests

Please include:

- what changed
- why it changed
- how you validated it
- any configuration or migration notes

The repository includes a pull request template under `.github/pull_request_template.md`.

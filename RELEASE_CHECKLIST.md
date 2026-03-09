# Release Checklist

Use this checklist before making the repository public or cutting the first tagged release.

## Security and cleanup

- Rotate any Ark or other API keys that were ever pasted into terminals, chats, or screenshots.
- Confirm `.env`, local databases, generated reports, and `.venv` are ignored and not staged.
- Review commit history for accidental secrets or local machine paths.
- Remove or sanitize any real user data in `data/` and `reports/`.

## Repository metadata

- Set the repository display name you want to use publicly.
- Copy the suggested GitHub description and topics from [docs/github-launch-kit.md](docs/github-launch-kit.md).
- Upload a social preview image using [assets/brand/cover.svg](assets/brand/cover.svg) or [assets/brand/cover-ark.jpg](assets/brand/cover-ark.jpg).
- Add a short repository website or demo link if you host one later.

## Documentation

- Make sure [README.md](README.md) matches current behavior and screenshots.
- Confirm installation, configuration, and usage examples run from a clean checkout.
- Check that environment variable names in [README.md](README.md) and [.env.example](.env.example) are aligned.
- Review [CONTRIBUTING.md](CONTRIBUTING.md) and [SECURITY.md](SECURITY.md) for clarity.

## Packaging and CI

- Verify `pip install -e .` works in a new virtual environment.
- Run `python3 -m unittest discover -s tests`.
- Confirm GitHub Actions succeeds on the default branch.
- Decide whether to publish to PyPI now or keep installation source-only.

## Product polish

- Confirm the dashboard loads cleanly on a fresh database.
- Run one real batch and inspect the report output.
- Review generated topic names and manual-review controls for public-facing quality.
- Capture and update the dashboard screenshot if the UI changed materially.

## Repository settings

- Enable branch protection for `main`.
- Enable secret scanning and dependency alerts in GitHub security settings.
- Add issue templates and a pull request template if you want external contributions immediately.
- Decide whether Discussions should be enabled for roadmap and feedback.

## First release

- Create a `v0.1.0` tag after the public repository passes CI.
- Draft release notes with highlights, setup instructions, and known limitations.
- Include one screenshot and one report example in the release description.

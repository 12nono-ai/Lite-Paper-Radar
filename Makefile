PYTHON ?= python3
VENV_PYTHON ?= .venv/bin/python
PIP ?= .venv/bin/pip

.PHONY: setup init-ark init-openai doctor test run dashboard period-report

setup:
	$(PYTHON) -m venv .venv
	$(PIP) install -U pip
	$(PIP) install -e .

init-ark:
	$(VENV_PYTHON) -m arxiv_llm_watch.cli init --provider ark

init-openai:
	$(VENV_PYTHON) -m arxiv_llm_watch.cli init --provider openai_compatible

doctor:
	$(VENV_PYTHON) -m arxiv_llm_watch.cli doctor

test:
	$(VENV_PYTHON) -m unittest discover -s tests

run:
	$(VENV_PYTHON) -m arxiv_llm_watch.cli run

dashboard:
	$(VENV_PYTHON) -m arxiv_llm_watch.cli dashboard

period-report:
	$(VENV_PYTHON) -m arxiv_llm_watch.cli period-report --days 7

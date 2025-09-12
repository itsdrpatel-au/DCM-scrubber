PY=python

.PHONY: setup lint test run gui cli

setup:
	uv venv || true
	uv pip install -e .[dev]

lint:
	ruff check .
	black --check .
	mypy dicom_deid_gui

test:
	pytest -q

run:
	$(PY) -m dicom_deid_gui

gui:
	$(PY) -m dicom_deid_gui

cli:
	dicom-deid --help



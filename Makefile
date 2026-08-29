PYTHON ?= python

.PHONY: gateway benchmark train export
gateway:
	uvicorn gateway.api:app --reload
benchmark:
	$(PYTHON) benchmark/run_model_matrix.py
train:
	$(PYTHON) router/train.py
export:
	$(PYTHON) scripts/export_results.py

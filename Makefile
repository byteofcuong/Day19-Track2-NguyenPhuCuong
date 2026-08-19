## Day 19 — Vector Store + Feature Store lab.
## Two paths: lightweight (default, no Docker) and full Docker.

VENV     := .venv
# venv console scripts live in Scripts/ on Windows and bin/ on macOS/Linux.
# Detect rather than assume, so `make test` / `make notebooks` work on all three.
VENV_BIN := $(if $(wildcard $(VENV)/Scripts/python.exe),$(VENV)/Scripts,$(VENV)/bin)
PY       := $(VENV_BIN)/python
PIP      := $(VENV_BIN)/pip
JUPYTER  := $(VENV_BIN)/jupyter
JUPYTEXT := $(VENV_BIN)/jupytext
UVICORN  := $(VENV_BIN)/uvicorn
PYTEST   := $(VENV_BIN)/pytest

.DEFAULT_GOAL := help

help: ## Show this help
	@awk 'BEGIN {FS = ":.*##"; printf "\nUsage:\n  make \033[36m<target>\033[0m\n\nLightweight path (default):\n"} \
	      /^[a-zA-Z_-]+:.*?##/ { printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2 }' $(MAKEFILE_LIST)

# ─────────────────────────────────────────────────────────────
# Lightweight path (default) — no Docker, in-process Qdrant
# ─────────────────────────────────────────────────────────────

setup-lite: ## [lite] Create venv + install + seed corpus + smoke test
	@bash setup-lite.sh

verify-lite: ## [lite] 5-second smoke test (Qdrant memory + BM25 + Feast SQLite)
	@$(PY) scripts/verify_lite.py

seed: ## [both] (Re)generate data/corpus_vn.jsonl + data/golden_set.jsonl
	@$(PY) scripts/seed_corpus.py

api: ## [lite] Start FastAPI /search on http://localhost:8000
	@$(UVICORN) app.main:app --reload --port 8000

lab: ## [lite] Open Jupyter Lab on http://localhost:8888
	@$(JUPYTEXT) --to notebook --update notebooks/[0-9]*.py 2>/dev/null || true
	@$(JUPYTER) lab --notebook-dir=notebooks --ServerApp.token='' --no-browser

benchmark: ## [both] Precision@10 (keyword/semantic/hybrid) + P99 latency table
	@$(PY) scripts/benchmark.py

test: ## [both] Run pytest (app + scripts)
	@$(PYTEST) -q

gen-advanced: ## [both] Generate data for the advanced missions (NB6 + NB8)
	@$(PY) scripts/gen_agent_queries.py
	@$(PY) scripts/gen_spend.py

notebooks: ## [both] Execute ALL notebooks headless (what the grader runs)
	@$(JUPYTEXT) --to notebook --update notebooks/[0-9]*.py >/dev/null 2>&1 || true
	@for nb in notebooks/[0-9]*.ipynb; do \
		printf '%-42s' "$$nb"; \
		PATH="$(CURDIR)/$(VENV_BIN):$$PATH" $(JUPYTER) nbconvert --to notebook \
			--execute --inplace "$$nb" --ExecutePreprocessor.timeout=900 \
			>/dev/null 2>&1 && echo PASS || echo FAIL; \
	done

screenshots: ## [both] Render executed-notebook output into submission/screenshots/
	@$(PY) scripts/make_screenshots.py

clean-lite: ## [lite] Wipe venv + data + Feast registry
	rm -rf $(VENV) data/corpus_vn.jsonl data/golden_set.jsonl data/qdrant_storage \
	       data/agent_queries.jsonl \
	       app/feast_repo/data app/feast_repo/registry.db app/feast_repo/online_store.db \
	       app/feast_repo_ondemand/data app/feast_repo_ondemand/registry.db \
	       app/feast_repo_ondemand/online_store.db \
	       notebooks/*.ipynb notebooks/.ipynb_checkpoints

# ─────────────────────────────────────────────────────────────
# Docker path (full stack: Qdrant + Redis + Postgres)
# ─────────────────────────────────────────────────────────────

setup-docker: ## [docker] Bring up Docker stack + venv + seed + smoke test
	@bash setup-docker.sh

runtime-check: ## [docker] Report docker / podman / apple-container versions + capabilities
	@bash scripts/runtime-check.sh

container-up: ## [apple] Start the 3 services with Apple container (no compose)
	@bash scripts/container-up.sh

container-down: ## [apple] Stop the Apple container stack (add ARGS=--wipe to drop volumes)
	@bash scripts/container-down.sh $(ARGS)

verify-docker: ## [docker] Verify all 3 services reachable + Feast wired
	@$(PY) scripts/verify_docker.py

docker-up: ## [docker] Just bring services up (no venv changes)
	docker compose up -d

docker-down: ## [docker] Stop services (data persists)
	docker compose down

docker-clean: ## [docker] Stop AND wipe Qdrant + Redis + Postgres volumes
	docker compose down -v

.PHONY: help setup-lite verify-lite seed gen-advanced notebooks screenshots api lab benchmark test clean-lite \
        setup-docker verify-docker docker-up docker-down docker-clean \
        runtime-check container-up container-down

.PHONY: bench-native bench-docker-build bench-docker test clean-docker-cache \
        dev dev-docker-build dev-docker serve fetch-models measure release

# ---------------------------------------------------------------------------
# Two-mode development (docs/PLAN.md). Mode A (native) is for fast iteration
# on Apple Silicon MPS; its numbers are NOT trustworthy for decisions — see
# docs/03-decisions.md D3/D8 for a concrete example of why (MPS correctness
# warning; allocator-history-dependent RSS deltas). Mode B (this container,
# --cpus=2 --memory=16g) is the one whose numbers go in models.yaml,
# README, or any capacity decision.
# ---------------------------------------------------------------------------

# Fast, exploratory. Uses MPS if available. Never quote these numbers.
bench-native:
	uv run python -m attnlab.bench

# Build the Mode B benchmark image. Deliberately built WITHOUT --platform
# so it's native linux/arm64 on Apple Silicon — no Rosetta emulation.
bench-docker-build:
	docker build -f docker/Dockerfile.bench -t attnlab-bench .

# Run the full benchmark matrix inside the constrained container.
#
#   --memory=6g --memory-swap=6g   D1 originally targeted 16GB (HF free
#                                   CPU Basic), but THIS machine's Docker
#                                   Desktop VM is fixed at 7.7GB total
#                                   RAM (docker info --format
#                                   '{{.MemTotal}}') — a systemwide
#                                   setting, deliberately left unchanged
#                                   (see docs/03-decisions.md D1
#                                   amendment). 6GB fits inside that with
#                                   ~1.5GB headroom for the VM/kernel
#                                   itself, and happens to match
#                                   MI_RAM_BUDGET_GB already — this was
#                                   always the more realistic number to
#                                   test against than the raw 16GB spec.
#   --cpus=2                        CFS quota; settings.py + this
#                                    Dockerfile's ENV force torch to
#                                    actually respect it (see comments
#                                    in Dockerfile.bench)
#   named volume for HF cache       so the ~4.8GB of model downloads
#                                    happen once, not once per run
#   bind mount for bench_results    so RESULTS.md lands in the repo,
#                                    not trapped inside the container
bench-docker: bench-docker-build
	docker volume create attnlab-hf-cache >/dev/null
	docker run --rm \
		--platform linux/arm64 \
		--memory=6g --memory-swap=6g --cpus=2 \
		-v attnlab-hf-cache:/hf-cache \
		-v $(PWD)/bench_results:/app/bench_results \
		attnlab-bench

test:
	uv run pytest -v

clean-docker-cache:
	docker volume rm attnlab-hf-cache

# ---------------------------------------------------------------------------
# Stage 0b+ API server
# ---------------------------------------------------------------------------

# Fast local iteration (Mode A). CPU, like the server: on Apple Silicon it
# measured faster than MPS for every model here, and exact (D16). Run with
# MI_DEVICE=mps to compare. See the Mode A/B caveat at the top of this file
# before trusting any number it produces.
dev:
	MallocLargeCache=0 uv run uvicorn attnlab.api.app:app --reload --port 8000

dev-docker-build:
	docker build -f docker/Dockerfile.api -t attnlab-api .

# Mode B: the real 2-CPU/6GB envelope (see the bench-docker comment above
# for why 6GB, not the original 16GB target). Named volume for the HF
# cache is shared with bench-docker's, so models already benchmarked
# don't get downloaded again.
dev-docker: dev-docker-build
	docker volume create attnlab-hf-cache >/dev/null
	docker run --rm \
		--platform linux/arm64 \
		--memory=6g --memory-swap=6g --cpus=2 \
		-p 8000:8000 \
		-v attnlab-hf-cache:/hf-cache \
		attnlab-api

# ---------------------------------------------------------------------------
# Serving (docs/04-self-hosting.md, docs/06-releasing.md)
# ---------------------------------------------------------------------------

# The production configuration, on this machine: deploy/server.env, the
# built frontend, one origin at http://127.0.0.1:8000. The closest thing to
# the server you can run locally. Needs the models on disk (make fetch-models).
serve:
	cd web && npm run build
	set -a && . deploy/server.env && set +a && \
		uv run uvicorn attnlab.api.app:app --host 127.0.0.1 --port 8000 --workers 1

# Download every model and tokenizer the server needs (only what fits the
# server's budget; nothing is loaded into memory).
fetch-models:
	MI_RAM_BUDGET_GB=7 uv run python scripts/fetch_models.py

# Measure a model before adding it to models.yaml: make measure MODEL=gpt2-small
measure:
	MallocLargeCache=0 MI_THREADS=6 uv run python scripts/measure_model.py $(MODEL)

# Bump the version, tag, and print the push/deploy commands: make release V=0.2.0
release:
	scripts/release.sh $(V)

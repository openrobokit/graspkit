
# sim needs no torch; for real, CUDA=real-cu128 (or real-cu126) adds SAM2 + nvdiffrast
CUDA   ?=
EXTRAS ?= $(if $(CUDA),--extra $(CUDA)) --extra sim --extra dev
SRC := src scripts

.PHONY: ci install format format-check lint check demo-grasp demo-teleop

ci: format-check lint check

install:
	uv sync $(EXTRAS)

format:
	uvx ruff@0.15.22 format $(SRC)

format-check:
	uvx ruff@0.15.22 format --check $(SRC)

lint:
	uvx ruff@0.15.22 check --fix $(SRC)

check:
	uv run $(EXTRAS) mypy src/hug

demo-grasp:
	uv run --extra sim python scripts/sim/grasp_demo.py

demo-teleop:
	uv run --extra sim python scripts/sim/teleop.py

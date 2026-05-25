.PHONY: build install dev test lint clean docker-build

IMAGE_NAME ?= rbg-planner
IMAGE_TAG ?= latest

install:
	pip install -e .

dev:
	pip install -e ".[dev]"

test:
	pytest tests/ -v

lint:
	python -m py_compile rbg_planner/*.py rbg_planner/utils/*.py

clean:
	rm -rf build/ dist/ *.egg-info __pycache__
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

docker-build:
	docker build -t $(IMAGE_NAME):$(IMAGE_TAG) .

run:
	python -m rbg_planner.main

run-dry:
	python -m rbg_planner.main --no-operation

.PHONY: build generate manifests test lint clean docker-build docker-build-planner docker-build-profiler

CONTROLLER_GEN ?= $(shell which controller-gen 2>/dev/null || echo $(shell go env GOPATH)/bin/controller-gen)
OPERATOR_IMG ?= rbg-planner-operator:latest
PLANNER_IMG ?= rbg-planner:latest
PROFILER_IMG ?= rbg-profiler:latest

##@ General

help: ## Display this help.
	@awk 'BEGIN {FS = ":.*##"; printf "\nUsage:\n  make \033[36m<target>\033[0m\n"} /^[a-zA-Z_0-9-]+:.*?##/ { printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2 }' $(MAKEFILE_LIST)

##@ Development

generate: ## Generate deepcopy methods.
	$(CONTROLLER_GEN) object paths=./api/...

manifests: ## Generate CRD manifests.
	$(CONTROLLER_GEN) crd:allowDangerousTypes=true paths=./api/... output:crd:dir=config/crd
	$(CONTROLLER_GEN) rbac:roleName=rbg-planner-operator paths=./internal/... output:rbac:dir=config/rbac

fmt: ## Run go fmt.
	go fmt ./...

vet: ## Run go vet.
	go vet ./...

lint: vet ## Run linters.
	go vet ./...

test: ## Run Go tests.
	go test ./... -v

test-python: ## Run Python planner tests.
	cd python/planner && pip install -e ".[dev]" && pytest tests/ -v

##@ Build

build: generate fmt vet ## Build operator binary.
	go build -o bin/manager cmd/main.go

run: generate fmt vet ## Run operator locally.
	go run cmd/main.go

##@ Docker

docker-build: ## Build operator Docker image.
	docker build -t $(OPERATOR_IMG) .

docker-build-planner: ## Build planner Docker image.
	docker build -t $(PLANNER_IMG) -f python/planner/Dockerfile python/planner/

docker-build-profiler: ## Build profiler Docker image.
	docker build -t $(PROFILER_IMG) -f python/profiler/Dockerfile python/profiler/

docker-build-all: docker-build docker-build-planner docker-build-profiler ## Build all Docker images.

##@ Deployment

install: manifests ## Install CRDs into cluster.
	kubectl apply -f config/crd/

uninstall: ## Uninstall CRDs from cluster.
	kubectl delete -f config/crd/

deploy: manifests ## Deploy operator to cluster (requires kustomize or manual apply).
	@echo "Apply CRDs and operator manifests to your cluster"
	kubectl apply -f config/crd/

##@ Cleanup

clean: ## Clean build artifacts.
	rm -rf bin/

.PHONY: build generate manifests test lint clean docker-build docker-build-planner docker-build-profiler controller-gen golangci-lint fmt-verify ci-lint lint-python fmt-python test-go test-python update-helm helm-lint verify

##@ Build Tools

LOCALBIN ?= $(shell pwd)/bin
CONTROLLER_TOOLS_VERSION ?= v0.21.0
GOLANGCI_LINT_VERSION ?= v2.1.4
YAML_PROCESSOR_LOG_LEVEL ?= info

CONTROLLER_GEN ?= $(LOCALBIN)/controller-gen
GOLANGCI_LINT ?= $(LOCALBIN)/golangci-lint

OPERATOR_IMG ?= rbg-planner-operator:latest
PLANNER_IMG ?= rbg-planner:latest
PROFILER_IMG ?= rbg-profiler:latest

define go-install-tool
@[ -f "$(1)" ] || { set -e; \
mkdir -p $(LOCALBIN); \
GOBIN=$(LOCALBIN) go install "$(2)"; \
}
endef

controller-gen: ## Download controller-gen locally if necessary.
	$(call go-install-tool,$(CONTROLLER_GEN),sigs.k8s.io/controller-tools/cmd/controller-gen@$(CONTROLLER_TOOLS_VERSION))

golangci-lint: ## Download golangci-lint locally if necessary.
	$(call go-install-tool,$(GOLANGCI_LINT),github.com/golangci/golangci-lint/v2/cmd/golangci-lint@$(GOLANGCI_LINT_VERSION))

##@ General

help: ## Display this help.
	@awk 'BEGIN {FS = ":.*##"; printf "\nUsage:\n  make \033[36m<target>\033[0m\n"} /^[a-zA-Z_0-9-]+:.*?##/ { printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2 }' $(MAKEFILE_LIST)

##@ Development

generate: controller-gen ## Generate deepcopy methods.
	$(CONTROLLER_GEN) object paths=./api/...

manifests: controller-gen ## Generate CRD manifests.
	$(CONTROLLER_GEN) crd:allowDangerousTypes=true paths=./api/... output:crd:dir=config/crd
	$(CONTROLLER_GEN) rbac:roleName=rbg-planner-operator paths=./internal/... output:rbac:dir=config/rbac
	cp -f ./config/crd/inference-extension.rolebasedgroup.io_autoscalers.yaml ./charts/rbg-planner/crds/

update-helm: manifests ## Sync generated manifests to Helm chart.
	GOFLAGS=-mod=mod go run -modfile=hack/tools/yaml-processor/go.mod \
	  sigs.k8s.io/kueue/hack/tools/yaml-processor \
	  -zap-log-level=$(YAML_PROCESSOR_LOG_LEVEL) hack/processing-plan.yaml

helm-lint: ## Lint the Helm chart.
	helm lint charts/rbg-planner/ --set prometheus.endpoint=http://test:9090

fmt: ## Run go fmt.
	go fmt ./...

vet: ## Run go vet.
	go vet ./...

fmt-verify: ## Verify go fmt.
	@files=$$(gofmt -l .); if [ -n "$$files" ]; then echo "Unformatted files:"; echo "$$files"; exit 1; fi

ci-lint: golangci-lint ## Run golangci-lint.
	$(GOLANGCI_LINT) run --timeout 15m0s

lint-python: ## Run Python linter (ruff).
	cd python/planner && pip install -e ".[dev]" && ruff check . && ruff format --check .

fmt-python: ## Format Python code (ruff).
	cd python/planner && ruff format .

lint: ci-lint lint-python ## Run all linters.

test-go: ## Run Go tests with coverage.
	go test ./... -v -coverprofile=cover.out

test-python: ## Run Python planner tests with coverage.
	cd python/planner && pip install -e ".[dev]" && pytest tests/ -v --cov=rbg_planner

test: test-go test-python ## Run all tests.

verify: manifests generate update-helm ## Verify no drift in generated artifacts.
	git --no-pager diff --exit-code config api charts

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

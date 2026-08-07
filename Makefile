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

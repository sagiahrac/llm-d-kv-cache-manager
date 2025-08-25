SHELL := /usr/bin/env bash

# Defaults
PROJECT_NAME ?= llm-d-kv-cache-manager
DEV_VERSION ?= 0.0.1
PROD_VERSION ?= 0.0.0
IMAGE_TAG_BASE ?= ghcr.io/llm-d/$(PROJECT_NAME)
IMG = $(IMAGE_TAG_BASE):$(DEV_VERSION)
NAMESPACE ?= hc4ai-operator

TARGETOS ?= $(shell go env GOOS)
TARGETARCH ?= $(shell go env GOARCH)
UNAME_S := $(shell uname -s)

TOOLS_DIR := $(shell pwd)/hack/tools
CONTAINER_TOOL := $(shell { command -v docker >/dev/null 2>&1 && echo docker; } || { command -v podman >/dev/null 2>&1 && echo podman; } || echo "")
BUILDER := $(shell command -v buildah >/dev/null 2>&1 && echo buildah || echo $(CONTAINER_TOOL))

# go source files
SRC = $(shell find . -type f -name '*.go')

.PHONY: help
help: ## Print help
	@awk 'BEGIN {FS = ":.*##"; printf "\nUsage:\n  make \033[36m<target>\033[0m\n"} /^[a-zA-Z_0-9-]+:.*?##/ { printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2 } /^##@/ { printf "\n\033[1m%s\033[0m\n", substr($$0, 5) } ' $(MAKEFILE_LIST)

##@ Tokenizer & Linking

TOKENIZER_LIB = lib/libtokenizers.a

# Extract RELEASE_VERSION from Dockerfile
TOKENIZER_VERSION := $(shell grep '^ARG RELEASE_VERSION=' Dockerfile | cut -d'=' -f2)

.PHONY: download-tokenizer
download-tokenizer: $(TOKENIZER_LIB)
$(TOKENIZER_LIB):
	## Download the HuggingFace tokenizer bindings.
	@echo "Downloading HuggingFace tokenizer bindings for version $(TOKENIZER_VERSION)..."
	mkdir -p lib
	curl -L https://github.com/daulet/tokenizers/releases/download/$(TOKENIZER_VERSION)/libtokenizers.$(TARGETOS)-$(TARGETARCH).tar.gz | tar -xz -C lib
	ranlib lib/*.a

##@ Python Configuration

PYTHON_VERSION := 3.12
VENV_DIR := $(shell pwd)/build/venv
VENV_BIN := $(VENV_DIR)/bin

# Attempt to find Python 3.9 executable.
PYTHON_EXE := $(shell command -v python$(PYTHON_VERSION) || command -v python3)

# Unified Python configuration detection. This block runs once.
# It prioritizes python-config, then pkg-config, for reliability.
ifeq ($(UNAME_S),Darwin)
    # macOS: Find Homebrew's python-config script for the most reliable flags.
    BREW_PREFIX := $(shell command -v brew >/dev/null 2>&1 && brew --prefix python@$(PYTHON_VERSION) 2>/dev/null)
    PYTHON_CONFIG := $(BREW_PREFIX)/bin/python$(PYTHON_VERSION)-config
    ifneq ($(shell $(PYTHON_CONFIG) --cflags 2>/dev/null),)
        PYTHON_CFLAGS := $(shell $(PYTHON_CONFIG) --cflags)
        # Use --ldflags --embed to get all necessary flags for linking
        PYTHON_LDFLAGS := $(shell $(PYTHON_CONFIG) --ldflags --embed)
        PYTHON_LIBS :=
    else
        $(error "Could not execute 'python$(PYTHON_VERSION)-config' from Homebrew. Please ensure Python is installed correctly with: 'brew install python@$(PYTHON_VERSION)'")
    endif
else ifeq ($(UNAME_S),Linux)
    # Linux: Use standard system tools to find flags.
    PYTHON_CONFIG := $(shell command -v python$(PYTHON_VERSION)-config || command -v python3-config)
    ifneq ($(shell $(PYTHON_CONFIG) --cflags 2>/dev/null),)
		# Use python-config if available and correct
        PYTHON_CFLAGS := $(shell $(PYTHON_CONFIG) --cflags)
        PYTHON_LDFLAGS := $(shell $(PYTHON_CONFIG) --ldflags --embed)
        PYTHON_LIBS :=
    else ifneq ($(shell pkg-config --cflags python-$(PYTHON_VERSION) 2>/dev/null),)
        # Fallback to pkg-config
        PYTHON_CFLAGS := $(shell pkg-config --cflags python-$(PYTHON_VERSION))
        PYTHON_LDFLAGS := $(shell pkg-config --libs python-$(PYTHON_VERSION))
        PYTHON_LIBS :=
    else
        $(error "Python $(PYTHON_VERSION) development headers not found. Please install with: 'sudo apt install python$(PYTHON_VERSION)-dev' or 'sudo dnf install python$(PYTHON_VERSION)-devel'")
    endif
else
    $(error "Unsupported OS: $(UNAME_S)")
endif

# Final CGO flags with all dependencies
CGO_CFLAGS_FINAL := $(PYTHON_CFLAGS) -Ilib
CGO_LDFLAGS_FINAL := $(PYTHON_LDFLAGS) $(PYTHON_LIBS) -Llib -ltokenizers -ldl -lm

.PHONY: detect-python
detect-python: ## Detects Python and prints the configuration.
	@printf "\033[33;1m==== Python Configuration ====\033[0m\n"
	@if [ -z "$(PYTHON_EXE)" ]; then \
		echo "ERROR: Python 3 not found in PATH."; \
		exit 1; \
	fi
	@# Verify the version of the found python executable using its exit code
	@if ! $(PYTHON_EXE) -c "import sys; sys.exit(0 if sys.version_info[:2] == ($(shell echo $(PYTHON_VERSION) | cut -d. -f1), $(shell echo $(PYTHON_VERSION) | cut -d. -f2)) else 1)"; then \
		echo "ERROR: Found Python at '$(PYTHON_EXE)' but it is not version $(PYTHON_VERSION)."; \
		echo "Please ensure 'python$(PYTHON_VERSION)' or a compatible 'python3' is in your PATH."; \
		exit 1; \
	fi
	@echo "Python executable: $(PYTHON_EXE) ($$($(PYTHON_EXE) --version))"
	@echo "Python CFLAGS:     $(PYTHON_CFLAGS)"
	@echo "Python LDFLAGS:    $(PYTHON_LDFLAGS)"
	@if [ -z "$(PYTHON_CFLAGS)" ]; then \
		echo "ERROR: Python development headers not found. See installation instructions above."; \
		exit 1; \
	fi
	@printf "\033[33;1m==============================\033[0m\n"

.PHONY: install-python-deps
install-python-deps: detect-python ## Sets up the Python virtual environment and installs dependencies.
	@printf "\033[33;1m==== Setting up Python virtual environment in $(VENV_DIR) ====\033[0m\n"
	@if [ ! -f "$(VENV_BIN)/pip" ]; then \
		echo "Creating virtual environment..."; \
		$(PYTHON_EXE) -m venv $(VENV_DIR) || { \
			echo "ERROR: Failed to create virtual environment."; \
			echo "Your Python installation may be missing the 'venv' module."; \
			echo "Try: 'sudo apt install python$(PYTHON_VERSION)-venv' or 'sudo dnf install python$(PYTHON_VERSION)-devel'"; \
			exit 1; \
		}; \
	fi
	@echo "Upgrading pip and installing dependencies..."
	@$(VENV_BIN)/pip install --upgrade pip
	@$(VENV_BIN)/pip install -q -r pkg/preprocessing/chat_completions/requirements.txt
	@echo "Verifying transformers installation..."
	@$(VENV_BIN)/python -c "import transformers; print('✅ Transformers version ' + transformers.__version__ + ' installed.')" || { \
		echo "ERROR: transformers library not properly installed in venv."; \
		exit 1; \
	}

##@ Precommit code checks
.PHONY: precommit lint tidy-go copr-fix
precommit: tidy-go lint copr-fix

tidy-go:
	@echo "Tidying up go.mod and go.sum..."
	@go mod tidy

lint:
	@echo "==== Running linting ===="
	@golangci-lint run

copr-fix:
	@echo "Adding copyright headers..."
	@docker run -i --rm -v $(shell pwd):/github/workspace apache/skywalking-eyes header fix

##@ Development

# Common environment variables for Go tests and builds
export CGO_ENABLED=1
export CGO_CFLAGS=$(CGO_CFLAGS_FINAL)
export CGO_LDFLAGS=$(CGO_LDFLAGS_FINAL)
export PYTHONPATH=$(shell pwd)/pkg/preprocessing/chat_completions:$(VENV_DIR)/lib/python$(PYTHON_VERSION)/site-packages

.PHONY: test
test: unit-test e2e-test ## Run all tests

.PHONY: unit-test
unit-test: download-tokenizer install-python-deps download-zmq ## Run unit tests
	@printf "\033[33;1m==== Running unit tests ====\033[0m\n"
	@go test -v ./pkg/...

.PHONY: e2e-test
e2e-test: download-tokenizer install-python-deps download-zmq ## Run end-to-end tests
	@printf "\033[33;1m==== Running e2e tests ====\033[0m\n"
	@go test -v ./tests/...

.PHONY: bench
bench: download-tokenizer install-python-deps download-zmq ## Run benchmarks
	@printf "\033[33;1m==== Running chat template benchmarks ====\033[0m\n"
	@go test -bench=. -benchmem ./pkg/preprocessing/chat_completions/

.PHONY: run
run: build ## Run the application locally
	@printf "\033[33;1m==== Running application ====\033[0m\n"
	@./bin/$(PROJECT_NAME)

##@ Build

.PHONY: build
build: check-go download-tokenizer install-python-deps download-zmq ## Build the application binary
	@printf "\033[33;1m==== Building application binary ====\033[0m\n"
	@go build -o bin/$(PROJECT_NAME) examples/kv_events/online/main.go


.PHONY:	image-build
image-build: check-container-tool load-version-json ## Build Docker image
	@printf "\033[33;1m==== Building Docker image $(IMG) ====\033[0m\n"
	$(CONTAINER_TOOL) build \
		--platform $(TARGETOS)/$(TARGETARCH) \
		--build-arg TARGETOS=$(TARGETOS) \
		--build-arg TARGETARCH=$(TARGETARCH) \
		-t $(IMG) .
.PHONY: image-push
image-push: check-container-tool load-version-json ## Push Docker image $(IMG) to registry
	@printf "\033[33;1m==== Pushing Docker image $(IMG) ====\033[0m\n"
	$(CONTAINER_TOOL) push $(IMG)

##@ Install/Uninstall Targets

# Default install/uninstall (Docker)
install: install-docker ## Default install using Docker
	@echo "Default Docker install complete."
uninstall: uninstall-docker ## Default uninstall using Docker
	@echo "Default Docker uninstall complete."
### Docker Targets

.PHONY: install-docker
install-docker: check-container-tool ## Install app using $(CONTAINER_TOOL)
	@echo "Starting container with $(CONTAINER_TOOL)..."
	$(CONTAINER_TOOL) run -d --name $(PROJECT_NAME)-container $(IMG)
	@echo "$(CONTAINER_TOOL) installation complete."
	@echo "To use $(PROJECT_NAME), run:"
	@echo "alias $(PROJECT_NAME)='$(CONTAINER_TOOL) exec -it $(PROJECT_NAME)-container /app/$(PROJECT_NAME)'"

.PHONY: uninstall-docker
uninstall-docker: check-container-tool ## Uninstall app from $(CONTAINER_TOOL)
	@echo "Stopping and removing container in $(CONTAINER_TOOL)..."
	-$(CONTAINER_TOOL) stop $(PROJECT_NAME)-container && $(CONTAINER_TOOL) rm $(PROJECT_NAME)-container
	@echo "$(CONTAINER_TOOL) uninstallation complete. Remove alias if set: unalias $(PROJECT_NAME)"

### Kubernetes Targets (kubectl)

.PHONY: install-k8s
install-k8s: check-kubectl check-kustomize check-envsubst ## Install on Kubernetes
	export PROJECT_NAME=${PROJECT_NAME}
	export NAMESPACE=${NAMESPACE}
	@echo "Creating namespace (if needed) and setting context to $(NAMESPACE)..."
	kubectl create namespace $(NAMESPACE) 2>/dev/null || true
	kubectl config set-context --current --namespace=$(NAMESPACE)
	@echo "Deploying resources from deploy/ ..."
	# Build the kustomization from deploy, substitute variables, and apply the YAML
	kustomize build deploy | envsubst | kubectl apply -f -
	@echo "Waiting for pod to become ready..."
	sleep 5
	@POD=$$(kubectl get pod -l app=$(PROJECT_NAME)-statefulset -o jsonpath='{.items[0].metadata.name}'); \
	echo "Kubernetes installation complete."; \
	echo "To use the app, run:"; \
	echo "alias $(PROJECT_NAME)='kubectl exec -n $(NAMESPACE) -it $$POD -- /app/$(PROJECT_NAME)'"

.PHONY: uninstall-k8s
uninstall-k8s: check-kubectl check-kustomize check-envsubst ## Uninstall from Kubernetes
	export PROJECT_NAME=${PROJECT_NAME}
	export NAMESPACE=${NAMESPACE}
	@echo "Removing resources from Kubernetes..."
	kustomize build deploy | envsubst | kubectl delete --force -f - || true
	POD=$$(kubectl get pod -l app=$(PROJECT_NAME)-statefulset -o jsonpath='{.items[0].metadata.name}'); \
	echo "Deleting pod: $$POD"; \
	kubectl delete pod "$$POD" --force --grace-period=0 || true; \
	echo "Kubernetes uninstallation complete. Remove alias if set: unalias $(PROJECT_NAME)"

### OpenShift Targets (oc)

.PHONY: install-openshift
install-openshift: check-kubectl check-kustomize check-envsubst ## Install on OpenShift
	exit 0

.PHONY: uninstall-openshift
uninstall-openshift: check-kubectl check-kustomize check-envsubst ## Uninstall from OpenShift
	exit 0

### RBAC Targets (using kustomize and envsubst)

.PHONY: install-rbac
install-rbac: check-kubectl check-kustomize check-envsubst ## Install RBAC
	@echo "Applying RBAC configuration from deploy/rbac..."
	kustomize build deploy/rbac | envsubst '$$PROJECT_NAME $$NAMESPACE $$IMAGE_TAG_BASE $$VERSION' | kubectl apply -f -

.PHONY: uninstall-rbac
uninstall-rbac: check-kubectl check-kustomize check-envsubst ## Uninstall RBAC
	@echo "Removing RBAC configuration from deploy/rbac..."
	kustomize build deploy/rbac | envsubst '$$PROJECT_NAME $$NAMESPACE $$IMAGE_TAG_BASE $$VERSION' | kubectl delete -f - || true

##@ Version Extraction
.PHONY: version dev-registry prod-registry extract-version-info

dev-version: check-jq
	@jq -r '.dev-version' .version.json

prod-version: check-jq
	@jq -r '.prod-version' .version.json

dev-registry: check-jq
	@jq -r '."dev-registry"' .version.json

prod-registry: check-jq
	@jq -r '."prod-registry"' .version.json

extract-version-info: check-jq
	@echo "DEV_VERSION=$$(jq -r '."dev-version"' .version.json)"
	@echo "PROD_VERSION=$$(jq -r '."prod-version"' .version.json)"
	@echo "DEV_IMAGE_TAG_BASE=$$(jq -r '."dev-registry"' .version.json)"
	@echo "PROD_IMAGE_TAG_BASE=$$(jq -r '."prod-registry"' .version.json)"

##@ Load Version JSON

.PHONY: load-version-json
load-version-json: check-jq
	@if [ "$(DEV_VERSION)" = "0.0.1" ]; then \
	  DEV_VERSION=$$(jq -r '."dev-version"' .version.json); \
	  PROD_VERSION=$$(jq -r '."dev-version"' .version.json); \
	  echo "Loaded DEV_VERSION from .version.json: $$DEV_VERSION"; \
	  echo "Loaded PROD_VERSION from .version.json: $$PROD_VERSION"; \
	  export DEV_VERSION; \
	  export PROD_VERSION; \
	fi && \
	CURRENT_DEFAULT="ghcr.io/llm-d/$(PROJECT_NAME)"; \
	if [ "$(IMAGE_TAG_BASE)" = "$$CURRENT_DEFAULT" ]; then \
	  IMAGE_TAG_BASE=$$(jq -r '."dev-registry"' .version.json); \
	  echo "Loaded IMAGE_TAG_BASE from .version.json: $$IMAGE_TAG_BASE"; \
	  export IMAGE_TAG_BASE; \
	fi && \
	echo "Final values: DEV_VERSION=$$DEV_VERSION, PROD_VERSION=$$PROD_VERSION, IMAGE_TAG_BASE=$$IMAGE_TAG_BASE"

.PHONY: env
env: load-version-json ## Print environment variables
	@echo "DEV_VERSION=$(DEV_VERSION)"
	@echo "PROD_VERSION=$(PROD_VERSION)"
	@echo "IMAGE_TAG_BASE=$(IMAGE_TAG_BASE)"
	@echo "IMG=$(IMG)"
	@echo "CONTAINER_TOOL=$(CONTAINER_TOOL)"

##@ Tools

.PHONY: check-tools
check-tools: \
  check-go \
  check-ginkgo \
  check-golangci-lint \
  check-cmake \
  check-jq \
  check-kustomize \
  check-envsubst \
  check-container-tool \
  check-kubectl \
  check-buildah \
  check-podman
	@echo "All required tools are installed."
.PHONY: check-go
check-go:
	@command -v go >/dev/null 2>&1 || { \
	  echo "Go is not installed. Install it from https://golang.org/dl/"; exit 1; }

.PHONY: check-ginkgo
check-ginkgo:
	@command -v ginkgo >/dev/null 2>&1 || { \
	  echo "ginkgo is not installed. Install with: go install github.com/onsi/ginkgo/v2/ginkgo@latest"; exit 1; }

.PHONY: check-golangci-lint
check-golangci-lint:
	@command -v golangci-lint >/dev/null 2>&1 || { \
	  echo "golangci-lint is not installed. Install from https://golangci-lint.run/usage/install/"; exit 1; }

.PHONY: check-jq
check-jq:
	@command -v jq >/dev/null 2>&1 || { \
	  echo "jq is not installed. Install it from https://stedolan.github.io/jq/download/"; exit 1; }

.PHONY: check-kustomize
check-kustomize:
	@command -v kustomize >/dev/null 2>&1 || { \
	  echo "kustomize is not installed. Install it from https://kubectl.docs.kubernetes.io/installation/kustomize/"; exit 1; }

.PHONY: check-envsubst
check-envsubst:
	@command -v envsubst >/dev/null 2>&1 || { \
	  echo "envsubst is not installed. It is part of gettext."; \
	  echo "Try: sudo apt install gettext OR brew install gettext"; exit 1; }

.PHONY: check-container-tool
check-container-tool:
	@command -v $(CONTAINER_TOOL) >/dev/null 2>&1 || { \
	  echo "$(CONTAINER_TOOL) is not installed."; \
	  echo "Try: sudo apt install $(CONTAINER_TOOL) OR brew install $(CONTAINER_TOOL)"; exit 1; }

.PHONY: check-kubectl
check-kubectl:
	@command -v kubectl >/dev/null 2>&1 || { \
	  echo "kubectl is not installed. Install it from https://kubernetes.io/docs/tasks/tools/"; exit 1; }

.PHONY: check-builder
check-builder:
	@if [ -z "$(BUILDER)" ]; then \
		echo "No container builder tool (buildah, docker, or podman) found."; \
		exit 1; \
	else \
		echo "Using builder: $(BUILDER)"; \
	fi

.PHONY: check-podman
check-podman:
	@command -v podman >/dev/null 2>&1 || { \
	  echo "Podman is not installed. You can install it with:"; \
	  echo "sudo apt install podman  OR  brew install podman"; exit 1; }

check-cmake:
	@command -v cmake >/dev/null 2>&1 || { \
	  echo "CMake is not installed. Install it with your system package manager."; exit 1; }

##@ Alias checking
.PHONY: check-alias
check-alias: check-container-tool
	@echo "Checking alias functionality for container '$(PROJECT_NAME)-container'..."
	@if ! $(CONTAINER_TOOL) exec $(PROJECT_NAME)-container /app/$(PROJECT_NAME) --help >/dev/null 2>&1; then \
	  echo "The container '$(PROJECT_NAME)-container' is running, but the alias might not work."; \
	  echo "Try: $(CONTAINER_TOOL) exec -it $(PROJECT_NAME)-container /app/$(PROJECT_NAME)"; \
	else \
	  echo "Alias is likely to work: alias $(PROJECT_NAME)='$(CONTAINER_TOOL) exec -it $(PROJECT_NAME)-container /app/$(PROJECT_NAME)'"; \
	fi

.PHONY: print-namespace
print-namespace: ## Print the current namespace
	@echo "$(NAMESPACE)"

.PHONY: print-project-name
print-project-name: ## Print the current project name
	@echo "$(PROJECT_NAME)"

.PHONY: clean
clean: ## Clean build artifacts
	@printf "\033[33;1m==== Cleaning build artifacts ====\033[0m\n"
	@rm -rf build/
	@echo "Build artifacts cleaned."

.PHONY: install-hooks
install-hooks: ## Install git hooks
	git config core.hooksPath hooks


##@ ZMQ Setup

.PHONY: download-zmq
download-zmq: ## Install ZMQ dependencies based on OS/ARCH
	@echo "Checking if ZMQ is already installed..."
	@if pkg-config --exists libzmq; then \
	  echo "✅ ZMQ is already installed."; \
	else \
	  echo "Installing ZMQ dependencies..."; \
	  if [ "$(TARGETOS)" = "linux" ]; then \
	    if [ -x "$$(command -v apt)" ]; then \
	      apt update && apt install -y libzmq3-dev; \
	    elif [ -x "$$(command -v dnf)" ]; then \
	      dnf install -y zeromq-devel; \
	    else \
	      echo "Unsupported Linux package manager. Install libzmq manually."; \
	      exit 1; \
	    fi; \
	  elif [ "$(TARGETOS)" = "darwin" ]; then \
	    if [ -x "$$(command -v brew)" ]; then \
	      brew install zeromq; \
	    else \
	      echo "Homebrew is not installed and is required to install zeromq. Install it from https://brew.sh/"; \
	      exit 1; \
	    fi; \
	  else \
	    echo "Unsupported OS: $(TARGETOS). Install libzmq manually - check https://zeromq.org/download/ for guidance."; \
	    exit 1; \
	  fi; \
	  echo "✅ ZMQ dependencies installed."; \
	fi
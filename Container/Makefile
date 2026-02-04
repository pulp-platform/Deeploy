# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

# Variables
TOOLCHAIN_IMAGE ?= ghcr.io/pulp-platform/deeploy-toolchain:latest
DEEPOY_IMAGE ?= ghcr.io/pulp-platform/deeploy:latest
DEEPOY_GAP9_IMAGE ?= ghcr.io/pulp-platform/deeploy-gap9:latest

SSH_PRIVATE_KEY ?= ~/.ssh/id_ed25519

.PHONY: all toolchain deeploy deeploy-gap9 push clean help

help:
	@echo "Makefile for building Deeploy Docker images"
	@echo ""
	@echo "Usage:"
	@echo "  make toolchain    # Build only the toolchain image"
	@echo "  make deeploy      # Build only the deeploy image"
	@echo "  make deeploy-gap9 # Build only the GAP9 deeploy image"
	@echo "  make all          # Build all images"
	@echo "  make clean        # Remove all images (optional and dangerous)"
	@echo ""
	@echo "Build Variables:"
	@echo "  TOOLCHAIN_IMAGE   # Name of the toolchain image (default: $(TOOLCHAIN_IMAGE))"
	@echo "  DEEPOY_IMAGE      # Name of the deeploy image (default: $(DEEPOY_IMAGE))"
	@echo "  DEEPOY_GAP9_IMAGE # Name of the GAP9 deeploy image (default: $(DEEPOY_GAP9_IMAGE))"
	@echo "  SSH_PRIVATE_KEY   # Path to SSH private key for GAP9 build (default: $(SSH_PRIVATE_KEY))"
	@echo ""
	@echo "Example Usage:"
	@echo "  make toolchain TOOLCHAIN_IMAGE=my-toolchain:latest"
	@echo "  make deeploy DEEPOY_IMAGE=my-deeploy:latest"
	@echo "  make deeploy-gap9 DEEPOY_GAP9_IMAGE=my-deeploy-gap9:latest"
	@echo "  make all TOOLCHAIN_IMAGE=my-toolchain:latest DEEPOY_IMAGE=my-deeploy:latest"
	@echo "  make deeploy-gap9 SSH_PRIVATE_KEY=/path/to/key"

# Build only the toolchain image
toolchain:
	DOCKER_BUILDKIT=1 docker build \
	-f Dockerfile.toolchain \
	-t $(TOOLCHAIN_IMAGE) ..

# Build the final Deeploy image using the toolchain image
deeploy:
	DOCKER_BUILDKIT=1 docker build \
	--ssh default \
	-f Dockerfile.deeploy \
	--build-arg BASE_IMAGE=$(TOOLCHAIN_IMAGE) \
	-t $(DEEPOY_IMAGE) ..

# Build the GAP9 Deeploy image
deeploy-gap9:
	eval $$(ssh-agent) && \
	ssh-add $(SSH_PRIVATE_KEY) && \
	DOCKER_BUILDKIT=1 docker build \
	--ssh default \
	-f Dockerfile.deeploy-gap9 \
	--build-arg DEEPOY_IMAGE=$(DEEPOY_IMAGE) \
	-t $(DEEPOY_GAP9_IMAGE) ..

# Build all images
all: toolchain deeploy deeploy-gap9

# Clean all images (optional and dangerous)
clean:
	docker rmi $(TOOLCHAIN_IMAGE) $(DEEPOY_IMAGE) $(DEEPOY_GAP9_IMAGE) || true

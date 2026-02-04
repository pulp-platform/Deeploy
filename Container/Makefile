# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

# Variables
TOOLCHAIN_IMAGE ?= ghcr.io/pulp-platform/deeploy-toolchain:latest
DEEPOY_IMAGE ?= ghcr.io/pulp-platform/deeploy:latest

.PHONY: all toolchain deeploy push clean help

help:
	@echo "Makefile for building Deeploy Docker images"
	@echo ""
	@echo "Usage:"
	@echo "  make toolchain   # Build only the toolchain image"
	@echo "  make deeploy     # Build only the deploy image"
	@echo "  make all         # Build both images"
	@echo "  make clean       # Remove all images (optional and dangerous)"
	@echo ""
	@echo "Build Variables:"
	@echo "  TOOLCHAIN_IMAGE  # Name of the toolchain image (default: $(TOOLCHAIN_IMAGE))"
	@echo "  DEEPOY_IMAGE     # Name of the deploy image (default: $(DEEPOY_IMAGE))"
	@echo ""
	@echo "Example Usage:"
	@echo "  make toolchain TOOLCHAIN_IMAGE=my-toolchain:latest"
	@echo "  make deeploy DEEPOY_IMAGE=my-deploy:latest"
	@echo "  make all TOOLCHAIN_IMAGE=my-toolchain:latest DEEPOY_IMAGE=my-deploy:latest"

# Build only the toolchain image
toolchain:
	DOCKER_BUILDKIT=1 docker build -f Dockerfile.toolchain -t $(TOOLCHAIN_IMAGE) ..

# Build the final deploy image using the toolchain image
deeploy:
	DOCKER_BUILDKIT=1 docker build -f Dockerfile.deeploy --build-arg BASE_IMAGE=$(TOOLCHAIN_IMAGE) -t $(DEEPOY_IMAGE) ..

# Build both
all: toolchain deeploy

# Clean all images (optional and dangerous)
clean:
	docker rmi $(TOOLCHAIN_IMAGE) $(DEEPOY_IMAGE) || true

# ----------------------------------------------------------------------
#
# File: Makefile
#
# Created: 20.05.2025
#
# Copyright (C) 2025, ETH Zurich and University of Bologna.
#
# Authors:
# - Philip Wiese, ETH Zurich
#
# ----------------------------------------------------------------------
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the License); you may
# not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an AS IS BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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

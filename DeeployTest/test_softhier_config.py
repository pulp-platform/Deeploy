# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0
"""Test configuration for SoftHier platform."""

# SoftHier platform uses gvsoc simulator
# Default configuration: 1 cluster

DEFAULT_NUM_CLUSTERS = 1

KERNEL_TESTS = [
    "Adder",
]

MODEL_TESTS = []

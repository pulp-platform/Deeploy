# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0
"""Test configuration for Chimera platform."""

# Chimera platform uses gvsoc simulator
# Currently only Adder test is in CI

KERNEL_TESTS = [
    "Adder",
]

MODEL_TESTS = []

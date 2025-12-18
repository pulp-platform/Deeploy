# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

"""
Legacy test file for Generic platform (DEPRECATED).

This file is kept for backwards compatibility but will be removed in the future.
Please use test_platforms.py instead, which supports multiple platforms.

To run only Generic platform tests:
    pytest -m generic

To run Generic kernel tests:
    pytest -m "generic and kernels"

To run Generic model tests:
    pytest -m "generic and models"
"""

# Import all test functions from the new centralized test file
from test_platforms import test_generic_kernels, test_generic_models

__all__ = ["test_generic_kernels", "test_generic_models"]

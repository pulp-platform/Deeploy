# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

import re
from dataclasses import dataclass
from typing import Optional


@dataclass
class TestResult:
    success: bool
    error_count: int
    total_count: int
    stdout: str
    stderr: str = ""
    runtime_cycles: Optional[int] = None


def parse_test_output(stdout: str, stderr: str = "") -> TestResult:

    output = stdout + stderr

    # Look for "Errors: X out of Y" pattern
    error_match = re.search(r'Errors:\s*(\d+)\s*out\s*of\s*(\d+)', output)

    if error_match:
        error_count = int(error_match.group(1))
        total_count = int(error_match.group(2))
        success = (error_count == 0)
    else:
        # Could not parse output - treat as failure
        error_count = -1
        total_count = -1
        success = False

    runtime_cycles = None
    cycle_match = re.search(r'Runtime:\s*(\d+)\s*cycles', output)
    if cycle_match:
        runtime_cycles = int(cycle_match.group(1))

    return TestResult(
        success=success,
        error_count=error_count,
        total_count=total_count,
        stdout=stdout,
        stderr=stderr,
        runtime_cycles=runtime_cycles,
    )

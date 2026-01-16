# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

import os
from dataclasses import dataclass
from typing import List, Literal, Optional


@dataclass
class DeeployTestConfig:
    """Configuration for a single test case."""
    test_name: str
    test_dir: str
    platform: str
    simulator: Literal['gvsoc', 'banshee', 'qemu', 'vsim', 'vsim.gui', 'host', 'none']
    tiling: bool
    gen_dir: str
    build_dir: str
    toolchain: str = "LLVM"
    toolchain_install_dir: Optional[str] = None
    cmake_args: List[str] = None
    gen_args: List[str] = None
    verbose: int = 0
    debug: bool = False

    def __post_init__(self):
        if self.cmake_args is None:
            self.cmake_args = []
        if self.gen_args is None:
            self.gen_args = []
        if self.toolchain_install_dir is None:
            self.toolchain_install_dir = os.environ.get('LLVM_INSTALL_DIR')

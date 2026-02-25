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
    training: bool = False
    # None means "auto-detect from ONNX graph / inputs.npz during codegen"
    n_train_steps: Optional[int] = None
    n_accum_steps: Optional[int] = None
    training_num_data_inputs: Optional[int] = None
    # Directory containing the optimizer ONNX (network.onnx with SGD nodes).
    # If None, defaults to <test_dir>/../simplemlp_optimizer when training=True.
    optimizer_dir: Optional[str] = None

    def __post_init__(self):
        if self.cmake_args is None:
            self.cmake_args = []
        if self.gen_args is None:
            self.gen_args = []
        if self.toolchain_install_dir is None:
            self.toolchain_install_dir = os.environ.get('LLVM_INSTALL_DIR')

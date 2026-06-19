#!/usr/bin/python
# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0
"""
Standalone runner for GAP9 training tests (forward + backward + optimizer).

Usage example (minimal — all training params auto-detected):
    python deeployTrainingRunner_gap9.py \\
        -t /app/Onnx4Deeploy/onnx/model/simplemlp_train

Usage example (explicit):
    python deeployTrainingRunner_gap9.py \\
        -t /app/Onnx4Deeploy/onnx/model/simplemlp_train \\
        --n-steps 2 --n-accum 2 --cores 8
"""

import os
import sys
from pathlib import Path

from testUtils.core import DeeployTestConfig, run_complete_test
from testUtils.core.paths import get_test_paths
from testUtils.deeployRunner import (DeeployRunnerArgumentParser, print_colored_result,
                                     print_configuration)

if __name__ == '__main__':

    def _setup_parser(parser):
        parser.add_argument('--cores',
                            type = int,
                            default = 8,
                            help = 'Number of cluster cores (default: 8)\n')
        parser.add_argument('--num-data-inputs',
                            type = int,
                            dest = 'num_data_inputs',
                            default = None,
                            help = 'Inputs that change each mini-batch. '
                                   'Auto-detected from ONNX graph if not given.\n')
        parser.add_argument('--n-steps',
                            type = int,
                            dest = 'n_steps',
                            default = None,
                            help = 'N_TRAIN_STEPS: optimizer steps. '
                                   'Auto-detected from inputs.npz if not given.\n')
        parser.add_argument('--n-accum',
                            type = int,
                            dest = 'n_accum',
                            default = None,
                            help = 'N_ACCUM_STEPS: mini-batches per step. '
                                   'Auto-detected from inputs.npz if not given.\n')
        parser.add_argument('--optimizer-dir',
                            type = str,
                            dest = 'optimizer_dir',
                            default = None,
                            help = 'Directory containing the optimizer network.onnx '
                                   '(default: auto-derived as <test_dir>/../simplemlp_optimizer)\n')

    parser = DeeployRunnerArgumentParser(tiling_arguments = False, platform_required = False)
    _setup_parser(parser)
    args = parser.parse_args()

    platform = 'GAP9'
    simulator = args.simulator if args.simulator else 'gvsoc'

    base_dir = Path(__file__).resolve().parent
    test_dir = args.dir
    gen_dir, test_dir_abs, test_name = get_test_paths(test_dir, platform, base_dir = str(base_dir))

    worker_id = os.environ.get('PYTEST_XDIST_WORKER', 'master')
    build_dir = str(base_dir / f'TEST_{platform.upper()}' / f'build_{worker_id}')

    cmake_args = [f'-DNUM_CORES={args.cores}', '-D SIMULATOR=' + simulator]
    if args.cmake:
        cmake_args.extend(args.cmake)

    gen_args = [f'--cores={args.cores}']
    if args.input_type_map:
        gen_args.append('--input-type-map')
        gen_args.extend(args.input_type_map)
    if args.input_offset_map:
        gen_args.append('--input-offset-map')
        gen_args.extend(args.input_offset_map)

    config = DeeployTestConfig(
        test_name = test_name,
        test_dir = test_dir_abs,
        platform = platform,
        simulator = simulator,
        tiling = False,
        gen_dir = gen_dir,
        build_dir = build_dir,
        toolchain = args.toolchain,
        toolchain_install_dir = args.toolchain_install_dir,
        cmake_args = cmake_args,
        gen_args = gen_args,
        verbose = args.verbose,
        debug = args.debug,
        training = True,
        n_train_steps = args.n_steps,
        n_accum_steps = args.n_accum,
        training_num_data_inputs = args.num_data_inputs,
        optimizer_dir = args.optimizer_dir,
    )

    print_configuration(config)

    try:
        result = run_complete_test(config, skipgen = args.skipgen, skipsim = args.skipsim)
        print_colored_result(result, config.test_name)
        sys.exit(0 if result.success else 1)

    except Exception as e:
        RED = '\033[91m'
        RESET = '\033[0m'
        print(f'\n{RED}✗ Test {config.test_name} FAILED with exception: {e}{RESET}')
        sys.exit(1)

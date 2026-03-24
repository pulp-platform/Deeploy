# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0
"""
Common entry point for Siracusa training test runners (non-tiled and tiled).

Usage:
    from testUtils.deeployTrainingRunner import main
    sys.exit(main(tiling_enabled=False))   # non-tiled
    sys.exit(main(tiling_enabled=True))    # tiled (SBTiler)
"""

import os
import sys
from pathlib import Path
from typing import Optional

# gapy (gvsoc launcher) uses `#!/usr/bin/env python3`.  Put /usr/bin first so
# it resolves to /usr/bin/python3 which has all required packages (gapylib,
# prettytable, …) rather than the minimal venv python.
os.environ['PATH'] = '/usr/bin:' + os.environ.get('PATH', '')

from .core import DeeployTestConfig, run_complete_test
from .core.paths import get_test_paths
from .deeployRunner import DeeployRunnerArgumentParser, print_colored_result, print_configuration


def main(tiling_enabled: bool = False, default_platform: str = 'Siracusa', default_simulator: str = 'gvsoc'):
    """
    Build parser, parse args, create DeeployTestConfig, and run the training test.

    Parameters
    ----------
    tiling_enabled:
        True  → passes tiling args (--l1, --l2, …) and sets tiling=True in config.
    default_platform:
        Platform used when -p is not given on the command line.
    default_simulator:
        Simulator used when -s is not given on the command line.
    """

    parser = DeeployRunnerArgumentParser(tiling_arguments = tiling_enabled, platform_required = False)

    parser.add_argument('--cores', type = int, default = 8, help = 'Number of cluster cores (default: 8)\n')
    parser.add_argument('--n-steps',
                        metavar = '<N>',
                        dest = 'n_steps',
                        type = int,
                        default = None,
                        help = 'N_TRAIN_STEPS: optimizer steps (auto-detected if not given)\n')
    parser.add_argument('--n-accum',
                        metavar = '<N>',
                        dest = 'n_accum',
                        type = int,
                        default = None,
                        help = 'N_ACCUM_STEPS: mini-batches per update step (auto-detected if not given)\n')
    parser.add_argument('--num-data-inputs',
                        metavar = '<N>',
                        dest = 'num_data_inputs',
                        type = int,
                        default = None,
                        help = 'Inputs that change each mini-batch (auto-detected if not given)\n')
    parser.add_argument('--optimizer-dir',
                        metavar = '<dir>',
                        dest = 'optimizer_dir',
                        type = str,
                        default = None,
                        help = 'Directory containing the optimizer network.onnx '
                               "(default: auto-derived by replacing '_train' with '_optimizer')\n")
    parser.add_argument('--tolerance',
                        metavar = '<tol>',
                        dest = 'tolerance',
                        type = float,
                        default = None,
                        help = 'Absolute loss tolerance for pass/fail comparison (default: auto from generateTrainingNetwork.py)\n')

    args = parser.parse_args()

    platform = default_platform
    simulator = args.simulator if args.simulator else default_simulator

    script_path = Path(__file__).resolve()
    base_dir = script_path.parent.parent

    gen_dir, test_dir_abs, test_name = get_test_paths(args.dir, platform, base_dir = str(base_dir))

    worker_id = os.environ.get('PYTEST_XDIST_WORKER', 'master')
    build_dir = str(base_dir / f'TEST_{platform.upper()}' / f'build_{worker_id}')

    cmake_args = [f'-DNUM_CORES={args.cores}']
    if args.cmake:
        cmake_args.extend(args.cmake)

    gen_args = [f'--cores={args.cores}']
    if args.tolerance is not None:
        gen_args.append(f'--tolerance={args.tolerance}')
    if args.input_type_map:
        gen_args.extend(['--input-type-map'] + list(args.input_type_map))
    if args.input_offset_map:
        gen_args.extend(['--input-offset-map'] + list(args.input_offset_map))

    if tiling_enabled:
        if getattr(args, 'defaultMemLevel', None):
            gen_args.append(f'--defaultMemLevel={args.defaultMemLevel}')
        if getattr(args, 'l1', None):
            gen_args.append(f'--l1={args.l1}')
        if getattr(args, 'l2', None) and args.l2 != 1024000:
            gen_args.append(f'--l2={args.l2}')
        if getattr(args, 'memAllocStrategy', None):
            gen_args.append(f'--memAllocStrategy={args.memAllocStrategy}')
        if getattr(args, 'searchStrategy', None):
            gen_args.append(f'--searchStrategy={args.searchStrategy}')
        if getattr(args, 'profileTiling', False):
            gen_args.append('--profileTiling')
        if getattr(args, 'plotMemAlloc', False):
            gen_args.append('--plotMemAlloc')

    config = DeeployTestConfig(
        test_name = test_name,
        test_dir = test_dir_abs,
        platform = platform,
        simulator = simulator,
        tiling = tiling_enabled,
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
        return 0 if result.success else 1
    except Exception as e:
        RED = '\033[91m'
        RESET = '\033[0m'
        print(f'\n{RED}✗ Test {config.test_name} FAILED with exception: {e}{RESET}')
        return 1

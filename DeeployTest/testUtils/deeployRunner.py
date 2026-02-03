# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

import argparse
import codecs
import os
import sys
from pathlib import Path
from typing import Optional

import coloredlogs

from Deeploy.Logging import DEFAULT_FMT
from Deeploy.Logging import DEFAULT_LOGGER as log
from Deeploy.Logging import DETAILED_FILE_LOG_FORMAT

from .core import DeeployTestConfig, run_complete_test
from .core.paths import get_test_paths


def cmake_str(arg_str):
    return "-D" + codecs.decode(str(arg_str), 'unicode_escape')


class _ArgumentDefaultMetavarTypeFormatter(argparse.ArgumentDefaultsHelpFormatter, argparse.MetavarTypeHelpFormatter):

    def __init__(self, prog: str, indent_increment: int = 2, max_help_position: int = 100, width = None) -> None:
        super().__init__(prog, indent_increment, max_help_position, width)


class DeeployRunnerArgumentParser(argparse.ArgumentParser):

    def __init__(self,
                 tiling_arguments: bool,
                 description: Optional[str] = None,
                 platform_required: bool = True,
                 allow_extra_args: bool = False):
        formatter = _ArgumentDefaultMetavarTypeFormatter

        if description is None:
            super().__init__(description = "Deeploy Code Generation and Test Utility.", formatter_class = formatter)
        else:
            super().__init__(description = description, formatter_class = formatter)

        self.allow_extra_args = allow_extra_args

        self.tiling_arguments = tiling_arguments

        self.add_argument('-t',
                          metavar = '<dir>',
                          dest = 'dir',
                          type = str,
                          required = True,
                          help = 'Test directory (e.g., Tests/Kernels/Integer/Add/Regular)\n')
        self.add_argument('-p',
                          metavar = '<platform>',
                          dest = 'platform',
                          type = str,
                          required = platform_required,
                          default = None,
                          help = 'Target platform (e.g., Generic, QEMU-ARM, Siracusa, Snitch)\n')
        self.add_argument('-s',
                          metavar = '<simulator>',
                          dest = 'simulator',
                          type = str,
                          default = None,
                          help = 'Simulator to use (gvsoc, banshee, qemu, vsim, host, none)\n')
        self.add_argument('-v', action = 'count', dest = 'verbose', default = 0, help = 'Increase verbosity level\n')
        self.add_argument('-D',
                          dest = 'cmake',
                          action = 'extend',
                          nargs = "*",
                          type = cmake_str,
                          help = "Create or update a cmake cache entry\n")
        self.add_argument('--debug',
                          dest = 'debug',
                          action = 'store_true',
                          default = False,
                          help = 'Enable debugging mode\n')
        self.add_argument('--skipgen',
                          dest = 'skipgen',
                          action = 'store_true',
                          default = False,
                          help = 'Skip network generation (reuse existing generated code)\n')
        self.add_argument('--skipsim',
                          dest = 'skipsim',
                          action = 'store_true',
                          default = False,
                          help = 'Skip simulation (build only)\n')
        self.add_argument('--profileUntiled',
                          '--profile-untiled',
                          dest = 'profileUntiled',
                          action = 'store_true',
                          default = False,
                          help = 'Enable untiled profiling (Siracusa only)\n')
        self.add_argument('--toolchain',
                          metavar = '<LLVM|GCC>',
                          dest = 'toolchain',
                          type = str,
                          default = "LLVM",
                          help = 'Compiler toolchain\n')
        self.add_argument('--toolchain-install-dir',
                          metavar = '<dir>',
                          dest = 'toolchain_install_dir',
                          type = str,
                          default = os.environ.get('LLVM_INSTALL_DIR'),
                          help = 'Toolchain installation directory\n')
        self.add_argument('--input-type-map',
                          nargs = '*',
                          default = [],
                          type = str,
                          help = '(Optional) mapping of input names to data types. '
                          'Example: --input-type-map input_0=int8_t input_1=float32_t\n')
        self.add_argument('--input-offset-map',
                          nargs = '*',
                          default = [],
                          type = str,
                          help = '(Optional) mapping of input names to offsets. '
                          'Example: --input-offset-map input_0=0 input_1=128\n')

        if self.tiling_arguments:
            self.add_argument('--defaultMemLevel',
                              metavar = '<level>',
                              dest = 'defaultMemLevel',
                              type = str,
                              default = "L2",
                              help = 'Default memory level (L2 or L3)\n')
            self.add_argument('--doublebuffer', action = 'store_true', help = 'Enable double buffering\n')
            self.add_argument('--l1',
                              metavar = '<size>',
                              dest = 'l1',
                              type = int,
                              default = 64000,
                              help = 'L1 size in bytes\n')
            self.add_argument('--l2',
                              metavar = '<size>',
                              dest = 'l2',
                              type = int,
                              default = 1024000,
                              help = 'L2 size in bytes\n')
            self.add_argument('--randomizedMemoryScheduler',
                              action = "store_true",
                              help = 'Enable randomized memory scheduler\n')
            self.add_argument('--profileTiling', action = 'store_true', help = 'Enable tiling profiling\n')
            self.add_argument('--memAllocStrategy',
                              metavar = '<strategy>',
                              dest = 'memAllocStrategy',
                              type = str,
                              default = "MiniMalloc",
                              help = 'Memory allocation strategy: TetrisRandom, TetrisCo-Opt, MiniMalloc\n')
            self.add_argument('--searchStrategy',
                              metavar = '<strategy>',
                              dest = 'searchStrategy',
                              type = str,
                              default = "random-max",
                              help = 'CP solver search strategy: random-max, max, min\n')
            self.add_argument('--plotMemAlloc',
                              action = 'store_true',
                              help = 'Plot memory allocation and save in deeployState folder\n')

        self.args = None

    def parse_args(self, args = None, namespace = None) -> argparse.Namespace:

        self.args = super().parse_args(args, namespace)

        if self.args.verbose > 2:
            coloredlogs.install(level = 'DEBUG', logger = log, fmt = DETAILED_FILE_LOG_FORMAT)
        elif self.args.verbose > 1:
            coloredlogs.install(level = 'DEBUG', logger = log, fmt = DEFAULT_FMT)
        elif self.args.verbose > 0:
            coloredlogs.install(level = 'INFO', logger = log, fmt = DEFAULT_FMT)
        else:
            coloredlogs.install(level = 'WARNING', logger = log, fmt = DEFAULT_FMT)

        return self.args


def create_config_from_args(args: argparse.Namespace,
                            platform: str,
                            simulator: str,
                            tiling: bool,
                            platform_specific_cmake_args: Optional[list] = None) -> DeeployTestConfig:

    script_path = Path(__file__).resolve()
    base_dir = script_path.parent.parent

    test_dir = args.dir
    gen_dir, test_dir_abs, test_name = get_test_paths(test_dir, platform, base_dir = str(base_dir))

    # Use worker-specific build directory to avoid collisions with parallel execution with pytest-xdist
    worker_id = os.environ.get("PYTEST_XDIST_WORKER", "master")
    if worker_id == "master":
        build_dir = str(base_dir / f"TEST_{platform.upper()}" / "build_master")
    else:
        build_dir = str(base_dir / f"TEST_{platform.upper()}" / f"build_{worker_id}")

    cmake_args_list = list(args.cmake) if args.cmake else []

    # Add platform-specific CMake args
    if platform_specific_cmake_args:
        cmake_args_list.extend(platform_specific_cmake_args)

    # Prepare generation args
    gen_args_list = []

    if args.input_type_map:
        gen_args_list.append("--input-type-map")
        gen_args_list.extend(args.input_type_map)
    if args.input_offset_map:
        gen_args_list.append("--input-offset-map")
        gen_args_list.extend(args.input_offset_map)

    if tiling:
        if hasattr(args, 'defaultMemLevel') and args.defaultMemLevel:
            gen_args_list.append(f"--defaultMemLevel={args.defaultMemLevel}")
        if hasattr(args, 'doublebuffer') and args.doublebuffer:
            gen_args_list.append("--doublebuffer")
        if hasattr(args, 'l1') and args.l1:
            gen_args_list.append(f"--l1={args.l1}")
        if hasattr(args, 'l2') and args.l2 and args.l2 != 1024000:
            gen_args_list.append(f"--l2={args.l2}")
        if hasattr(args, 'randomizedMemoryScheduler') and args.randomizedMemoryScheduler:
            gen_args_list.append("--randomizedMemoryScheduler")
        if hasattr(args, 'profileTiling') and args.profileTiling:
            gen_args_list.append("--profileTiling")
        if hasattr(args, 'memAllocStrategy') and args.memAllocStrategy:
            gen_args_list.append(f"--memAllocStrategy={args.memAllocStrategy}")
        if hasattr(args, 'searchStrategy') and args.searchStrategy:
            gen_args_list.append(f"--searchStrategy={args.searchStrategy}")
        if hasattr(args, 'plotMemAlloc') and args.plotMemAlloc:
            gen_args_list.append("--plotMemAlloc")

    if not tiling and getattr(args, 'profileUntiled', False):
        gen_args_list.append("--profileUntiled")

    config = DeeployTestConfig(
        test_name = test_name,
        test_dir = test_dir_abs,
        platform = platform,
        simulator = simulator,
        tiling = tiling,
        gen_dir = gen_dir,
        build_dir = build_dir,
        toolchain = args.toolchain,
        toolchain_install_dir = args.toolchain_install_dir,
        gvsoc_install_dir = getattr(args, 'gvsoc_install_dir', None),
        cmake_args = cmake_args_list,
        gen_args = gen_args_list,
        verbose = args.verbose,
        debug = args.debug,
    )

    return config


def print_colored_result(result, test_name: str):

    GREEN = '\033[92m'
    RED = '\033[91m'
    RESET = '\033[0m'

    if result.success and result.error_count == 0:
        print(f"\n{GREEN}✓ Test {test_name} PASSED - No errors found{RESET}")
        if result.runtime_cycles is not None:
            print(f"{GREEN}  Runtime: {result.runtime_cycles} cycles{RESET}")
    else:
        print(f"\n{RED}✗ Test {test_name} FAILED - {result.error_count} errors out of {result.total_count}{RESET}")
        if result.runtime_cycles is not None:
            print(f"{RED}  Runtime: {result.runtime_cycles} cycles{RESET}")


def print_configuration(config: DeeployTestConfig):

    CYAN = '\033[96m'
    BOLD = '\033[1m'
    RESET = '\033[0m'

    print(f"\n{BOLD}{CYAN}═══════════════════════════════════════════════════════════════{RESET}")
    print(f"{BOLD}{CYAN}                    Deeploy Test Configuration                 {RESET}")
    print(f"{BOLD}{CYAN}═══════════════════════════════════════════════════════════════{RESET}\n")

    print(f"{BOLD}Test Configuration:{RESET}")
    print(f"  Test Name           : {config.test_name}")
    print(f"  Test Directory      : {config.test_dir}")
    print(f"  Generation Directory: {config.gen_dir}")
    print(f"  Build Directory     : {config.build_dir}")

    print(f"\n{BOLD}Platform Configuration:{RESET}")
    print(f"  Platform            : {config.platform}")
    print(f"  Simulator           : {config.simulator}")
    print(f"  Tiling Enabled      : {'Yes' if config.tiling else 'No'}")

    print(f"\n{BOLD}Build Configuration:{RESET}")
    print(f"  Toolchain           : {config.toolchain}")
    if config.toolchain_install_dir:
        print(f"  Toolchain Directory : {config.toolchain_install_dir}")
    if config.cmake_args:
        print(f"  CMake Arguments     : {' '.join(config.cmake_args)}")

    print(f"\n{BOLD}Runtime Configuration:{RESET}")
    print(f"  Verbosity Level     : {config.verbose}")
    print(f"  Debug Mode          : {'Enabled' if config.debug else 'Disabled'}")
    if config.gen_args:
        print(f"  Generation Arguments: {' '.join(config.gen_args)}")

    print(f"\n{BOLD}{CYAN}═══════════════════════════════════════════════════════════════{RESET}\n")


def main(default_platform: Optional[str] = None,
         default_simulator: Optional[str] = None,
         tiling_enabled: bool = False,
         platform_specific_cmake_args: Optional[list] = None,
         parsed_args: Optional[argparse.Namespace] = None,
         parser_setup_callback = None):
    """
    Main entry point for Deeploy test runners.
    
    Args:
        default_platform: Default platform if not specified via -p
        default_simulator: Default simulator if not specified via -s
        tiling_enabled: Whether tiling is enabled
        platform_specific_cmake_args: Additional CMake arguments for platform-specific configurations
        parsed_args: Pre-parsed arguments (if None, will parse from sys.argv)
        parser_setup_callback: Optional callback to configure parser before parsing (receives parser as arg)
    """

    if parsed_args is None:
        # Make -p optional if default_platform is provided
        parser = DeeployRunnerArgumentParser(tiling_arguments = tiling_enabled,
                                             platform_required = (default_platform is None))

        # Allow platform-specific runners to add their own arguments
        if parser_setup_callback:
            parser_setup_callback(parser)

        args = parser.parse_args()
    else:
        args = parsed_args

    platform_map = {
        "generic": "Generic",
        "qemu-arm": "QEMU-ARM",
        "mempool": "MemPool",
        "siracusa": "Siracusa",
        "siracusa_w_neureka": "Siracusa_w_neureka",
        "snitch": "Snitch",
        "chimera": "Chimera",
        "softhier": "SoftHier",
    }

    if args.platform:
        platform = platform_map.get(args.platform.lower(), args.platform)
    else:
        platform = default_platform

    # Validate platform if default is provided
    if default_platform and args.platform:
        normalized_specified = platform_map.get(args.platform.lower(), args.platform)
        if normalized_specified != default_platform:
            RED = '\033[91m'
            BOLD = '\033[1m'
            RESET = '\033[0m'
            print(f"\n{RED}{BOLD}ERROR: Platform mismatch!{RESET}", file = sys.stderr)
            print(f"{RED}This runner is designed for the '{default_platform}' platform.{RESET}", file = sys.stderr)
            print(f"{RED}You specified platform: '{args.platform}' (normalized to '{normalized_specified}'){RESET}\n",
                  file = sys.stderr)
            print(f"Please use one of the following options:", file = sys.stderr)
            print(f"  1. Remove the '-p {args.platform}' argument to use the default platform", file = sys.stderr)
            print(f"  2. Use the correct platform-specific runner script for '{normalized_specified}'",
                  file = sys.stderr)
            sys.exit(1)

    simulator = args.simulator if args.simulator else default_simulator

    if platform is None:
        print("Error: Platform must be specified with -p or provided as default", file = sys.stderr)
        sys.exit(1)

    if simulator is None:
        simulator_map = {
            "Generic": "host",
            "QEMU-ARM": "qemu",
            "MemPool": "banshee",
            "Siracusa": "gvsoc",
            "Siracusa_w_neureka": "gvsoc",
            "Snitch": "gvsoc",
            "Chimera": "gvsoc",
            "SoftHier": "gvsoc",
        }
        simulator = simulator_map.get(platform, "host")
        log.info(f"No simulator specified, using default for {platform}: {simulator}")

    # Extract platform-specific CMake args from parsed args if available
    if platform_specific_cmake_args is None:
        platform_specific_cmake_args = []

    # Check for platform-specific arguments in args object and build CMake args
    if hasattr(args, 'cores'):
        platform_specific_cmake_args.append(f"-DNUM_CORES={args.cores}")
    elif hasattr(args, 'num_cores'):
        platform_specific_cmake_args.append(f"-DNUM_CORES={args.num_cores}")

    if hasattr(args, 'num_clusters'):
        platform_specific_cmake_args.append(f"-DNUM_CLUSTERS={args.num_clusters}")

    config = create_config_from_args(args, platform, simulator, tiling_enabled, platform_specific_cmake_args)

    print_configuration(config)

    try:
        result = run_complete_test(config, skipgen = args.skipgen, skipsim = args.skipsim)

        print_colored_result(result, config.test_name)

        return 0 if result.success else 1

    except Exception as e:
        RED = '\033[91m'
        RESET = '\033[0m'
        print(f"\n{RED}✗ Test {config.test_name} FAILED with exception: {e}{RESET}")
        return 1


if __name__ == "__main__":
    sys.exit(main())

# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Literal, Optional, Tuple

from Deeploy.Logging import DEFAULT_LOGGER as log


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


@dataclass
class TestResult:
    """Results from running a test."""
    success: bool
    error_count: int
    total_count: int
    stdout: str
    stderr: str = ""
    runtime_cycles: Optional[int] = None


def get_test_paths(test_dir: str, platform: str, base_dir: Optional[str] = None) -> Tuple[str, str, str]:
    """
    Args:
        test_dir: Path to test directory (e.g., "Tests/Adder" or absolute path)
        platform: Platform name (e.g., "Generic")
        base_dir: Base directory for tests (defaults to DeeployTest/)
        
    Returns:
        Tuple of (gen_dir, test_dir_abs, test_name)
    """
    if base_dir is None:
        # Get the absolute path of this script's parent directory (testUtils -> DeeployTest)
        script_path = Path(__file__).resolve()
        base_dir = script_path.parent.parent
    else:
        base_dir = Path(base_dir)

    test_path = Path(test_dir)
    if not test_path.is_absolute():
        test_path = base_dir / test_dir

    test_path = test_path.resolve()
    test_name = test_path.name

    gen_dir_name = f"TEST_{platform.upper()}"

    # Check if path is inside base_dir
    try:
        rel_path = test_path.relative_to(base_dir)
        gen_dir = base_dir / gen_dir_name / rel_path
    except ValueError:
        # Path is outside base_dir
        gen_dir = base_dir / gen_dir_name / test_name
        log.warning(f"Test path {test_path} is outside base directory. Using {gen_dir}")

    return str(gen_dir), str(test_path), test_name


def generate_network(config: DeeployTestConfig, skip: bool = False) -> None:
    """
    Args:
        config: Test configuration
        skip: If True, skip generation (useful for re-running tests)
        
    Raises:
        RuntimeError: If network generation fails
    """
    if skip:
        log.info(f"Skipping network generation for {config.test_name}")
        return

    script_dir = Path(__file__).parent.parent

    if config.tiling:
        generation_script = script_dir / "testMVP.py"
    else:
        generation_script = script_dir / "generateNetwork.py"

    cmd = [
        "python",
        str(generation_script),
        "-d",
        config.gen_dir,
        "-t",
        config.test_dir,
        "-p",
        config.platform,
    ]

    # Add verbosity flags
    if config.verbose > 0:
        cmd.append("-" + "v" * config.verbose)

    # Add debug flag
    if config.debug:
        cmd.append("--debug")

    # Add additional generation arguments
    cmd.extend(config.gen_args)

    log.debug(f"[pytestRunner] Generation command: {' '.join(cmd)}")

    result = subprocess.run(cmd, check = False)

    if result.returncode != 0:
        log.error(f"Network generation failed with return code {result.returncode}")
        raise RuntimeError(f"Network generation failed for {config.test_name}")


def configure_cmake(config: DeeployTestConfig) -> None:
    """
    Args:
        config: Test configuration
        
    Raises:
        RuntimeError: If CMake configuration fails
    """
    assert config.toolchain_install_dir is not None, \
        "LLVM_INSTALL_DIR environment variable not set"

    cmake_cmd = os.environ.get("CMAKE", "cmake")
    if cmake_cmd == "cmake" and shutil.which("cmake") is None:
        raise RuntimeError("CMake not found. Please install CMake or set CMAKE environment variable")

    # Build CMake command
    cmd = [
        cmake_cmd,
        f"-DTOOLCHAIN={config.toolchain}",
        f"-DTOOLCHAIN_INSTALL_DIR={config.toolchain_install_dir}",
        f"-DGENERATED_SOURCE={config.gen_dir}",
        f"-Dplatform={config.platform}",
        f"-DTESTNAME={config.test_name}",
        f"-B{config.build_dir}",
    ]

    # Add custom CMake arguments
    for arg in config.cmake_args:
        if not arg.startswith("-D"):
            arg = "-D" + arg
        cmd.append(arg)

    # Add simulator flags
    if config.simulator == 'banshee':
        cmd.append("-Dbanshee_simulation=ON")
    else:
        cmd.append("-Dbanshee_simulation=OFF")

    if config.simulator == 'gvsoc':
        cmd.append("-Dgvsoc_simulation=ON")
    else:
        cmd.append("-Dgvsoc_simulation=OFF")

    # Last argument is the source directory
    script_dir = Path(__file__).parent.parent
    cmd.append(str(script_dir.parent))

    env = os.environ.copy()
    if config.verbose >= 3:
        env["VERBOSE"] = "1"

    log.debug(f"[pytestRunner] CMake command: {' '.join(cmd)}")

    result = subprocess.run(cmd, check = False, env = env)

    if result.returncode != 0:
        log.error(f"CMake configuration failed with return code {result.returncode}")
        raise RuntimeError(f"CMake configuration failed for {config.test_name}")


def build_binary(config: DeeployTestConfig) -> None:
    """
    Args:
        config: Test configuration
        
    Raises:
        RuntimeError: If build fails
    """
    cmake_cmd = os.environ.get("CMAKE", "cmake")

    cmd = [
        cmake_cmd,
        "--build",
        config.build_dir,
        "--target",
        config.test_name,
    ]

    env = os.environ.copy()
    if config.verbose >= 3:
        env["VERBOSE"] = "1"

    log.debug(f"[pytestRunner] Build command: {' '.join(cmd)}")

    result = subprocess.run(cmd, check = False, env = env)

    if result.returncode != 0:
        log.error(f"Build failed with return code {result.returncode}")
        raise RuntimeError(f"Build failed for {config.test_name}")


def run_simulation(config: DeeployTestConfig, skip: bool = False) -> TestResult:
    """
    Args:
        config: Test configuration
        skip: If True, skip simulation (useful for build-only tests)
        
    Returns:
        TestResult with parsed output
        
    Raises:
        RuntimeError: If simulation cannot be executed
    """
    if skip:
        log.info(f"Skipping simulation for {config.test_name}")
        return TestResult(success = True, error_count = 0, total_count = 0, stdout = "Skipped")

    if config.simulator == 'none':
        raise RuntimeError("No simulator specified!")

    if config.simulator == 'host':
        # Run binary directly
        binary_path = Path(config.build_dir) / "bin" / config.test_name
        cmd = [str(binary_path)]
    else:
        # Run via CMake target
        cmake_cmd = os.environ.get("CMAKE", "cmake")
        cmd = [
            cmake_cmd,
            "--build",
            config.build_dir,
            "--target",
            f"{config.simulator}_{config.test_name}",
        ]

    env = os.environ.copy()
    if config.verbose >= 3:
        env["VERBOSE"] = "1"

    # Add banshee-specific logging
    if config.simulator == 'banshee':
        if config.verbose == 1:
            env["BANSHEE_LOG"] = "warn"
        elif config.verbose == 2:
            env["BANSHEE_LOG"] = "info"
        elif config.verbose >= 3:
            env["BANSHEE_LOG"] = "debug"

    log.debug(f"[pytestRunner] Simulation command: {' '.join(cmd)}")

    result = subprocess.run(cmd, capture_output = True, text = True, env = env)

    # Print captured output so it's visible when running with pytest -s
    if result.stdout:
        print(result.stdout, end = '')
    if result.stderr:
        print(result.stderr, end = '', file = sys.stderr)

    # Parse output for error count
    output = result.stdout + result.stderr

    # Look for "Errors: X out of Y" pattern
    error_match = re.search(r'Errors:\s*(\d+)\s*out\s*of\s*(\d+)', output)

    if error_match:
        error_count = int(error_match.group(1))
        total_count = int(error_match.group(2))
        success = (error_count == 0)
    else:
        # Could not parse output - treat as failure
        log.warning(f"Could not parse error count from output:\n{output}")
        error_count = -1
        total_count = -1
        success = False

    # Try to parse runtime cycles
    runtime_cycles = None
    cycle_match = re.search(r'Runtime:\s*(\d+)\s*cycles', output)
    if cycle_match:
        runtime_cycles = int(cycle_match.group(1))

    return TestResult(
        success = success,
        error_count = error_count,
        total_count = total_count,
        stdout = result.stdout,
        stderr = result.stderr,
        runtime_cycles = runtime_cycles,
    )


def run_complete_test(config: DeeployTestConfig, skipgen: bool = False, skipsim: bool = False) -> TestResult:
    """
    Run a complete test: generate, configure, build, and simulate.
    
    Args:
        config: Test configuration
        skipgen: Skip network generation
        skipsim: Skip simulation
        
    Returns:
        TestResult with parsed output
    """
    log.info(f"################## Testing {config.test_name} on {config.platform} Platform ##################")

    # Step 1: Generate network
    generate_network(config, skip = skipgen)

    # Step 2: Configure CMake
    configure_cmake(config)

    # Step 3: Build binary
    build_binary(config)

    # Step 4: Run simulation
    result = run_simulation(config, skip = skipsim)

    return result


def get_worker_id() -> str:
    """
    Get the pytest-xdist worker ID for parallel test execution.
    
    Returns:
        Worker ID string (e.g., 'gw0', 'gw1', 'master' for non-parallel)
    """
    return os.environ.get("PYTEST_XDIST_WORKER", "master")


def create_test_config(
    test_name: str,
    platform: str,
    simulator: Literal['gvsoc', 'banshee', 'qemu', 'vsim', 'vsim.gui', 'host', 'none'],
    deeploy_test_dir: str,
    toolchain: str,
    toolchain_dir: Optional[str],
    cmake_args: List[str],
    tiling: bool = False,
    cores: Optional[int] = None,
    l1: Optional[int] = None,
    l2: int = 1024000,
    default_mem_level: str = "L2",
    double_buffer: bool = False,
    mem_alloc_strategy: str = "MiniMalloc",
    search_strategy: str = "random-max",
    profile_tiling: bool = False,
    plot_mem_alloc: bool = False,
    randomized_mem_scheduler: bool = False,
) -> DeeployTestConfig:
    """
    Create DeeployTestConfig for a specific test and platform.
    
    Args:
        test_name: Name of the test
        platform: Target platform (e.g., "Generic", "QEMU-ARM", "Siracusa")
        simulator: Simulator to use
        deeploy_test_dir: Base DeeployTest directory
        toolchain: Toolchain to use - LLVM/GCC
        toolchain_dir: Path to toolchain installation
        cmake_args: Additional CMake arguments
        tiling: Whether to use tiling
        cores: Number of cores (for Siracusa platforms)
        l1: L1 memory size in bytes (for tiled platforms)
        l2: L2 memory size in bytes (default: 1024000)
        default_mem_level: Default memory level ("L2" or "L3")
        double_buffer: Enable double buffering
        mem_alloc_strategy: Memory allocation strategy
        search_strategy: CP solver search strategy
        profile_tiling: Enable tiling profiling
        plot_mem_alloc: Enable memory allocation plotting
        randomized_mem_scheduler: Enable randomized memory scheduler
        
    Returns:
        DeeployTestConfig instance
    """
    test_dir = f"Tests/{test_name}"

    gen_dir, test_dir_abs, test_name_clean = get_test_paths(test_dir, platform, base_dir = deeploy_test_dir)

    worker_id = get_worker_id()
    
    # Build directory: shared per worker, not per test (for ccache efficiency)
    # Only add worker suffix for parallel execution (worker_id != "master")
    if worker_id == "master":
        build_dir = str(Path(deeploy_test_dir) / f"TEST_{platform.upper()}" / "build_master")
    else:
        build_dir = str(Path(deeploy_test_dir) / f"TEST_{platform.upper()}" / f"build_{worker_id}")

    cmake_args_list = list(cmake_args) if cmake_args else []
    if cores is not None:
        cmake_args_list.append(f"NUM_CORES={cores}")

    gen_args_list = []
    
    if cores is not None and platform in ["Siracusa", "Siracusa_w_neureka"]:
        gen_args_list.append(f"--cores={cores}")
    
    if tiling:
        if l1 is not None:
            gen_args_list.append(f"--l1={l1}")
        if l2 != 1024000:
            gen_args_list.append(f"--l2={l2}")
        if default_mem_level != "L2":
            gen_args_list.append(f"--defaultMemLevel={default_mem_level}")
        if double_buffer:
            gen_args_list.append("--doublebuffer")
        if mem_alloc_strategy != "MiniMalloc":
            gen_args_list.append(f"--memAllocStrategy={mem_alloc_strategy}")
        if search_strategy != "random-max":
            gen_args_list.append(f"--searchStrategy={search_strategy}")
        if profile_tiling:
            gen_args_list.append("--profileTiling")
        if plot_mem_alloc:
            gen_args_list.append("--plotMemAlloc")
        if randomized_mem_scheduler:
            gen_args_list.append("--randomizedMemoryScheduler")

    config = DeeployTestConfig(
        test_name = test_name_clean,
        test_dir = test_dir_abs,
        platform = platform,
        simulator = simulator,
        tiling = tiling,
        gen_dir = gen_dir,
        build_dir = build_dir,
        toolchain = toolchain,
        toolchain_install_dir = toolchain_dir,
        cmake_args = cmake_args_list,
        gen_args = gen_args_list,
    )

    return config


def run_and_assert_test(test_name: str, config: DeeployTestConfig, skipgen: bool, skipsim: bool) -> None:
    """
    Shared helper function to run a test and assert its results.
    
    Args:
        test_name: Name of the test
        config: DeeployTestConfig instance
        skipgen: Whether to skip network generation
        skipsim: Whether to skip simulation
        
    Raises:
        AssertionError: If test fails or has errors
    """
    result = run_complete_test(config, skipgen = skipgen, skipsim = skipsim)

    assert result.success, (f"Test {test_name} failed with {result.error_count} errors out of {result.total_count}\n"
                            f"Output:\n{result.stdout}")

    if result.error_count >= 0:
        assert result.error_count == 0, (f"Found {result.error_count} errors out of {result.total_count} tests")

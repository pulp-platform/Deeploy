# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

import os
import re
import shutil
import subprocess
import sys
import threading
from pathlib import Path

from Deeploy.Logging import DEFAULT_LOGGER as log

from .config import DeeployTestConfig
from .output_parser import TestResult, parse_test_output


def generate_network(config: DeeployTestConfig, skip: bool = False) -> None:
    """
    Generate network code from ONNX model.

    Raises:
        RuntimeError: If network generation fails
    """
    if skip:
        log.info(f"Skipping network generation for {config.test_name}")
        return

    script_dir = Path(__file__).parent.parent.parent

    if config.platform == "XDNA2":
        generation_script = script_dir / "generateNetwork_xdna2.py"
    elif config.tiling:
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

    log.debug(f"[Execution] Generation command: {' '.join(cmd)}")

    result = subprocess.run(cmd, check = False)

    if result.returncode != 0:
        log.error(f"Network generation failed with return code {result.returncode}")
        raise RuntimeError(f"Network generation failed for {config.test_name}")


def configure_cmake(config: DeeployTestConfig) -> None:

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

    # Add GVSOC_INSTALL_DIR if available
    if config.gvsoc_install_dir:
        cmd.append(f"-DGVSOC_INSTALL_DIR={config.gvsoc_install_dir}")

    for arg in config.cmake_args:
        if not arg.startswith("-D"):
            arg = "-D" + arg
        cmd.append(arg)

    if config.simulator == 'banshee':
        cmd.append("-Dbanshee_simulation=ON")
    else:
        cmd.append("-Dbanshee_simulation=OFF")

    if config.simulator == 'gvsoc':
        cmd.append("-Dgvsoc_simulation=ON")
    else:
        cmd.append("-Dgvsoc_simulation=OFF")

    # Last argument is the source directory
    script_dir = Path(__file__).parent.parent.parent
    cmd.append(str(script_dir.parent))

    env = os.environ.copy()
    if config.verbose >= 3:
        env["VERBOSE"] = "1"

    log.debug(f"[Execution] CMake command: {' '.join(cmd)}")

    result = subprocess.run(cmd, check = False, env = env)

    if result.returncode != 0:
        log.error(f"CMake configuration failed with return code {result.returncode}")
        raise RuntimeError(f"CMake configuration failed for {config.test_name}")


def build_binary(config: DeeployTestConfig) -> None:

    cmake_cmd = os.environ.get("CMAKE", "cmake")

    cmd = [
        cmake_cmd,
        "--build",
        config.build_dir,
        "--target",
        config.test_name,
    ]

    # GAP9 requires the 'image' target to generate MRAM .bin files for GVSOC
    if config.platform == 'GAP9':
        cmd.append("image")

    env = os.environ.copy()
    if config.verbose >= 3:
        env["VERBOSE"] = "1"

    log.debug(f"[Execution] Build command: {' '.join(cmd)}")

    result = subprocess.run(cmd, check = False, env = env)

    if result.returncode != 0:
        log.error(f"Build failed with return code {result.returncode}")
        raise RuntimeError(f"Build failed for {config.test_name}")


# Source: https://stackoverflow.com/a/38662876
def escapeAnsi(line):
    ansi_escape = re.compile(r'(?:\x1B[@-_]|[\x80-\x9F])[0-?]*[ -/]*[@-~]')
    return ansi_escape.sub('', line)


def run_simulation(config: DeeployTestConfig, skip: bool = False) -> TestResult:
    """
    Run simulation and parse output.

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
        # Propagate verbosity to the host binary (e.g. XDNA2 main.cpp uses -v)
        if config.platform == "XDNA2" and config.verbose >= 1:
            cmd.append("-v")
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

    if config.simulator == 'banshee':
        if config.verbose == 1:
            env["BANSHEE_LOG"] = "warn"
        elif config.verbose == 2:
            env["BANSHEE_LOG"] = "info"
        elif config.verbose >= 3:
            env["BANSHEE_LOG"] = "debug"

    log.debug(f"[Execution] Simulation command: {' '.join(cmd)}")

    process = subprocess.Popen(
        cmd,
        stdout = subprocess.PIPE,
        stderr = subprocess.PIPE,
        text = True,
        env = env,
        bufsize = 1,
    )

    stdout_chunks = []
    stderr_chunks = []

    def _stream_reader(pipe, chunks, is_stderr: bool = False) -> None:
        assert pipe is not None
        for line in iter(pipe.readline, ''):
            chunks.append(line)
            if is_stderr:
                print(line, end = '', file = sys.stderr, flush = True)
            else:
                print(line, end = '', flush = True)
        pipe.close()

    stdout_thread = threading.Thread(target = _stream_reader, args = (process.stdout, stdout_chunks), daemon = True)
    stderr_thread = threading.Thread(target = _stream_reader, args = (process.stderr, stderr_chunks, True), daemon = True)

    stdout_thread.start()
    stderr_thread.start()

    returncode = process.wait()
    stdout_thread.join()
    stderr_thread.join()

    stdout = ''.join(stdout_chunks)
    stderr = ''.join(stderr_chunks)

    # Parse output for error count and cycles
    test_result = parse_test_output(stdout, stderr)

    if not test_result.success and test_result.error_count == -1:
        log.warning(f"Could not parse error count from output")

    return test_result


def run_complete_test(config: DeeployTestConfig, skipgen: bool = False, skipsim: bool = False) -> TestResult:
    """
    Run a complete test: generate, configure, build, and simulate.
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

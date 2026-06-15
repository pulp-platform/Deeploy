# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

import json
import os
import re
import shutil
import subprocess
from pathlib import Path
import sys

from Deeploy.Logging import DEFAULT_LOGGER as log

from .config import DeeployTestConfig
from .output_parser import TestResult, parse_test_output
import threading


def _augment_path(env: dict) -> dict:
    """Prepend gvsoc/llvm bin dirs to PATH based on installed env vars.

    The install dirs are already set as env vars (GVSOC_INSTALL_DIR,
    LLVM_INSTALL_DIR) but their bin/ subdirectories may not be in PATH.
    """
    extra = []
    for var in ('GVSOC_INSTALL_DIR', 'LLVM_INSTALL_DIR'):
        install_dir = env.get(var, '')
        if install_dir:
            bin_dir = str(Path(install_dir) / 'bin')
            current = env.get('PATH', '').split(':')
            if bin_dir not in current:
                extra.append(bin_dir)
    if extra:
        env['PATH'] = ':'.join(extra) + ':' + env.get('PATH', '')
    return env


def _resolve_optimizer_dir(config: DeeployTestConfig) -> str:
    """Return the optimizer ONNX directory for this config.

    Falls back to <test_dir>/../simplemlp_optimizer if not explicitly set.
    """
    if config.optimizer_dir:
        return config.optimizer_dir
    test_parent = Path(config.test_dir).parent
    return str(test_parent / "tinytransformer_optimizer")


def generate_network(config: DeeployTestConfig, skip: bool = False) -> None:
    """
    Generate network code from ONNX model.

    In training mode, generates both TrainingNetwork (fwd+bwd) and
    OptimizerNetwork (SGD) into the same gen_dir.  Auto-detected training
    parameters (n_steps, n_accum, num_data_inputs) are written to
    gen_dir/training_meta.json and read back into config after codegen.

    Raises:
        RuntimeError: If network generation fails
    """
    if skip:
        log.info(f"Skipping network generation for {config.test_name}")
        return

    script_dir = Path(__file__).parent.parent.parent

    if config.training:
        # --- Step 1: Training network (forward + backward + accumulation) ---
        generation_script = script_dir / "generateTrainingNetwork.py"
        cmd = [
            sys.executable,
            str(generation_script),
            "-d", config.gen_dir,
            "-t", config.test_dir,
            "-p", config.platform,
        ]
        # Only pass values when explicitly set; otherwise let the script auto-detect
        if config.n_train_steps is not None:
            cmd.append(f"--n-steps={config.n_train_steps}")
        if config.n_accum_steps is not None:
            cmd.append(f"--n-accum={config.n_accum_steps}")
        if config.training_num_data_inputs is not None:
            cmd.append(f"--num-data-inputs={config.training_num_data_inputs}")

        if config.verbose > 0:
            cmd.append("-" + "v" * config.verbose)
        if config.debug:
            cmd.append("--debug")
        cmd.extend(config.gen_args)

        log.debug(f"[Execution] Training generation command: {' '.join(cmd)}")
        result = subprocess.run(cmd, check=False)
        if result.returncode != 0:
            raise RuntimeError(f"Training network generation failed for {config.test_name}")

        # Read back auto-detected values written by generateTrainingNetwork.py
        meta_path = Path(config.gen_dir) / "training_meta.json"
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
            config.n_train_steps = meta["n_train_steps"]
            config.n_accum_steps = meta["n_accum_steps"]
            config.training_num_data_inputs = meta["training_num_data_inputs"]
            log.info(f"[Execution] Training meta: {meta}")

        # --- Step 2: Optimizer network (SGD) ---
        opt_dir = _resolve_optimizer_dir(config)
        opt_script = script_dir / "generateOptimizerNetwork.py"

        if not Path(opt_dir).exists():
            log.warning(f"Optimizer directory not found: {opt_dir} — skipping optimizer codegen")
        elif not opt_script.exists():
            log.warning(f"generateOptimizerNetwork.py not found — skipping optimizer codegen")
        else:
            opt_cmd = [
                sys.executable,
                str(opt_script),
                "-d", config.gen_dir,
                "-t", opt_dir,
                "-p", config.platform,
            ]
            for arg in config.gen_args:
                if arg.startswith("--cores"):
                    opt_cmd.append(arg)
            if config.verbose > 0:
                opt_cmd.append("-" + "v" * config.verbose)

            log.debug(f"[Execution] Optimizer generation command: {' '.join(opt_cmd)}")
            result = subprocess.run(opt_cmd, check=False)
            if result.returncode != 0:
                raise RuntimeError(f"Optimizer network generation failed for {config.test_name}")

        return  # early return — training path complete

    elif config.tiling:
        generation_script = script_dir / "testMVP.py"
        cmd = [
            sys.executable,
            str(generation_script),
            "-d", config.gen_dir,
            "-t", config.test_dir,
            "-p", config.platform,
        ]
    else:
        generation_script = script_dir / "generateNetwork.py"
        cmd = [
            sys.executable,
            str(generation_script),
            "-d", config.gen_dir,
            "-t", config.test_dir,
            "-p", config.platform,
        ]

    if config.verbose > 0:
        cmd.append("-" + "v" * config.verbose)
    if config.debug:
        cmd.append("--debug")
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

    if config.training:
        cmd.append("-DTRAINING=ON")
        # Only add cmake defines when the values are known (after codegen)
        if config.n_train_steps is not None:
            cmd.append(f"-DN_TRAIN_STEPS={config.n_train_steps}")
        if config.n_accum_steps is not None:
            cmd.append(f"-DN_ACCUM_STEPS={config.n_accum_steps}")
        if config.training_num_data_inputs is not None:
            cmd.append(f"-DTRAINING_NUM_DATA_INPUTS={config.training_num_data_inputs}")

    script_dir = Path(__file__).parent.parent.parent
    cmd.append(str(script_dir.parent))

    env = _augment_path(os.environ.copy())
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
    env = _augment_path(os.environ.copy())
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

    env = _augment_path(os.environ.copy())
    if config.verbose >= 3:
        env["VERBOSE"] = "1"

    use_shell = True
    if config.simulator == 'host':
        binary_path = Path(config.build_dir) / "bin" / config.test_name
        cmd = [str(binary_path)]
        use_shell = False

    elif config.simulator == 'gvsoc' and config.platform != 'GAP9':
        # Run gvsoc directly instead of through cmake to avoid USES_TERMINAL
        # pipe buffering — cmake's USES_TERMINAL connects gvsoc to the real
        # terminal, bypassing our pipe and causing apparent hangs.
        gvsoc_exe = str(Path(env.get('GVSOC_INSTALL_DIR', '')) / 'bin' / 'gvsoc')
        binary_path = str(Path(config.build_dir) / 'bin' / config.test_name)
        workdir = str(Path(config.build_dir) / 'gvsoc_workdir')
        os.makedirs(workdir, exist_ok=True)
        # gvsoc target name is derived from cmake's add_gvsoc_emulation call.
        # For Siracusa it is always "siracusa".
        gvsoc_target = config.platform.lower().replace('_w_neureka', '')
        cmd = [
            gvsoc_exe,
            f"--target={gvsoc_target}",
            f"--binary={binary_path}",
            f"--work-dir={workdir}",
        ]
        hex_dir = Path(config.gen_dir) / 'hex'
        if hex_dir.is_dir():
            for hex_file in sorted(hex_dir.glob('*.hex')):
                cmd.append(f"--flash-property={hex_file}@hyperflash:readfs:files")
        cmd += ["image", "flash", "run"]
        # gvsoc_launcher (the C++ simulation binary spawned by gvsoc) uses fully-
        # buffered stdout when its stdout is a pipe, preventing real-time output.
        # Wrap in `script -q -e` to allocate a PTY, which forces line-buffered
        # output so the readline loop below receives data as it is generated.
        # Pass inner command as a properly-shell-quoted string to `script -c`
        # using shell=False so there is no double-quoting ambiguity.
        import shlex as _shlex
        gvsoc_inner = ' '.join(_shlex.quote(c) for c in cmd)
        cmd = ['script', '-q', '-e', '-c', gvsoc_inner, '/dev/null']
        use_shell = False

    elif config.simulator == 'banshee':
        if config.verbose == 1:
            env["BANSHEE_LOG"] = "warn"
        elif config.verbose == 2:
            env["BANSHEE_LOG"] = "info"
        elif config.verbose >= 3:
            env["BANSHEE_LOG"] = "debug"
        cmake_cmd = os.environ.get("CMAKE", "cmake")
        cmd = [cmake_cmd, "--build", config.build_dir, "--target",
               f"{config.simulator}_{config.test_name}"]

    else:
        cmake_cmd = os.environ.get("CMAKE", "cmake")
        cmd = [cmake_cmd, "--build", config.build_dir, "--target",
               f"{config.simulator}_{config.test_name}"]

    log.debug(f"[Execution] Simulation command: {' '.join(cmd)}")

    popen_cmd = cmd if not use_shell else " ".join(cmd)
    with subprocess.Popen(popen_cmd,
                          stdout = subprocess.PIPE,
                          stderr = subprocess.STDOUT,
                          shell = use_shell,
                          encoding = 'utf-8') as process:

        with open('out.txt', 'a', encoding = 'utf-8') as fileHandle:
            fileHandle.write(
                f"################## Testing {config.test_dir} on {config.platform} Platform ##################\n")

            result = ""
            while True:
                output = process.stdout.readline()
                if output == '' and process.poll() is not None:
                    break
                if output:
                    # Filter script(1) header/footer lines added by the PTY wrapper
                    if output.startswith('Script started on ') or output.startswith('Script done on '):
                        continue
                    print(output.rstrip('\r\n'))
                    result += output
                    fileHandle.write(f"{escapeAnsi(output)}")

    # Parse output for error count and cycles
    test_result = parse_test_output(result, "", returncode=process.returncode)

    if not test_result.success and test_result.error_count == -1:
        log.warning(f"Could not parse error count from output")

    return test_result


def run_complete_test(config: DeeployTestConfig, skipgen: bool = False, skipsim: bool = False) -> TestResult:
    """
    Run a complete test: generate, configure, build, and simulate.
    """
    log.info(f"################## Testing {config.test_name} on {config.platform} Platform ##################")

    generate_network(config, skip = skipgen)
    configure_cmake(config)
    build_binary(config)
    result = run_simulation(config, skip = skipsim)

    return result

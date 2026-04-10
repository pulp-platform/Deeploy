# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from Deeploy.Logging import DEFAULT_LOGGER as log

from .config import DeeployTestConfig
from .output_parser import TestResult, parse_test_output


def _augment_path(env: dict) -> dict:
    """Prepend gvsoc/llvm bin dirs to PATH based on installed env vars.

    The install dirs are already set as env vars (GVSOC_INSTALL_DIR,
    LLVM_INSTALL_DIR) but their bin/ subdirectories may not be in PATH.

    If a virtual environment is active (VIRTUAL_ENV is set), its bin dir
    is prepended so that shebang-invoked scripts (kconfigtool.py, gapy)
    resolve python3 to the venv interpreter, which has kconfiglib.
    Without this, /usr/bin/python3 would be picked up instead, which
    lacks kconfiglib and causes CMake kconfig setup to fail.
    """
    venv = env.get('VIRTUAL_ENV', '')
    extra = [str(Path(venv) / 'bin')] if venv else ['/usr/bin']
    for var in ('GVSOC_INSTALL_DIR', 'LLVM_INSTALL_DIR'):
        install_dir = env.get(var, '')
        if install_dir:
            bin_dir = str(Path(install_dir) / 'bin')
            current = env.get('PATH', '').split(':')
            if bin_dir not in current:
                extra.append(bin_dir)
    env['PATH'] = ':'.join(extra) + ':' + env.get('PATH', '')
    return env


def _resolve_optimizer_dir(config: DeeployTestConfig) -> str:
    """Return the optimizer ONNX directory for this config.

    Falls back to <test_dir>/../<model>_optimizer if not explicitly set,
    where <model> is derived by replacing the '_train' suffix of the test
    directory name with '_optimizer' (e.g. simplemlp_train → simplemlp_optimizer,
    sleepconvit_train → sleepconvit_optimizer).
    """
    if config.optimizer_dir:
        return config.optimizer_dir
    test_parent = Path(config.test_dir).parent
    test_dir_name = Path(config.test_dir).name
    optimizer_name = test_dir_name.replace("_train", "_optimizer")
    return str(test_parent / optimizer_name)


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

    if config.training and config.tiling:
        # --- Tiled training: testMVPTraining.py (tiling pipeline + training init) ---
        generation_script = script_dir / "testMVPTraining.py"
        cmd = [
            sys.executable,
            str(generation_script),
            "-d",
            config.gen_dir,
            "-t",
            config.test_dir,
            "-p",
            config.platform,
        ]
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

        log.debug(f"[Execution] Tiled training generation command: {' '.join(cmd)}")
        result = subprocess.run(cmd, check = False)
        if result.returncode != 0:
            raise RuntimeError(f"Tiled training network generation failed for {config.test_name}")

        # Read back auto-detected values written by testMVPTraining.py
        meta_path = Path(config.gen_dir) / "training_meta.json"
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
            config.n_train_steps = meta["n_train_steps"]
            config.n_accum_steps = meta["n_accum_steps"]
            config.training_num_data_inputs = meta["training_num_data_inputs"]
            log.info(f"[Execution] Training meta: {meta}")

        # --- Step 2: Tiled optimizer network (SGD via testMVPOptimizer.py) ---
        opt_dir = _resolve_optimizer_dir(config)
        opt_script = script_dir / "testMVPOptimizer.py"

        if not Path(opt_dir).exists():
            log.warning(f"Optimizer directory not found: {opt_dir} — skipping optimizer codegen")
        elif not opt_script.exists():
            log.warning(f"testMVPOptimizer.py not found — skipping optimizer codegen")
        else:
            opt_cmd = [
                sys.executable,
                str(opt_script),
                "-d",
                config.gen_dir,
                "-t",
                opt_dir,
                "-p",
                config.platform,
                f"--training-dir={config.test_dir}",
            ]
            _OPT_PASSTHROUGH = ("--cores", "--l1", "--l2", "--defaultMemLevel", "--memAllocStrategy",
                                "--searchStrategy", "--plotMemAlloc", "--profileTiling")
            for arg in config.gen_args:
                if any(arg.startswith(p) for p in _OPT_PASSTHROUGH):
                    opt_cmd.append(arg)
            # If no --defaultMemLevel was passed through, default to L2
            if not any(arg.startswith("--defaultMemLevel") for arg in opt_cmd):
                opt_cmd.append("--defaultMemLevel=L2")
            if config.verbose > 0:
                opt_cmd.append("-" + "v" * config.verbose)

            log.debug(f"[Execution] Tiled optimizer generation command: {' '.join(opt_cmd)}")
            result = subprocess.run(opt_cmd, check = False)
            if result.returncode != 0:
                raise RuntimeError(f"Tiled optimizer network generation failed for {config.test_name}")

        return  # early return — tiled training path complete

    elif config.training:
        # --- Step 1: Training network (forward + backward + accumulation) ---
        generation_script = script_dir / "generateTrainingNetwork.py"
        cmd = [
            sys.executable,
            str(generation_script),
            "-d",
            config.gen_dir,
            "-t",
            config.test_dir,
            "-p",
            config.platform,
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
        result = subprocess.run(cmd, check = False)
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
                "-d",
                config.gen_dir,
                "-t",
                opt_dir,
                "-p",
                config.platform,
                f"--training-dir={config.test_dir}",
            ]
            _OPT_PASSTHROUGH = ("--cores", "--l1", "--l2", "--defaultMemLevel")
            for arg in config.gen_args:
                if any(arg.startswith(p) for p in _OPT_PASSTHROUGH):
                    opt_cmd.append(arg)
            if not any(arg.startswith("--defaultMemLevel") for arg in opt_cmd):
                opt_cmd.append("--defaultMemLevel=L2")
            if config.verbose > 0:
                opt_cmd.append("-" + "v" * config.verbose)

            log.debug(f"[Execution] Optimizer generation command: {' '.join(opt_cmd)}")
            result = subprocess.run(opt_cmd, check = False)
            if result.returncode != 0:
                raise RuntimeError(f"Optimizer network generation failed for {config.test_name}")

        return  # early return — training path complete

    elif config.tiling:
        generation_script = script_dir / "testMVP.py"
        cmd = [
            sys.executable,
            str(generation_script),
            "-d",
            config.gen_dir,
            "-t",
            config.test_dir,
            "-p",
            config.platform,
        ]
    else:
        generation_script = script_dir / "generateNetwork.py"
        cmd = [
            sys.executable,
            str(generation_script),
            "-d",
            config.gen_dir,
            "-t",
            config.test_dir,
            "-p",
            config.platform,
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
    else:
        cmd.append("-DTRAINING=OFF")

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
    if config.verbose >= 3:
        env["VERBOSE"] = "1"

    log.debug(f"[Execution] Build command: {' '.join(cmd)}")

    result = subprocess.run(cmd, check = False, env = env)

    if result.returncode != 0:
        log.error(f"Build failed with return code {result.returncode}")
        raise RuntimeError(f"Build failed for {config.test_name}")


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

    if config.simulator == 'host':
        binary_path = Path(config.build_dir) / "bin" / config.test_name
        cmd = [str(binary_path)]

    elif config.simulator == 'gvsoc':
        cmake_cmd = os.environ.get("CMAKE", "cmake")
        cmd = [cmake_cmd, "--build", config.build_dir, "--target", f"gvsoc_{config.test_name}"]

    elif config.simulator == 'banshee':
        if config.verbose == 1:
            env["BANSHEE_LOG"] = "warn"
        elif config.verbose == 2:
            env["BANSHEE_LOG"] = "info"
        elif config.verbose >= 3:
            env["BANSHEE_LOG"] = "debug"
        cmake_cmd = os.environ.get("CMAKE", "cmake")
        cmd = [cmake_cmd, "--build", config.build_dir, "--target", f"{config.simulator}_{config.test_name}"]

    else:
        cmake_cmd = os.environ.get("CMAKE", "cmake")
        cmd = [cmake_cmd, "--build", config.build_dir, "--target", f"{config.simulator}_{config.test_name}"]

    log.debug(f"[Execution] Simulation command: {' '.join(cmd)}")

    # Stream output in real-time (line-buffered) and capture for parsing.
    proc = subprocess.Popen(cmd,
                            stdout = subprocess.PIPE,
                            stderr = subprocess.STDOUT,
                            text = True,
                            env = env,
                            bufsize = 1)
    stdout_lines = []
    for line in proc.stdout:
        print(line, end = '', flush = True)
        stdout_lines.append(line)
    proc.stdout.close()
    proc.wait()
    stdout_output = ''.join(stdout_lines)

    test_result = parse_test_output(stdout_output, '')

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

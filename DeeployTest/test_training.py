# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0
"""
Unified pytest suite for all training-related tests on Siracusa and GAP9:
  - test_training            : non-tiled model training
  - test_training_tiled      : tiled model training (SBTiler)
  - test_grad_tiled          : tiled gradient-operator kernel tests

Run all training tests:
    pytest test_training.py -v

Filter by platform:
    pytest test_training.py -m siracusa_train -v
    pytest test_training.py -m gap9_train -v
    pytest test_training.py -m siracusa_train_tiled -v
    pytest test_training.py -m gap9_train_tiled -v
    pytest test_training.py -m siracusa_grad_tiled -v
    pytest test_training.py -m gap9_grad_tiled -v

Filter by model/kernel:
    pytest test_training.py -k "simplemlp_train" -v
    pytest test_training.py -k "ReluGrad" -v

Skip simulation (codegen + build only):
    pytest test_training.py --skipsim -v
"""

import pytest

from test_training_config import DEFAULT_CORES, GRAD_TILED_TESTS, SIMULATOR, TRAINING_TESTS, TRAINING_TILED_TESTS
from testUtils.pytestRunner import create_test_config, create_training_test_config, run_and_assert_test

_TRAIN_PLATFORMS = [
    ("Siracusa", pytest.mark.siracusa_train),
    ("GAP9", pytest.mark.gap9_train),
]

_TRAIN_TILED_PLATFORMS = [
    ("Siracusa", pytest.mark.siracusa_train_tiled),
    ("GAP9", pytest.mark.gap9_train_tiled),
]

_GRAD_PLATFORMS = [
    ("Siracusa", pytest.mark.siracusa_grad_tiled),
    ("GAP9", pytest.mark.gap9_grad_tiled),
]


# ── Non-tiled model training ──────────────────────────────────────────────────

def _training_params():
    params = []
    for platform, mark in _TRAIN_PLATFORMS:
        for test_name, cfg in TRAINING_TESTS[platform].items():
            params.append(pytest.param((platform, test_name, cfg), marks=mark, id=f"{platform}-{test_name}"))
    return params


@pytest.mark.models
@pytest.mark.parametrize("test_cfg", _training_params())
def test_training(test_cfg, deeploy_test_dir, toolchain, toolchain_dir, cmake_args, skipgen, skipsim) -> None:
    platform, test_name, params = test_cfg

    config = create_training_test_config(
        test_name = test_name,
        platform = platform,
        simulator = SIMULATOR,
        deeploy_test_dir = str(deeploy_test_dir),
        toolchain = toolchain,
        toolchain_dir = toolchain_dir,
        cmake_args = cmake_args,
        cores = DEFAULT_CORES,
        n_train_steps = params["n_train_steps"],
        n_accum_steps = params["n_accum_steps"],
        training_num_data_inputs = params["num_data_inputs"],
        optimizer_dir = params.get("optimizer_dir"),
    )

    run_and_assert_test(test_name, config, skipgen = skipgen, skipsim = skipsim)


# ── Tiled model training ──────────────────────────────────────────────────────

def _training_tiled_params():
    params = []
    for platform, mark in _TRAIN_TILED_PLATFORMS:
        for test_name, cfg in TRAINING_TILED_TESTS.items():
            params.append(pytest.param((platform, test_name, cfg), marks=mark, id=f"{platform}-{test_name}"))
    return params


@pytest.mark.models
@pytest.mark.parametrize("test_cfg", _training_tiled_params())
def test_training_tiled(test_cfg, deeploy_test_dir, toolchain, toolchain_dir, cmake_args, skipgen,
                        skipsim) -> None:
    platform, test_name, params = test_cfg

    config = create_training_test_config(
        test_name = test_name,
        platform = platform,
        simulator = SIMULATOR,
        deeploy_test_dir = str(deeploy_test_dir),
        toolchain = toolchain,
        toolchain_dir = toolchain_dir,
        cmake_args = cmake_args,
        cores = DEFAULT_CORES,
        n_train_steps = params["n_train_steps"],
        n_accum_steps = params["n_accum_steps"],
        training_num_data_inputs = params["num_data_inputs"],
        optimizer_dir = params.get("optimizer_dir"),
        tiling = True,
    )

    run_and_assert_test(test_name, config, skipgen = skipgen, skipsim = skipsim)


# ── Tiled gradient-operator kernel tests ─────────────────────────────────────

def _grad_tiled_params():
    params = []
    for platform, mark in _GRAD_PLATFORMS:
        for test_name, l1_list in GRAD_TILED_TESTS.items():
            for l1 in l1_list:
                params.append(
                    pytest.param((platform, test_name, l1),
                                 marks=mark,
                                 id=f"{platform}-{test_name}-{l1}-L2-singlebuffer"))
    return params


@pytest.mark.kernels
@pytest.mark.singlebuffer
@pytest.mark.l2
@pytest.mark.parametrize("test_cfg", _grad_tiled_params())
def test_grad_tiled(test_cfg, deeploy_test_dir, toolchain, toolchain_dir, cmake_args, skipgen, skipsim) -> None:
    platform, test_name, l1 = test_cfg

    config = create_test_config(
        test_name = test_name,
        platform = platform,
        simulator = SIMULATOR,
        deeploy_test_dir = str(deeploy_test_dir),
        toolchain = toolchain,
        toolchain_dir = toolchain_dir,
        cmake_args = cmake_args,
        tiling = True,
        cores = DEFAULT_CORES,
        l1 = l1,
        default_mem_level = "L2",
        double_buffer = False,
    )

    run_and_assert_test(test_name, config, skipgen = skipgen, skipsim = skipsim)

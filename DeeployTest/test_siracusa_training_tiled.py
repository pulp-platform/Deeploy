# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0
"""
Pytest suite for Siracusa tiled training tests (forward + backward + optimizer, SBTiler).

Run all:
    pytest test_siracusa_training_tiled.py -m siracusa_train_tiled -v

Run one test:
    pytest test_siracusa_training_tiled.py -k "tinytransformer_train" -v

Skip simulation (codegen + build only):
    pytest test_siracusa_training_tiled.py --skipsim -v
"""

import pytest

from test_siracusa_train_tiled_config import DEFAULT_CORES, PLATFORM_NAME, SIMULATOR, TRAINING_TILED_TESTS
from testUtils.pytestRunner import create_training_test_config, run_and_assert_test


@pytest.mark.siracusa_train_tiled
@pytest.mark.models
@pytest.mark.parametrize("test_name", list(TRAINING_TILED_TESTS.keys()), ids = list(TRAINING_TILED_TESTS.keys()))
def test_siracusa_training_tiled(test_name, deeploy_test_dir, toolchain, toolchain_dir, cmake_args, skipgen,
                                 skipsim) -> None:
    params = TRAINING_TILED_TESTS[test_name]

    config = create_training_test_config(
        test_name = test_name,
        platform = PLATFORM_NAME,
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

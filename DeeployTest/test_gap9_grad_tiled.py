# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0
"""
Pytest suite for GAP9 tiled gradient-operator kernel tests (single backward-pass ops, SBTiler).

Run all:
    pytest test_gap9_grad_tiled.py -m gap9_grad_tiled -v

Run one operator group:
    pytest test_gap9_grad_tiled.py -k "SoftmaxGrad" -v

Skip simulation (codegen + build only):
    pytest test_gap9_grad_tiled.py --skipsim -v
"""

import pytest

from test_gap9_grad_tiled_config import DEFAULT_CORES, L2_SINGLEBUFFER_GRAD_KERNELS, PLATFORM_NAME, SIMULATOR
from testUtils.pytestRunner import create_test_config, run_and_assert_test


def _generate_params(test_dict, config_name):
    """Expand {test_name: [l1, ...]} into (test_name, l1, config_name) tuples."""
    return [(name, l1, config_name) for name, l1_list in test_dict.items() for l1 in l1_list]


def _param_id(param):
    test_name, l1, config = param
    return f"{test_name}-{l1}-{config}"


@pytest.mark.gap9_grad_tiled
@pytest.mark.kernels
@pytest.mark.singlebuffer
@pytest.mark.l2
@pytest.mark.parametrize(
    "test_params",
    _generate_params(L2_SINGLEBUFFER_GRAD_KERNELS, "L2-singlebuffer"),
    ids = _param_id,
)
def test_gap9_grad_tiled_kernels_l2_singlebuffer(test_params, deeploy_test_dir, toolchain, toolchain_dir,
                                                  cmake_args, skipgen, skipsim) -> None:
    test_name, l1, _ = test_params
    config = create_test_config(
        test_name = test_name,
        platform = PLATFORM_NAME,
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

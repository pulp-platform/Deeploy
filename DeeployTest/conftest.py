# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

import os
from pathlib import Path

import coloredlogs
import pytest

from Deeploy.Logging import DEFAULT_FMT, DEFAULT_LOGGER as log


def pytest_addoption(parser: pytest.Parser) -> None:
    """Native PyTest hook: add custom command-line options for Deeploy tests."""
    parser.addoption(
        "--skipgen",
        action="store_true",
        default=False,
        help="Skip network generation step",
    )
    parser.addoption(
        "--skipsim",
        action="store_true",
        default=False,
        help="Skip simulation step (only generate and build)",
    )
    parser.addoption(
        "--toolchain",
        action="store",
        default="LLVM",
        help="Compiler toolchain to use (LLVM or GCC)",
    )
    parser.addoption(
        "--toolchain-install-dir",
        action="store",
        default=os.environ.get("LLVM_INSTALL_DIR"),
        help="Path to toolchain installation directory",
    )
    parser.addoption(
        "--cmake-args",
        action="append",
        default=[],
        help="Additional CMake arguments (can be used multiple times)",
    )

def pytest_configure(config: pytest.Config) -> None:
    """Native PyTest hook: configure pytest for Deeploy tests."""
    # Register custom markers
    config.addinivalue_line(
        "markers", "generic: mark test as a Generic platform test"
    )
    config.addinivalue_line(
        "markers", "kernels: mark test as a kernel test (individual operators)"
    )
    config.addinivalue_line(
        "markers", "models: mark test as a model test (full networks)"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
    
    # Configure logging based on verbosity
    verbosity = config.option.verbose
    if verbosity >= 3:
        coloredlogs.install(level='DEBUG', logger=log, fmt=DEFAULT_FMT)
    elif verbosity >= 2:
        coloredlogs.install(level='INFO', logger=log, fmt=DEFAULT_FMT)
    else:
        coloredlogs.install(level='WARNING', logger=log, fmt=DEFAULT_FMT)

@pytest.fixture(scope="session")
def deeploy_test_dir():
    """Return the DeeployTest directory path."""
    return Path(__file__).parent

@pytest.fixture(scope="session")
def tests_dir(deeploy_test_dir):
    """Return the Tests directory path."""
    return deeploy_test_dir / "Tests"

@pytest.fixture(scope="session")
def toolchain_dir(request):
    """Return the toolchain installation directory."""
    toolchain_install = request.config.getoption("--toolchain-install-dir")
    if toolchain_install is None:
        pytest.skip(reason="LLVM_INSTALL_DIR not set")
    return toolchain_install

@pytest.fixture(scope="session")
def ccache_dir():
    """Setup and return ccache directory."""
    ccache_path = Path("/app/.ccache")
    if ccache_path.exists():
        os.environ["CCACHE_DIR"] = str(ccache_path)
        return ccache_path
    return None

@pytest.fixture
def skipgen(request):
    """Return whether to skip network generation."""
    return request.config.getoption("--skipgen")

@pytest.fixture
def skipsim(request):
    """Return whether to skip simulation."""
    return request.config.getoption("--skipsim")

@pytest.fixture
def toolchain(request):
    """Return the toolchain to use."""
    return request.config.getoption("--toolchain")

@pytest.fixture
def cmake_args(request):
    """Return additional CMake arguments."""
    return request.config.getoption("--cmake-args")

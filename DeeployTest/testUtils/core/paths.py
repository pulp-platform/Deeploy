# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from typing import Optional, Tuple

from Deeploy.Logging import DEFAULT_LOGGER as log


def get_test_paths(test_dir: str, platform: str, base_dir: Optional[str] = None) -> Tuple[str, str, str]:
    """
    Resolve test paths for generation and build directories.

    Args:
        test_dir: Path to test directory (e.g., "Tests/Adder" or absolute path)
        platform: Platform name (e.g., "Generic")
        base_dir: Base directory for tests (defaults to DeeployTest/)

    Returns:
        Tuple of (gen_dir, test_dir_abs, test_name)
    """
    if base_dir is None:
        # Get the absolute path of this script's parent directory (core -> testUtils -> DeeployTest)
        script_path = Path(__file__).resolve()
        base_dir = script_path.parent.parent.parent
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

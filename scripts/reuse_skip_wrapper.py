#!/usr/bin/env python

# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

import argparse
import os
import subprocess


def skip(file: str) -> bool:
    if os.path.basename(file) in [
            ".clang-format",
            ".devcontainer.json",
            ".gitignore",
            ".gitmodules",
            ".isort.cfg",
            ".pre-commit-config.yaml",
            ".style.yapf",
            ".yamllint",
            ".yapfignore",
            "Deeploy.code-workspace",
            "pyproject.toml",
            "requirements-dev.txt",
            "setup.py",
            "CODEOWNERS",
    ]:
        return True

    # Skip license directory
    if "LICENSES" in os.path.split(file):
        return True

    return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog = "license-fix.py",
                                     usage = "%(prog)s [OPTIONS] FILE ...",
                                     description = "Helper script to fix the licenses with pre-commit")
    parser.add_argument("files", nargs = "+")

    args = parser.parse_args()

    files = [f for f in args.files if not skip(f)]
    try:
        subprocess.run(f"reuse lint-file {' '.join(files)}", shell = True, check = True)
    except subprocess.CalledProcessError:
        exit(1)

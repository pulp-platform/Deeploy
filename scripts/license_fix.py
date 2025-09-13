#!/usr/bin/env python

# Copyright (C) 2025, ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

import argparse
import os
import sys
from datetime import date
from typing import Callable, Sequence, Tuple

LICENSE_HEADER = f"""Copyright (C) {date.today().year}, ETH Zurich and University of Bologna.
Licensed under the Apache License, Version 2.0, see LICENSE for details.
SPDX-License-Identifier: Apache-2.0
"""

CommentOut = Callable[[str], str]
RemoveOldLicenseHeader = Callable[[str], str]


def _remove_old_license(lines: Sequence[str], end_idx: int) -> str:
    if any("Apache-2.0" in l for l in lines[:end_idx]):
        # Skip empty lines
        for line in lines[end_idx:]:
            if len(line.strip()) == 0:
                end_idx += 1
            else:
                break
        return "".join(lines[end_idx:])
    else:
        return "".join(lines)


def rst_comment_out(text: str) -> str:
    return "..\n" + "".join(f"   {line}" for line in text.splitlines(keepends = True))


def rst_remove_old_license_header(text: str) -> str:
    lines = text.splitlines(keepends = True)

    end_idx = 0
    if lines[0].strip() == "..":
        for line in lines[1:]:
            if not line.startswith("   "):
                break
            end_idx += 1

    return _remove_old_license(lines, end_idx)


def html_comment_out(text: str) -> str:
    return "<!--\n" + text + "-->\n"


def html_remove_old_license_header(text: str) -> str:
    lines = text.splitlines(keepends = True)

    end_idx = 0
    comment_ended = False
    if lines[0].strip().startswith("<!--"):
        for line in lines:
            end_idx += 1
            if line.strip().endswith("-->"):
                comment_ended = True
                break

    if not comment_ended:
        end_idx = 0

    return _remove_old_license(lines, end_idx)


def hash_comment_out(text: str) -> str:
    return "".join(f"# {line}" for line in text.splitlines(keepends = True))


def hash_remove_old_license_header(text: str) -> str:
    lines = text.splitlines(keepends = True)

    end_idx = 0
    for line in lines:
        if not line.strip().startswith("#"):
            break
        end_idx += 1

    return _remove_old_license(lines, end_idx)


def c_comment_out(text: str) -> str:
    return "/*\n" + "".join(f" * {line}" for line in text.splitlines(keepends = True)) + " */\n"


def c_remove_old_license_header(text: str) -> str:
    lines = text.splitlines(keepends = True)

    end_idx = 0
    comment_ended = False
    if lines[0].strip().startswith("/*"):
        for line in lines:
            end_idx += 1
            if line.strip().endswith("*/"):
                comment_ended = True
                break

    if not comment_ended:
        end_idx = 0

    return _remove_old_license(lines, end_idx)


def map_filename(filename: str) -> Tuple[CommentOut, RemoveOldLicenseHeader]:
    _, ext = os.path.splitext(filename)

    if ext in [".py", ".yml", ".yaml", ".cmake"] or filename == "CMakeLists.txt" or filename == "Makefile":
        return hash_comment_out, hash_remove_old_license_header
    elif ext in [".c", ".h", ".cpp", ".hpp"]:
        return c_comment_out, c_remove_old_license_header
    elif ext in [".rst"]:
        return rst_comment_out, rst_remove_old_license_header
    elif ext in [".html"]:
        return html_comment_out, html_remove_old_license_header
    else:
        raise RuntimeError(f"Unrecognized extension \"{ext}\" of file {filename}. "
                           f"Either implement the {ext[1:]}_comment_out and {ext[1:]}_remove_old_license_header "
                           "or add the file to the list of skipped files in the skip() function.")


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
            "LICENSE",
            "pyproject.toml",
            "requirements-dev.txt",
            "setup.py",
            "CODEOWNERS",
    ]:
        return True

    # Skip runable scripts, i.e., that start with a shebang. Currently, it's only files in _scripts_ folder
    if "scripts" in os.path.split(file):
        with open(file, "r") as f:
            first_line = f.readline()
        if first_line.startswith("#!"):
            return True

    if "Dockerfile" in os.path.basename(file):
        return True

    return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog = "license-fix.py",
                                     usage = "%(prog)s [OPTIONS] FILE ...",
                                     description = "Script to fix the licenses")
    parser.add_argument(
        "--check",
        action = 'store_true',
        default = False,
        help =
        "Doesn't change any files but just returns an error code 1 if license headers need to be fixed, otherwise 0.")
    parser.add_argument("--dry-run",
                        action = 'store_true',
                        default = False,
                        help = "Prints the fixed file(s) to stdout")
    parser.add_argument("files", nargs = "+")

    args = parser.parse_args()

    updated_files = []

    for file in args.files:
        if skip(file):
            continue

        comment_out, remove_old_license_header = map_filename(os.path.basename(file))

        commented_license = comment_out(LICENSE_HEADER)

        with open(file, "r") as f:
            filetext = f.read()

        if filetext.startswith(commented_license):
            continue

        filetext = commented_license + "\n" + remove_old_license_header(filetext)

        if not args.check:
            if args.dry_run:
                print(filetext)
            else:
                with open(file, "w") as f:
                    f.write(filetext)

        updated_files.append(file)

    if args.check and len(updated_files) != 0:
        print(f"Check failed. Files that need their license header fixed:\n{', '.join(updated_files)}",
              file = sys.stderr)
        exit(1)

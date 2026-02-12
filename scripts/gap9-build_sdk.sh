#!/usr/bin/env bash

# SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

# gap9-build_sdk.sh
# Helper script to clone, patch and build the GAP9 SDK. Intended to be
# invoked from the Makefile with environment variables set:
#   GAP9_SDK_INSTALL_DIR (required)
#   GAP9_SDK_COMMIT_HASH (optional, fallback provided)
#   ROOT_DIR (optional, defaults to script dir)

ROOT_DIR="${ROOT_DIR:-$(cd "$(dirname "$0")" && pwd)}"
GAP9_SDK_INSTALL_DIR="${GAP9_SDK_INSTALL_DIR:?GAP9_SDK_INSTALL_DIR must be set}"
GAP9_SDK_COMMIT_HASH="${GAP9_SDK_COMMIT_HASH:-897955d7ab326bd31684429eb16a2e485ab89afb}"
GAP_SDK_URL="${GAP_SDK_URL:-'git@iis-git.ee.ethz.ch:wiesep/gap9_sdk.git'}"

echo "Preparing GAP9 SDK in: ${GAP9_SDK_INSTALL_DIR}"

if [ -d "${GAP9_SDK_INSTALL_DIR}/.git" ]; then
	echo "Directory ${GAP9_SDK_INSTALL_DIR} already exists and looks like a git repo. Updating remote URL and fetching latest changes..."
	git remote set-url origin "${GAP_SDK_URL}" || true
else
	echo "Cloning GAP9 SDK..."
	git clone "${GAP_SDK_URL}" "${GAP9_SDK_INSTALL_DIR}"
fi

cd "${GAP9_SDK_INSTALL_DIR}"
echo "Checking out commit ${GAP9_SDK_COMMIT_HASH} (stash and fetch if necessary)"
git fetch --all --tags || true
git stash || true
git checkout "${GAP9_SDK_COMMIT_HASH}"
git submodule update --init --recursive

# Platform specific patch
ARCH=$(dpkg --print-architecture 2>/dev/null || true)
if [ -z "$ARCH" ]; then
	ARCH=$(uname -m)
fi
case "$ARCH" in
amd64 | x86_64) PATCH=gap9-amd64.patch ;;
arm64 | aarch64) PATCH=gap9-arm64.patch ;;
*) PATCH= ;;
esac

set -e # Enable strict error handling for the build process
if [ -n "$PATCH" ] && [ -f "${ROOT_DIR}/${PATCH}" ]; then
	echo "Applying platform patch: $PATCH"
	git apply "${ROOT_DIR}/${PATCH}"
else
	echo "No platform-specific patch to apply for architecture '$ARCH' (looked for ${ROOT_DIR}/${PATCH})"
fi
set +e # Disable strict error handling to allow deactivation even if build fails

echo "Setting up Python virtual environment and installing dependencies"
python -m venv .gap9-venv
. .gap9-venv/bin/activate
pip install "numpy<2.0.0"
echo "Sourcing GAP9 SDK environment"
. configs/gap9_evk_audio.sh || true

echo "Invoking make install_dependency cmake_sdk.build"
set -e # Enable strict error handling for the build process
make install_dependency cmake_sdk.build openocd.all
set +e # Disable strict error handling to allow deactivation even if build fails

deactivate

echo "GAP9 SDK ready at: ${GAP9_SDK_INSTALL_DIR}"

exit 0

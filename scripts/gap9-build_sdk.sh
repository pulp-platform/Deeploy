#!/usr/bin/env bash

# SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

# gap9-build_sdk.sh
#
# Clone, prepare, patch, and build a GAP9 SDK checkout in place.
# This script is intended to be called from the project Makefile.
#
# Environment variables:
#   GAP9_SDK_INSTALL_DIR  Required. Target directory for the SDK checkout/build.
#   GAP9_SDK_COMMIT_HASH  Required. Commit to checkout.
#   GAP_SDK_URL           Required. Git remote used for fetching the SDK.
#   ROOT_DIR              Optional. Base directory for patch lookup.
#                         Default: directory of this script.

ROOT_DIR="${ROOT_DIR:-$(cd "$(dirname "$0")" && pwd)}"
GAP9_SDK_INSTALL_DIR="${GAP9_SDK_INSTALL_DIR:?GAP9_SDK_INSTALL_DIR must be set}"
GAP9_SDK_COMMIT_HASH="${GAP9_SDK_COMMIT_HASH:?GAP9_SDK_COMMIT_HASH must be set}"
GAP_SDK_URL="${GAP_SDK_URL:?GAP_SDK_URL must be set}"

echo "Preparing GAP9 SDK in: ${GAP9_SDK_INSTALL_DIR}"
mkdir -p "${GAP9_SDK_INSTALL_DIR}"
# Fail if CD fails, e.g. due to missing permissions or if the path is a file.
cd "${GAP9_SDK_INSTALL_DIR}" || exit 1

if [ ! -d ".git" ]; then
	# Support reusing a pre-created directory that is not yet a git repo.
	echo "Directory exists but .git folder is missing. Reinitializing git repository..."
	git init
	git remote add origin "${GAP_SDK_URL}"
	git fetch --depth=1 origin ${GAP9_SDK_COMMIT_HASH}
	GIT_LFS_SKIP_SMUDGE=1 git reset --hard "${GAP9_SDK_COMMIT_HASH}"
else
	# Keep local uncommitted changes out of the way before checkout.
	echo "Setting remote URL and fetching latest changes..."
	git remote set-url origin "${GAP_SDK_URL}"
	git fetch --depth=1 origin ${GAP9_SDK_COMMIT_HASH}
	git add .
	git stash || true # Stash may fail if there are no changes, ignore that.
fi

echo "Checking out commit ${GAP9_SDK_COMMIT_HASH} (stash and fetch if necessary)"
GIT_LFS_SKIP_SMUDGE=1 git checkout "${GAP9_SDK_COMMIT_HASH}"
git submodule update --init --recursive
git lfs pull --include="*.so"

# Select platform patch by Debian arch first, then uname fallback.
ARCH=$(dpkg --print-architecture 2>/dev/null || true)
if [ -z "$ARCH" ]; then
	ARCH=$(uname -m)
fi
case "$ARCH" in
amd64 | x86_64) PATCH=gap9-amd64.patch ;;
arm64 | aarch64) PATCH=gap9-arm64.patch ;;
*) PATCH= ;;
esac

set -e # Fail fast for patch/build operations.
if [ -n "$PATCH" ]; then
	if [ -f "${ROOT_DIR}/${PATCH}" ]; then
		echo "Applying platform patch from ${ROOT_DIR}: $PATCH"
		git apply "${ROOT_DIR}/${PATCH}"
	elif [ -f "${ROOT_DIR}/Container/${PATCH}" ]; then
		echo "Applying platform patch from ${ROOT_DIR}/Container: $PATCH"
		git apply "${ROOT_DIR}/Container/${PATCH}"
	else
		echo "No platform-specific patch to apply for architecture '$ARCH' (looked for ${ROOT_DIR}/${PATCH} and ${ROOT_DIR}/Container/${PATCH})"
	fi
else
	echo "No platform-specific patch to apply for architecture '$ARCH'"
fi

set +e # Relax mode so cleanup/deactivate still runs on non-critical failures.

echo "Setting up Python virtual environment and installing dependencies"
python -m venv .gap9-venv
. .gap9-venv/bin/activate
pip install "numpy<2.0.0"
echo "Sourcing GAP9 SDK environment"
. configs/gap9_evk_audio.sh || true

echo "Invoking make install_dependency cmake_sdk.build"
set -e # Build must stop immediately on errors.
make install_dependency cmake_sdk.build openocd.all autotiler.all
set +e # Allow deactivation even if previous step failed.

deactivate

echo "GAP9 SDK ready at: ${GAP9_SDK_INSTALL_DIR}"

exit 0

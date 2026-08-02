#!/bin/bash
# SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0
# ----------------------------------------------------------------------
# deploy.sh - apply the TA solution into the live Deeploy source tree.
#
# Run from this directory (.../Deeploy/Tutorials/PartIII_solution/iLeakyReLU/).
#
# Usage:
#   ./deploy.sh           # apply scalar kernel (Step 3)
#   ./deploy.sh simd      # apply SIMD kernel (Step 6b) on top
#   ./deploy.sh undo      # revert the patch and remove the copied files
# ----------------------------------------------------------------------
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"

MODE="${1:-scalar}"

PATCH="$HERE/iLeakyReLU-core.patch"
if [ ! -f "$PATCH" ]; then
	echo "ERROR: $PATCH not found - the solution directory is incomplete." >&2
	exit 1
fi
TEMPLATES_DIR="$ROOT/Deeploy/Targets/PULPOpen/Templates"
TILECONSTR_DIR="$ROOT/Deeploy/Targets/PULPOpen/TileConstraints"
KERNEL_SRC_DIR="$ROOT/TargetLibraries/PULPOpen/src"
KERNEL_INC_DIR="$ROOT/TargetLibraries/PULPOpen/inc/kernel"
TESTS_DIR="$ROOT/DeeployTest/Tests/Kernels/Integer/LeakyReLU/Regular"
TEST_ARTIFACTS="network.onnx inputs.npz outputs.npz"

case "$MODE" in
undo)
	echo "Undoing iLeakyReLU additions..."
	if git -C "$ROOT" apply --reverse --check "$PATCH" 2>/dev/null; then
		git -C "$ROOT" apply --reverse "$PATCH"
		echo "  reverted the core-library patch"
	else
		echo "  core-library patch is not cleanly applied - leaving those files alone"
	fi
	# Remove only the artifacts we copied in, then prune the directories we
	# created - but only while they are empty, so anything a student put
	# alongside them survives.
	for f in $TEST_ARTIFACTS; do
		rm -f "$TESTS_DIR/$f"
	done
	rmdir "$TESTS_DIR" "$(dirname "$TESTS_DIR")" 2>/dev/null || true
	rm -f "$KERNEL_SRC_DIR/iLeakyReLU.c"
	rm -f "$KERNEL_INC_DIR/iLeakyReLU.h"
	rm -f "$TEMPLATES_DIR/iLeakyReLUTemplate.py"
	rm -f "$TILECONSTR_DIR/iLeakyReLUTileConstraint.py"
	echo "If the tree still isn't clean, use:"
	echo "  git -C \"$ROOT\" checkout -- Deeploy/Targets TargetLibraries/PULPOpen/inc/DeeployPULPMath.h"
	exit 0
	;;
scalar | simd) ;;
*)
	echo "Unknown mode '$MODE'. Try: scalar | simd | undo"
	exit 1
	;;
esac

echo "[1/5] Copy test artifacts -> $TESTS_DIR"
mkdir -p "$TESTS_DIR"
for f in $TEST_ARTIFACTS; do
	if [ ! -f "$HERE/$f" ]; then
		echo "  ERROR: $f not found in $HERE - run 'python generate.py' first." >&2
		exit 1
	fi
	cp "$HERE/$f" "$TESTS_DIR/"
done

echo "[2/5] Copy kernel header -> $KERNEL_INC_DIR/iLeakyReLU.h"
cp "$HERE/iLeakyReLU.h" "$KERNEL_INC_DIR/iLeakyReLU.h"

echo "[3/5] Copy kernel source ($MODE) -> $KERNEL_SRC_DIR/iLeakyReLU.c"
if [ "$MODE" = "simd" ]; then
	cp "$HERE/iLeakyReLU_simd.c" "$KERNEL_SRC_DIR/iLeakyReLU.c"
else
	cp "$HERE/iLeakyReLU.c" "$KERNEL_SRC_DIR/iLeakyReLU.c"
fi

echo "[4/5] Copy template + tile constraint"
cp "$HERE/iLeakyReLUTemplate.py" "$TEMPLATES_DIR/iLeakyReLUTemplate.py"
cp "$HERE/iLeakyReLUTileConstraint.py" "$TILECONSTR_DIR/iLeakyReLUTileConstraint.py"

echo "[5/5] Apply the core-library patch"
if git -C "$ROOT" apply --reverse --check "$PATCH" 2>/dev/null; then
	echo "  already applied - skipping"
elif git -C "$ROOT" apply "$PATCH" 2>/dev/null; then
	# Plain apply leaves the edits unstaged, exactly as a hand edit would.
	echo "  applied cleanly"
elif git -C "$ROOT" apply --3way "$PATCH"; then
	# --3way merged it, and also staged the result.
	echo "  applied via 3-way merge - check 'git diff --cached' before continuing"
else
	echo "" >&2
	echo "  ERROR: could not apply $(basename "$PATCH")." >&2
	echo "  Either Deeploy's sources moved since this patch was written, or you have" >&2
	echo "  uncommitted edits to the files it touches." >&2
	echo "  If git reported 'with conflicts' above, the files now carry <<<<<<< markers:" >&2
	echo "  resolve them by hand and re-run - everything else has already been merged." >&2
	echo "  To start over:  git -C \"$ROOT\" checkout -- Deeploy/Targets TargetLibraries/PULPOpen/inc/DeeployPULPMath.h" >&2
	exit 1
fi

echo
echo "Done. Now run the verification tests from DeeployTest/:"
echo "  python deeployRunner_siracusa.py        -t Tests/Kernels/Integer/LeakyReLU/Regular --cores=1                  # baseline"
echo "  python deeployRunner_siracusa.py        -t Tests/Kernels/Integer/LeakyReLU/Regular --cores=8                  # Step 4"
echo "  python deeployRunner_tiled_siracusa.py  -t Tests/Kernels/Integer/LeakyReLU/Regular --cores=8 --l1=32768 --defaultMemLevel=L2   # Step 5/6"

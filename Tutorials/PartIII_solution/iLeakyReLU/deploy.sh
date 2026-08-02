#!/bin/bash
# SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0
# ----------------------------------------------------------------------
# deploy.sh - apply the TA solution into the live Deeploy source tree.
#
# Run from this directory (.../Deeploy/Tutorials/PartIII_solution/iLeakyReLU/).
# Idempotent for the file copies; the source patches use grep guards so
# they only apply once.
#
# Usage:
#   ./deploy.sh           # apply scalar kernel (Step 3)
#   ./deploy.sh simd      # apply SIMD kernel (Step 6b) on top
#   ./deploy.sh undo      # remove the additions (best-effort)
# ----------------------------------------------------------------------
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"

MODE="${1:-scalar}"

PARSERS="$ROOT/Deeploy/Targets/Generic/Parsers.py"
BINDINGS="$ROOT/Deeploy/Targets/PULPOpen/Bindings.py"
PLATFORM="$ROOT/Deeploy/Targets/PULPOpen/Platform.py"
TILER="$ROOT/Deeploy/Targets/PULPOpen/Tiler.py"
PULPMATH_H="$ROOT/TargetLibraries/PULPOpen/inc/DeeployPULPMath.h"
TEMPLATES_DIR="$ROOT/Deeploy/Targets/PULPOpen/Templates"
TILECONSTR_DIR="$ROOT/Deeploy/Targets/PULPOpen/TileConstraints"
KERNEL_SRC_DIR="$ROOT/TargetLibraries/PULPOpen/src"
KERNEL_INC_DIR="$ROOT/TargetLibraries/PULPOpen/inc/kernel"
TESTS_DIR="$ROOT/DeeployTest/Tests/Kernels/Integer/LeakyReLU/Regular"
TEST_ARTIFACTS="network.onnx inputs.npz outputs.npz"

case "$MODE" in
undo)
	echo "Undoing iLeakyReLU additions (file copies only)..."
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
	echo "Note: hand-patches in Parsers.py / Bindings.py / Platform.py / Tiler.py / DeeployPULPMath.h were NOT removed."
	echo "If you need a fully clean tree, use: git checkout -- Deeploy/Targets TargetLibraries/PULPOpen/inc/DeeployPULPMath.h"
	exit 0
	;;
scalar | simd) ;;
*)
	echo "Unknown mode '$MODE'. Try: scalar | simd | undo"
	exit 1
	;;
esac

echo "[1/6] Copy test artifacts -> $TESTS_DIR"
mkdir -p "$TESTS_DIR"
for f in $TEST_ARTIFACTS; do
	if [ ! -f "$HERE/$f" ]; then
		echo "  ERROR: $f not found in $HERE - run 'python generate.py' first." >&2
		exit 1
	fi
	cp "$HERE/$f" "$TESTS_DIR/"
done

echo "[2/6] Copy kernel header -> $KERNEL_INC_DIR/iLeakyReLU.h"
cp "$HERE/iLeakyReLU.h" "$KERNEL_INC_DIR/iLeakyReLU.h"

echo "[3/6] Copy kernel source ($MODE) -> $KERNEL_SRC_DIR/iLeakyReLU.c"
if [ "$MODE" = "simd" ]; then
	cp "$HERE/iLeakyReLU_simd.c" "$KERNEL_SRC_DIR/iLeakyReLU.c"
else
	cp "$HERE/iLeakyReLU.c" "$KERNEL_SRC_DIR/iLeakyReLU.c"
fi

echo "[4/6] Copy template + tile constraint"
cp "$HERE/iLeakyReLUTemplate.py" "$TEMPLATES_DIR/iLeakyReLUTemplate.py"
cp "$HERE/iLeakyReLUTileConstraint.py" "$TILECONSTR_DIR/iLeakyReLUTileConstraint.py"

echo "[5/6] Patch DeeployPULPMath.h (idempotent)"
if ! grep -q 'kernel/iLeakyReLU.h' "$PULPMATH_H"; then
	# Insert before the final #endif
	awk '
    /^#endif/ && !done { print "#include \"kernel/iLeakyReLU.h\""; done=1 }
    { print }
  ' "$PULPMATH_H" >"$PULPMATH_H.tmp" && mv "$PULPMATH_H.tmp" "$PULPMATH_H"
fi

echo "[6/6] Patch Parsers.py / Bindings.py / Tiler.py / Platform.py (idempotent)"

# --- Generic/Parsers.py: append iLeakyReLUParser if absent
if ! grep -q 'class iLeakyReLUParser' "$PARSERS"; then
	cat >>"$PARSERS" <<'PARSER_EOF'


class iLeakyReLUParser(NodeParser):

    def __init__(self):
        super().__init__()

    def parseNode(self, node: gs.Node) -> bool:
        wellFormed = all([
            len(node.inputs) == 1,
            len(node.outputs) == 1,
            'mul' in node.attrs,
            'shift' in node.attrs,
        ])
        if not wellFormed:
            return False
        mul   = int(node.attrs['mul'])
        shift = int(node.attrs['shift'])
        # XPULP has no per-lane multiply for packed int8, so the SIMD kernel
        # can only compute max(x, x >> shift), i.e. mul == 1. Its v4s lanes
        # are 8 bits wide, so a shift of 8 or more is undefined behaviour.
        if mul != 1 or not 0 <= shift < 8:
            return False
        self.operatorRepresentation['mul']   = mul
        self.operatorRepresentation['shift'] = shift
        return True

    def parseNodeCtxt(self,
                      ctxt: NetworkContext,
                      node: gs.Node,
                      channels_first: bool = True):
        data_in  = ctxt.lookup(node.inputs[0].name)
        data_out = ctxt.lookup(node.outputs[0].name)
        self.operatorRepresentation['data_in']  = data_in.name
        self.operatorRepresentation['data_out'] = data_out.name
        self.operatorRepresentation['size']     = int(np.prod(data_in.shape))
        return ctxt, True
PARSER_EOF
fi

# --- PULPOpen/Bindings.py: append PULPiLeakyReLUBindings if absent
if ! grep -q 'PULPiLeakyReLUBindings' "$BINDINGS"; then
	# Add template import next to other PULPOpen template imports.
	python3 - <<PY
import re, pathlib
p = pathlib.Path("$BINDINGS")
src = p.read_text()
# Add iLeakyReLUTemplate to the existing PULPOpen.Templates import line.
new = re.sub(
    r"from Deeploy\\.Targets\\.PULPOpen\\.Templates import\\b",
    "from Deeploy.Targets.PULPOpen.Templates import iLeakyReLUTemplate\nfrom Deeploy.Targets.PULPOpen.Templates import",
    src, count=1)
if new == src:
    new = src + "\nfrom Deeploy.Targets.PULPOpen.Templates import iLeakyReLUTemplate\n"
p.write_text(new)
PY
	cat >>"$BINDINGS" <<'BIND_EOF'


PULPiLeakyReLUBindings = [
    NodeBinding(
        ReluChecker([PointerClass(int8_t)], [PointerClass(int8_t)]),
        iLeakyReLUTemplate.referenceTemplate,
        ForkTransformer)
]
BIND_EOF
fi

# --- PULPOpen/Tiler.py: append TilingReady bindings + import
if ! grep -q 'PULPiLeakyReLUTilingReadyBindings' "$TILER"; then
	python3 - <<PY
import pathlib
p = pathlib.Path("$TILER")
src = p.read_text()
extra_imports = (
    "\nfrom Deeploy.Targets.PULPOpen.TileConstraints.iLeakyReLUTileConstraint import iLeakyReLUTileConstraint\n"
    "from Deeploy.Targets.PULPOpen.Bindings import PULPiLeakyReLUBindings\n"
)
# Insert imports just after the last "from Deeploy" line.
idx = src.rfind("from Deeploy")
end = src.find("\n", idx) + 1
src = src[:end] + extra_imports + src[end:]
src += (
    "\nPULPiLeakyReLUTilingReadyBindings = TilingReadyNodeBindings(\n"
    "    nodeBindings = PULPiLeakyReLUBindings,\n"
    "    tileConstraint = iLeakyReLUTileConstraint())\n"
)
p.write_text(src)
PY
fi

# --- PULPOpen/Platform.py: register parser/layer import + mapper + PULPMapping entry
if ! grep -q 'iLeakyReLUMapper' "$PLATFORM"; then
	PLATFORM_FILE="$PLATFORM" python3 - <<'PY'
import os, pathlib
p = pathlib.Path(os.environ["PLATFORM_FILE"])
src = p.read_text()

# Insert the parser import as its own top-level statement immediately before the
# mapper definition.
MAPPER_ANCHOR = "iHardswishMapper = NodeMapper(iHardswishParser(), PULPiHardswishTilingReadyBindings)"
if MAPPER_ANCHOR not in src:
    raise SystemExit("deploy.sh: mapper anchor not found in Platform.py; "
                     "upstream renamed iHardswishMapper - update deploy.sh")

if "iLeakyReLUParser" not in src:
    src = src.replace(
        MAPPER_ANCHOR,
        "from Deeploy.Targets.Generic.Parsers import iLeakyReLUParser\n" + MAPPER_ANCHOR, 1)

if "PULPiLeakyReLUTilingReadyBindings" not in src:
    # Splice into the multi-line `from ... import \` block by extending the
    # line that already mentions PULPiHardswishTilingReadyBindings.
    needle = "PULPiHardswishTilingReadyBindings, \\\n"
    if needle in src:
        src = src.replace(
            needle,
            needle + "    PULPiLeakyReLUTilingReadyBindings, \\\n", 1)
    else:
        # Fallback: drop a standalone single-line import near the others.
        src = src.replace(
            "from Deeploy.Targets.PULPOpen.TopologyOptimizationPasses",
            "from Deeploy.Targets.PULPOpen.Tiler import PULPiLeakyReLUTilingReadyBindings\n"
            "from Deeploy.Targets.PULPOpen.TopologyOptimizationPasses",
            1)

# Insert the mapper line after iHardswishMapper
anchor = "iHardswishMapper = NodeMapper(iHardswishParser(), PULPiHardswishTilingReadyBindings)"
src = src.replace(
    anchor,
    anchor + "\niLeakyReLUMapper = NodeMapper(iLeakyReLUParser(), PULPiLeakyReLUTilingReadyBindings)", 1)

# Insert PULPMapping entry just after the iHardswish one
src = src.replace(
    "'iHardswish': iHardswishLayer([iHardswishMapper]),",
    "'iHardswish': iHardswishLayer([iHardswishMapper]),\n    'iLeakyReLU': iHardswishLayer([iLeakyReLUMapper]),", 1)

p.write_text(src)
PY
fi

echo
echo "Done. Now run the verification tests from DeeployTest/:"
echo "  python deeployRunner_siracusa.py        -t Tests/Kernels/Integer/LeakyReLU/Regular --cores=1                  # baseline"
echo "  python deeployRunner_siracusa.py        -t Tests/Kernels/Integer/LeakyReLU/Regular --cores=8                  # Step 4"
echo "  python deeployRunner_tiled_siracusa.py  -t Tests/Kernels/Integer/LeakyReLU/Regular --cores=8 --l1=32768 --defaultMemLevel=L2   # Step 5/6"

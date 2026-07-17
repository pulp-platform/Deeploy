#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Port Adam optimizer (AdamUpdateV/H/W kernels, parsers, type checkers,
# tile constraints, harness flag) from the Adam-enabled Deeploy04-Last/ fork
# into the current Deeploy/ (formerly DeeployThesis).
#
# Idempotent: re-running overwrites copied files but always restores from the
# Deeploy04-Last source. Originals of any patched file are saved under
# .adam_port_backup/ on the first run.
#
# Usage (run on host, before launching Docker):
#   bash scripts/port_adam_from_deeploy04.sh
# ---------------------------------------------------------------------------
set -euo pipefail

SRC="${SRC:-/home/ahmet/SemesterProject/Deeploy04-Last}"
DST="${DST:-/home/ahmet/SemesterProject/Deeploy}"

if [ ! -d "$SRC/Deeploy/Targets/PULPOpen/TileConstraints" ]; then
  echo "ERROR: SRC ($SRC) not found." >&2
  echo "" >&2
  echo "This script must run on the HOST (where Deeploy04-Last/ lives)," >&2
  echo "not inside the Docker container.  Exit Docker and run again from" >&2
  echo "the host shell:" >&2
  echo "" >&2
  echo "    cd /home/ahmet/SemesterProject/Deeploy" >&2
  echo "    bash scripts/port_adam_from_deeploy04.sh" >&2
  echo "" >&2
  echo "Or set SRC=<path> if Deeploy04-Last is at a different location." >&2
  exit 1
fi
if [ ! -d "$DST/Deeploy/Targets/PULPOpen/TileConstraints" ]; then
  echo "ERROR: DST ($DST) does not look like the target Deeploy." >&2
  exit 1
fi

BACKUP="$DST/.adam_port_backup"
mkdir -p "$BACKUP"

backup () {
  local f="$1"
  local rel="${f#$DST/}"
  local out="$BACKUP/$rel"
  if [ ! -f "$out" ] && [ -f "$f" ]; then
    mkdir -p "$(dirname "$out")"
    cp "$f" "$out"
    echo "    backed up $rel"
  fi
}

copy_if_missing () {
  local src="$1" dst="$2"
  if [ -f "$src" ]; then
    if [ -f "$dst" ]; then
      backup "$dst"
    fi
    cp "$src" "$dst"
    echo "    cp $(basename "$src") → $(dirname "$dst" | sed "s|$DST/||")/"
  fi
}

echo "================================================================"
echo " Adam port: $SRC → $DST"
echo "================================================================"

# ---------------------------------------------------------------------------
# Step 1 — copy standalone Python files (Templates + TileConstraints).
# These have no name collision with the target tree, so direct copy is safe.
# ---------------------------------------------------------------------------
echo ""
echo "[1/8] Copying standalone Adam Templates and TileConstraints..."

for f in "$SRC"/Deeploy/Targets/Generic/Templates/FloatAdam*.py; do
  copy_if_missing "$f" "$DST/Deeploy/Targets/Generic/Templates/$(basename "$f")"
done
for f in "$SRC"/Deeploy/Targets/PULPOpen/Templates/FloatAdam*.py; do
  copy_if_missing "$f" "$DST/Deeploy/Targets/PULPOpen/Templates/$(basename "$f")"
done
for f in "$SRC"/Deeploy/Targets/PULPOpen/TileConstraints/Adam*.py; do
  copy_if_missing "$f" "$DST/Deeploy/Targets/PULPOpen/TileConstraints/$(basename "$f")"
done

# ---------------------------------------------------------------------------
# Step 2 — patch Generic/Layers.py: append AdamLayer + AdamUpdate{V,H,W}Layer.
# ---------------------------------------------------------------------------
echo ""
echo "[2/8] Patching Deeploy/Targets/Generic/Layers.py..."
backup "$DST/Deeploy/Targets/Generic/Layers.py"
python3 - "$SRC" "$DST" <<'PY'
import re, sys
src, dst = sys.argv[1], sys.argv[2]
tgt = f"{dst}/Deeploy/Targets/Generic/Layers.py"
txt = open(tgt).read()
if "class AdamLayer" in txt:
    print("    AdamLayer already present, skipping append")
else:
    src_txt = open(f"{src}/Deeploy/Targets/Generic/Layers.py").read()
    # Pull each Adam* class block from source.
    blocks = re.findall(
        r"(class Adam[A-Za-z]*Layer\(.*?\):\n(?:.|\n)*?)(?=\n\nclass |\Z)",
        src_txt)
    if not blocks:
        print("    WARN: no Adam classes found in source"); sys.exit(0)
    marker = "\n\n# ---- ADAM PORT BEGIN ----\n"
    end    = "\n# ---- ADAM PORT END ----\n"
    txt = txt.rstrip() + marker + "\n\n".join(blocks) + end
    open(tgt, "w").write(txt)
    print(f"    appended {len(blocks)} Adam Layer class(es)")
PY

# ---------------------------------------------------------------------------
# Step 3 — patch Generic/Parsers.py.
# ---------------------------------------------------------------------------
echo ""
echo "[3/8] Patching Deeploy/Targets/Generic/Parsers.py..."
backup "$DST/Deeploy/Targets/Generic/Parsers.py"
python3 - "$SRC" "$DST" <<'PY'
import re, sys
src, dst = sys.argv[1], sys.argv[2]
tgt = f"{dst}/Deeploy/Targets/Generic/Parsers.py"
txt = open(tgt).read()
if "class AdamParser" in txt or "class AdamUpdateVParser" in txt:
    print("    AdamParser already present, skipping append")
else:
    src_txt = open(f"{src}/Deeploy/Targets/Generic/Parsers.py").read()
    blocks = re.findall(
        r"(class Adam[A-Za-z]*Parser\(.*?\):\n(?:.|\n)*?)(?=\n\nclass |\Z)",
        src_txt)
    if not blocks:
        print("    WARN: no Adam parser classes found in source"); sys.exit(0)
    txt = txt.rstrip() + "\n\n# ---- ADAM PORT BEGIN ----\n" \
        + "\n\n".join(blocks) + "\n# ---- ADAM PORT END ----\n"
    open(tgt, "w").write(txt)
    print(f"    appended {len(blocks)} Adam Parser class(es)")
PY

# ---------------------------------------------------------------------------
# Step 4 — patch Generic/TypeCheckers.py.
# ---------------------------------------------------------------------------
echo ""
echo "[4/8] Patching Deeploy/Targets/Generic/TypeCheckers.py..."
backup "$DST/Deeploy/Targets/Generic/TypeCheckers.py"
python3 - "$SRC" "$DST" <<'PY'
import re, sys
src, dst = sys.argv[1], sys.argv[2]
tgt = f"{dst}/Deeploy/Targets/Generic/TypeCheckers.py"
txt = open(tgt).read()
if "class AdamChecker" in txt or "class AdamUpdateVChecker" in txt:
    print("    AdamChecker already present, skipping append")
else:
    src_txt = open(f"{src}/Deeploy/Targets/Generic/TypeCheckers.py").read()
    blocks = re.findall(
        r"(class Adam[A-Za-z]*Checker\(.*?\):\n(?:.|\n)*?)(?=\n\nclass |\Z)",
        src_txt)
    if not blocks:
        print("    WARN: no Adam checker classes found in source"); sys.exit(0)
    txt = txt.rstrip() + "\n\n# ---- ADAM PORT BEGIN ----\n" \
        + "\n\n".join(blocks) + "\n# ---- ADAM PORT END ----\n"
    open(tgt, "w").write(txt)
    print(f"    appended {len(blocks)} Adam Checker class(es)")
PY

# ---------------------------------------------------------------------------
# Step 5 — patch PULPOpen/Bindings.py: add PULPAdamUpdate* bindings.
# ---------------------------------------------------------------------------
echo ""
echo "[5/8] Patching Deeploy/Targets/PULPOpen/Bindings.py..."
backup "$DST/Deeploy/Targets/PULPOpen/Bindings.py"
python3 - "$SRC" "$DST" <<'PY'
import re, sys
src, dst = sys.argv[1], sys.argv[2]
tgt = f"{dst}/Deeploy/Targets/PULPOpen/Bindings.py"
txt = open(tgt).read()
if "PULPAdamUpdateV" in txt:
    print("    PULPAdam* bindings already present, skipping")
else:
    src_txt = open(f"{src}/Deeploy/Targets/PULPOpen/Bindings.py").read()
    # Pull all PULPAdam* binding blocks.
    binds = re.findall(
        r"(PULPAdamUpdate[VHW]Bindings = \[(?:.|\n)*?\n\])",
        src_txt)
    if not binds:
        print("    WARN: no PULPAdam* bindings found in source"); sys.exit(0)
    # Imports to splice into the file header (best-effort).
    need_imports = (
        "from Deeploy.Targets.Generic.TypeCheckers import "
        "AdamUpdateHChecker, AdamUpdateVChecker, AdamUpdateWChecker\n"
        "from Deeploy.Targets.PULPOpen.Templates import "
        "FloatAdamUpdateHTemplate, FloatAdamUpdateVTemplate, FloatAdamUpdateWTemplate\n"
    )
    txt = need_imports + txt
    txt = txt.rstrip() + "\n\n# ---- ADAM PORT BEGIN ----\n" \
        + "\n\n".join(binds) + "\n# ---- ADAM PORT END ----\n"
    open(tgt, "w").write(txt)
    print(f"    appended {len(binds)} PULPAdam* binding block(s)")
PY

# ---------------------------------------------------------------------------
# Step 6 — patch PULPOpen/Platform.py: register Adam mappers + PULPMapping.
# ---------------------------------------------------------------------------
echo ""
echo "[6/8] Patching Deeploy/Targets/PULPOpen/Platform.py..."
backup "$DST/Deeploy/Targets/PULPOpen/Platform.py"
python3 - "$SRC" "$DST" <<'PY'
import re, sys
src, dst = sys.argv[1], sys.argv[2]
tgt = f"{dst}/Deeploy/Targets/PULPOpen/Platform.py"
txt = open(tgt).read()
if "AdamUpdateVMapper" in txt:
    print("    Adam mapper already registered, skipping")
else:
    # Append imports + mapper definitions + PULPMapping entries near the end.
    snippet = """

# ---- ADAM PORT BEGIN ----
from Deeploy.Targets.Generic.Layers   import AdamUpdateHLayer, AdamUpdateVLayer, AdamUpdateWLayer
from Deeploy.Targets.Generic.Parsers  import AdamUpdateHParser, AdamUpdateVParser, AdamUpdateWParser
from Deeploy.Targets.PULPOpen.Bindings import (
    PULPAdamUpdateHBindings,
    PULPAdamUpdateVBindings,
    PULPAdamUpdateWBindings,
)

AdamUpdateVMapper = NodeMapper(AdamUpdateVParser(), PULPAdamUpdateVBindings)
AdamUpdateHMapper = NodeMapper(AdamUpdateHParser(), PULPAdamUpdateHBindings)
AdamUpdateWMapper = NodeMapper(AdamUpdateWParser(), PULPAdamUpdateWBindings)

# Register in the PULPMapping dict.
try:
    PULPMapping['AdamUpdateV'] = AdamUpdateVLayer([AdamUpdateVMapper])
    PULPMapping['AdamUpdateH'] = AdamUpdateHLayer([AdamUpdateHMapper])
    PULPMapping['AdamUpdateW'] = AdamUpdateWLayer([AdamUpdateWMapper])
except NameError:
    # PULPMapping symbol may have a different name in this fork. Edit by hand.
    print("ADAM PORT: PULPMapping not found, register AdamUpdate{V,H,W} manually")
# ---- ADAM PORT END ----
"""
    txt = txt.rstrip() + snippet
    open(tgt, "w").write(txt)
    print("    appended Adam mapper registration block")
PY

# ---------------------------------------------------------------------------
# Step 7 — patch Siracusa CMakeLists.txt and replace deeploytraintest.c.
# ---------------------------------------------------------------------------
echo ""
echo "[7/8] Patching Siracusa harness for Adam (#ifdef OPTIMIZER_ADAM)..."
backup "$DST/DeeployTest/Platforms/Siracusa/CMakeLists.txt"
backup "$DST/DeeployTest/Platforms/Siracusa/src/deeploytraintest.c"

# Replace the C harness with the Adam-aware version from the source fork.
cp "$SRC/DeeployTest/Platforms/Siracusa/src/deeploytraintest.c" \
   "$DST/DeeployTest/Platforms/Siracusa/src/deeploytraintest.c"
echo "    replaced deeploytraintest.c"

# Append the OPTIMIZER_ADAM CMake option + define block if not present.
if grep -q "OPTIMIZER_ADAM" "$DST/DeeployTest/Platforms/Siracusa/CMakeLists.txt"; then
  echo "    OPTIMIZER_ADAM already in CMakeLists.txt"
else
  cat >> "$DST/DeeployTest/Platforms/Siracusa/CMakeLists.txt" <<'CM'

# ---- ADAM PORT BEGIN ----
option(OPTIMIZER_ADAM "Use Adam optimizer kernels instead of SGD" OFF)
if(TRAINING AND OPTIMIZER_ADAM)
    target_compile_definitions(${ProjectId} PRIVATE OPTIMIZER_ADAM=1)
endif()
# ---- ADAM PORT END ----
CM
  echo "    appended OPTIMIZER_ADAM option to CMakeLists.txt"
fi

# ---------------------------------------------------------------------------
# Step 8 — patch test runner to auto-detect Adam from optimizer ONNX and
# forward -DOPTIMIZER_ADAM=ON when needed.
# ---------------------------------------------------------------------------
echo ""
echo "[8/8] Patching testUtils runner + config + execution..."
for f in DeeployTest/testUtils/deeployTrainingRunner.py \
         DeeployTest/testUtils/core/config.py \
         DeeployTest/testUtils/core/execution.py ; do
  backup "$DST/$f"
done

python3 - "$SRC" "$DST" <<'PY'
import os, re, sys
src, dst = sys.argv[1], sys.argv[2]

# (a) config.py — add `optimizer_adam: bool = False` to DeeployTestConfig.
cfg_path = f"{dst}/DeeployTest/testUtils/core/config.py"
cfg = open(cfg_path).read()
if "optimizer_adam" not in cfg:
    # Insert after `n_train_steps` or after a `@dataclass` field area.
    cfg2, n = re.subn(
        r"(n_train_steps:\s*Optional\[int\]\s*=\s*None\n)",
        r"\1    optimizer_adam: bool = False\n",
        cfg, count=1)
    if n == 0:
        cfg2, n = re.subn(
            r"(\n\s*tiling:\s*bool[^\n]*\n)",
            r"\1    optimizer_adam: bool = False\n",
            cfg, count=1)
    if n == 0:
        cfg2 = cfg.rstrip() + "\n# ADAM PORT: add `optimizer_adam: bool = False` manually\n"
        print("    config.py: insertion site not found, marked TODO at EOF")
    else:
        print("    config.py: added optimizer_adam field")
    open(cfg_path, "w").write(cfg2)
else:
    print("    config.py: optimizer_adam already present")

# (b) execution.py — emit -DOPTIMIZER_ADAM=ON when config.optimizer_adam.
ex_path = f"{dst}/DeeployTest/testUtils/core/execution.py"
ex = open(ex_path).read()
if "OPTIMIZER_ADAM" not in ex:
    # Try to splice into the CMake-args section near N_TRAIN_STEPS.
    insert = """        if getattr(config, 'optimizer_adam', False):
            cmd.append("-DOPTIMIZER_ADAM=ON")
"""
    ex2, n = re.subn(
        r"(if config\.n_train_steps[^\n]*\n[^\n]*N_TRAIN_STEPS[^\n]*\n)",
        r"\1" + insert,
        ex, count=1)
    if n == 0:
        ex2 = ex.rstrip() + "\n# ADAM PORT: when training, append -DOPTIMIZER_ADAM=ON if config.optimizer_adam\n"
        print("    execution.py: insertion site not found, marked TODO at EOF")
    else:
        print("    execution.py: added OPTIMIZER_ADAM forwarding")
    open(ex_path, "w").write(ex2)
else:
    print("    execution.py: OPTIMIZER_ADAM already present")

# (c) runner — auto-detect Adam in the optimizer ONNX and pass flag down.
rp = f"{dst}/DeeployTest/testUtils/deeployTrainingRunner.py"
rt = open(rp).read()
if "AdamUpdate" not in rt:
    add = """
    # ---- ADAM PORT BEGIN ----
    _is_adam = False
    if getattr(args, 'optimizer_dir', None):
        import os as _os
        _opt_onnx = _os.path.join(args.optimizer_dir, "network.onnx")
        if _os.path.exists(_opt_onnx):
            try:
                import onnx as _onnx
                _opt_model = _onnx.load(_opt_onnx)
                _is_adam = any(n.op_type.startswith("AdamUpdate") for n in _opt_model.graph.node)
            except Exception:
                pass
    # ---- ADAM PORT END ----
"""
    # Insert just before the DeeployTestConfig(...) construction call.
    rt2, n = re.subn(
        r"(\n\s*config\s*=\s*DeeployTestConfig\()",
        add + r"\1",
        rt, count=1)
    if n == 0:
        rt2 = rt.rstrip() + "\n# ADAM PORT: auto-detect Adam from optimizer ONNX, then set config.optimizer_adam = True\n"
        print("    deeployTrainingRunner.py: insertion site not found, marked TODO at EOF")
    else:
        # Also try to splice `optimizer_adam = _is_adam` into the DeeployTestConfig kwargs.
        rt3, m = re.subn(
            r"(n_train_steps\s*=\s*args\.n_steps,\s*\n)",
            r"\1        optimizer_adam = _is_adam,\n",
            rt2, count=1)
        if m == 0:
            print("    deeployTrainingRunner.py: please add `optimizer_adam = _is_adam,` to DeeployTestConfig manually")
            open(rp, "w").write(rt2)
        else:
            print("    deeployTrainingRunner.py: added Adam auto-detect + optimizer_adam kwarg")
            open(rp, "w").write(rt3)
else:
    print("    deeployTrainingRunner.py: Adam auto-detect already present")
PY

echo ""
echo "================================================================"
echo " DONE."
echo ""
echo " Next steps:"
echo "   1. Smoke-test the imports:"
echo "        cd $DST"
echo "        python -c 'from Deeploy.Targets.PULPOpen.Platform import PULPMapping; print(\"AdamUpdateV\" in PULPMapping)'"
echo "      Should print True."
echo ""
echo "   2. Inside Docker, regenerate ONNXs and run the benchmark:"
echo "        cd /app/Deeploy/Deeploy/DeeployTest"
echo "        bash scripts/bench_sgd_vs_adam.sh"
echo ""
echo " If something fails, the originals are in $BACKUP/."
echo " To revert any file:   cp \$BACKUP/<rel-path> \$DST/<rel-path>"
echo "================================================================"

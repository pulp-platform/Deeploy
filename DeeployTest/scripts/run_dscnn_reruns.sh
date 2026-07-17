#!/usr/bin/env bash
# Re-run DSCNN Relu-checkpoint, Cost K=16 and Cost K=24 variants.
# Run from /app/Deeploy/Deeploy/DeeployTest inside the Docker container.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEEPLOY_TEST_DIR="$(dirname "$SCRIPT_DIR")"
LOG_DIR="$SCRIPT_DIR/logs/dscnn_reruns"
mkdir -p "$LOG_DIR"

# Force fresh build so no stale OPTIMIZER_ADAM flag leaks in.
rm -rf "$DEEPLOY_TEST_DIR/TEST_SIRACUSA/build_master"

VARIANTS=(relu cost16 cost24)

for v in "${VARIANTS[@]}"; do
  log="$LOG_DIR/dscnn_${v}.log"
  echo ">>> DSCNN $v  →  $log"
  ( cd "$DEEPLOY_TEST_DIR" && bash scripts/run_dscnn_${v}.sh ) > "$log" 2>&1 || {
    echo "    FAILED (see $log)"
    continue
  }
  bench=$(grep -E "^BENCH " "$log" | tail -1 || true)
  if [ -n "$bench" ]; then
    echo "    $bench"
  else
    # Fall back to multi-line BENCH format.
    grep -E "\[BENCH\][[:space:]]+(avg|total)" "$log" | head -4 | sed 's/^/    /'
  fi
done

echo ""
echo "DONE. Logs in $LOG_DIR/"

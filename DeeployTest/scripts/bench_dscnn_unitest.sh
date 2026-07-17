#!/usr/bin/env bash
# Bench DSCNN variants regenerated through the unified Onnx4Deeploy.
# Compares against the original run_dscnn_* results so we validate both
# Onnx4Deeploy AND Deeploy without touching the historical CSV.
#
# Run from inside Docker with workdir = DeeployTest/.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEEPLOY_TEST_DIR="$(dirname "$SCRIPT_DIR")"
OUT_CSV="$SCRIPT_DIR/logs/benchmark/dscnn_unitest.csv"
LOG_DIR="$SCRIPT_DIR/logs/dscnn_unitest"
mkdir -p "$LOG_DIR" "$(dirname "$OUT_CSV")"

echo "variant,l2_peak,fwd_bwd_cycles,opt_cycles" > "$OUT_CSV"

VARIANTS=(NOGCP COSTK1 COSTK2 COSTK5 COSTK8 COSTK12 COSTK16)

rm -rf "$DEEPLOY_TEST_DIR/TEST_SIRACUSA/build_master"

for v in "${VARIANTS[@]}"; do
  test_dir="$DEEPLOY_TEST_DIR/Tests/Models/Training/DSCNN_UNITEST_${v}/dscnn_train"
  opt_dir="$DEEPLOY_TEST_DIR/Tests/Models/Training/DSCNN_UNITEST_${v}/dscnn_optimizer"
  log="$LOG_DIR/dscnn_${v,,}.log"
  echo ">>> DSCNN $v"
  ( cd "$DEEPLOY_TEST_DIR" && python deeployTrainingRunner_tiled_siracusa.py \
      -t "$test_dir" \
      --optimizer-dir "$opt_dir" \
      --n-steps 1 --n-accum 4 \
      --l1 128000 --l2 2000000 --defaultMemLevel L2 \
      --memAllocStrategy MiniMalloc --searchStrategy random-max \
      --tolerance 0.09 -v ) > "$log" 2>&1 || { echo "    FAILED  (log $log)"; continue; }

  bench=$(grep -E "^BENCH " "$log" | tail -1 || true)
  if [ -n "$bench" ]; then
    tc=$(echo "$bench" | sed -E 's/.*train_cycles=([0-9]+).*/\1/')
    oc=$(echo "$bench" | sed -E 's/.*opt_cycles=([0-9]+).*/\1/')
  else
    tc=$(awk '/\[BENCH\] --- Training/{s=1} /\[BENCH\][[:space:]]+avg/{if(s){for(i=1;i<=NF;i++)if($i=="=")print $(i+1);exit}}' "$log")
    oc=$(awk '/\[BENCH\] --- Optimizer/{s=1} /\[BENCH\][[:space:]]+avg/{if(s){for(i=1;i<=NF;i++)if($i=="=")print $(i+1);exit}}' "$log")
  fi
  peak=$(grep -oP 'L2\s+MEMORYARENA_len\s*=\s*\K[0-9]+' "$log" | tail -1 || true)
  if [ -z "$peak" ]; then
    peak=$(grep -oP 'L2[[:space:]]+[0-9,]+[[:space:]]+\K[0-9,]+' "$log" | head -1 | tr -d ',' || true)
  fi
  echo "    L2=$peak  train=$tc  opt=$oc"
  echo "$v,${peak:-?},${tc:-?},${oc:-?}" >> "$OUT_CSV"
done

echo ""
echo "DONE. CSV: $OUT_CSV"

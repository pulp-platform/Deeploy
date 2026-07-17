#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# SGD vs Adam — per-step cycle benchmark on Siracusa gvsoc.
#
# Pre-requisite:
#   1. Run scripts/port_adam_from_deeploy04.sh on the host so this Deeploy
#      has the Adam kernels, parsers, tile constraints, harness #ifdef.
#   2. Use Onnx4Deeploy on the host (env: o4d) to generate
#         Tests/Models/Training/<MODEL>_BENCH/<model>_train/network.onnx        (SGD)
#         Tests/Models/Training/<MODEL>_BENCH/<model>_adam_train/network.onnx   (Adam)
#      (and the sibling *_optimizer dirs).
#
# Run from inside the Deeploy Docker container with workdir = DeeployTest.
#
# Usage:
#   cd <repo>/Deeploy/DeeployTest
#   bash scripts/bench_sgd_vs_adam.sh                # full sweep
#   bash scripts/bench_sgd_vs_adam.sh DSCNN          # one model
# ---------------------------------------------------------------------------
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEEPLOY_TEST_DIR="${DEEPLOY_TEST_DIR:-$(dirname "$SCRIPT_DIR")}"

OUT_CSV="${OUT_CSV:-$SCRIPT_DIR/logs/benchmark/sgd_vs_adam_bench.csv}"
LOG_DIR="${LOG_DIR:-$SCRIPT_DIR/logs/sgd_vs_adam}"

N_STEPS="${N_STEPS:-1}"
N_ACCUM="${N_ACCUM:-4}"
L1="${L1:-128000}"
L2="${L2:-2000000}"

mkdir -p "$LOG_DIR" "$(dirname "$OUT_CSV")"

# model_id : default_mem_level : lowercase tag
MODELS=(
  "DSCNN:L2:dscnn"
  "MobileNetV1:L3:mobilenetv1"
  "ResNet8:L3:resnet8"
  "CCT_2:L2:cct_2"
  "CCT_3:L2:cct_3"
)

FILTER="${1:-}"

if [ ! -f "$OUT_CSV" ]; then
  echo "model,optimizer,n_steps,n_accum,fwd_bwd_cycles,opt_cycles,total_cycles,weight_sram" > "$OUT_CSV"
fi


nuke_build () {
  # Force a clean rebuild between runs that change OPTIMIZER_ADAM.
  local b="$DEEPLOY_TEST_DIR/TEST_SIRACUSA/build_master"
  rm -f "$b/CMakeCache.txt" 2>/dev/null || true
  find "$b" -name "deeploytraintest.c.obj" -delete 2>/dev/null || true
  find "$b" -name "deeploytraintest.c.o"   -delete 2>/dev/null || true
}


run_bench () {
  local model_id="$1" mem_level="$2" lower="$3" optimizer="$4"
  local infix=""
  if [ "$optimizer" = "adam" ]; then infix="_adam"; fi
  local train_dir="$DEEPLOY_TEST_DIR/Tests/Models/Training/${model_id}_BENCH/${lower}${infix}_train"
  local opt_dir="$DEEPLOY_TEST_DIR/Tests/Models/Training/${model_id}_BENCH/${lower}${infix}_optimizer"
  local log_file="$LOG_DIR/${lower}_bench_${optimizer}.log"

  if [ ! -f "$train_dir/network.onnx" ] || [ ! -f "$opt_dir/network.onnx" ]; then
    echo "  MISSING ONNXs for $model_id/$optimizer"
    echo "    expected: $train_dir/network.onnx"
    echo "    expected: $opt_dir/network.onnx"
    echo "$model_id,$optimizer,$N_STEPS,$N_ACCUM,NO_ONNX,NO_ONNX,NO_ONNX,NO_ONNX" >> "$OUT_CSV"
    return
  fi

  echo "  benchmarking $model_id with $optimizer (mem=$mem_level)..."

  local cmd=( python deeployTrainingRunner_tiled_siracusa.py
              -t "$train_dir"
              --optimizer-dir "$opt_dir"
              --n-steps "$N_STEPS"
              --n-accum "$N_ACCUM"
              --l1 "$L1"
              --l2 "$L2"
              --defaultMemLevel "$mem_level"
              --memAllocStrategy MiniMalloc
              --searchStrategy random-max
              --tolerance 0.09
              -v )
  if [ "$optimizer" = "adam" ]; then
    cmd+=( -D OPTIMIZER_ADAM=ON )
  else
    cmd+=( -D OPTIMIZER_ADAM=OFF )
  fi

  ( cd "$DEEPLOY_TEST_DIR" && "${cmd[@]}" ) > "$log_file" 2>&1 || {
    echo "    FAILED (see $log_file)"
    echo "$model_id,$optimizer,$N_STEPS,$N_ACCUM,FAIL,FAIL,FAIL,FAIL" >> "$OUT_CSV"
    return
  }

  # The Adam-aware harness emits a multi-line summary block:
  #   [BENCH] --- Training (fwd+bwd) per step (=N mini-batches) ---
  #   [BENCH]   total  = N cycles
  #   [BENCH]   avg    = N cycles/step
  #   [BENCH] --- Optimizer kernel per step ---
  #   [BENCH]   total  = N cycles
  #   [BENCH]   avg    = N cycles/step
  local train opt total
  train=$(awk '
    $0 ~ "\\[BENCH\\] ---" { section = $0 }
    /\[BENCH\][[:space:]]+avg/ {
      if (section ~ "Training") { for (i=1;i<=NF;i++) if ($i=="=") {print $(i+1); exit} }
    }' "$log_file")
  opt=$(awk '
    $0 ~ "\\[BENCH\\] ---" { section = $0 }
    /\[BENCH\][[:space:]]+avg/ {
      if (section ~ "Optimizer") { for (i=1;i<=NF;i++) if ($i=="=") {print $(i+1); exit} }
    }' "$log_file")
  if [ -z "$train" ] || [ -z "$opt" ]; then
    echo "    no BENCH avg in $log_file"
    echo "$model_id,$optimizer,$N_STEPS,$N_ACCUM,NOBENCH,NOBENCH,NOBENCH,NOBENCH" >> "$OUT_CSV"
    return
  fi
  total=$((train + opt))
  echo "    train_avg=$train  opt_avg=$opt  total=$total cyc/step"
  echo "$model_id,$optimizer,$N_STEPS,$N_ACCUM,$train,$opt,$total,0" >> "$OUT_CSV"
}


main () {
  echo "Output CSV : $OUT_CSV"
  echo "Per-run log: $LOG_DIR"
  echo "Config     : n_steps=$N_STEPS n_accum=$N_ACCUM L1=$L1 L2=$L2"
  echo "============================================================"

  for spec in "${MODELS[@]}"; do
    IFS=":" read -r model_id mem_level lower <<<"$spec"
    if [ -n "$FILTER" ] && [ "$FILTER" != "$model_id" ]; then continue; fi
    echo ""
    echo ">>> $model_id  ($mem_level)"
    for opt in sgd adam; do
      nuke_build
      run_bench "$model_id" "$mem_level" "$lower" "$opt"
    done
  done

  echo ""
  echo "============================================================"
  echo "DONE. CSV: $OUT_CSV"
}

main "$@"

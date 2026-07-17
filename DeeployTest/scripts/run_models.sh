#!/usr/bin/env bash
set -u
cd "$(dirname "$0")/.."

LOG_DIR="$(pwd)/scripts/logs/run_models"
CSV_OUT="$(pwd)/scripts/logs/benchmark/run_models.csv"
mkdir -p "$LOG_DIR" "$(dirname "$CSV_OUT")"
if [ ! -f "$CSV_OUT" ]; then
  echo "model,variant,l2_peak,fwd_bwd_cycles,opt_cycles" > "$CSV_OUT"
fi

DEFAULT_L2="--l1 128000 --l2 2000000 --defaultMemLevel L2 --tolerance 0.09"
DEFAULT_COSTK_L2="--l1 128000 --l2 1500000 --defaultMemLevel L2 --tolerance 0.09"
RESNET_COSTK="--l1 128000 --l2 2000000 --defaultMemLevel L2 --tolerance 0.09"
MOBILENET_L2="--l1 128000 --l2 2000000 --defaultMemLevel L3 --tolerance 0.09"

TOPK_MODELS=(
  "DSCNN|dscnn|1,2,3,5,8,11,12,18|"
  "CCT_2|cct_2|1,2,4,9,17,19|"
  "CCT_3|cct_3|1,2,4,8,15,21|"
  "ResNet8|resnet8|1,2,4,8,11,13,16,20|RESNET"
  "MobileNetV1|mobilenetv1|1,4,8,16,24,32,40,48|MOBILENET"
)

SIZE_MODELS=(
  "CCT_2|cct_2|256,512,1024,2048,4096,8192|"
  "CCT_3|cct_3|256,512,1024,2048,4096,8192|"
)

FILTER=${1:-}

run_one () {
  local model=$1 variant=$2 test_dir=$3 opt_dir=$4 nsteps=$5 flags=$6
  local log="$LOG_DIR/${model,,}_${variant,,}.log"
  echo ">>> $model $variant"
  if [ ! -d "$test_dir" ]; then
    echo "    SKIP"
    echo "$model,$variant,MISSING,MISSING,MISSING" >> "$CSV_OUT"
    return
  fi
  python deeployTrainingRunner_tiled_siracusa.py \
    -t "$test_dir" --optimizer-dir "$opt_dir" \
    --n-steps $nsteps --n-accum 4 -vvv --plotMemAlloc \
    $flags > "$log" 2>&1
  if [ $? -ne 0 ]; then
    echo "    FAILED (log $log)"
    echo "$model,$variant,FAIL,FAIL,FAIL" >> "$CSV_OUT"
    return
  fi
  local bench=$(grep -E "^BENCH " "$log" | tail -1 || true)
  local tc oc
  if [ -n "$bench" ]; then
    tc=$(echo "$bench" | sed -E 's/.*train_cycles=([0-9]+).*/\1/')
    oc=$(echo "$bench" | sed -E 's/.*opt_cycles=([0-9]+).*/\1/')
  else
    tc=$(awk '/\[BENCH\] --- Training/{s=1} /\[BENCH\][[:space:]]+avg/{if(s){for(i=1;i<=NF;i++)if($i=="=")print $(i+1);exit}}' "$log")
    oc=$(awk '/\[BENCH\] --- Optimizer/{s=1} /\[BENCH\][[:space:]]+avg/{if(s){for(i=1;i<=NF;i++)if($i=="=")print $(i+1);exit}}' "$log")
  fi
  local peak=$(grep -oP 'L2\s+MEMORYARENA_len\s*=\s*\K[0-9]+' "$log" | tail -1 || true)
  [ -z "$peak" ] && peak=$(grep -oP 'L2[[:space:]]+[0-9,]+[[:space:]]+\K[0-9,]+' "$log" | head -1 | tr -d ',' || true)
  echo "    L2=${peak:-?}  train=${tc:-?}  opt=${oc:-?}"
  echo "$model,$variant,${peak:-?},${tc:-?},${oc:-?}" >> "$CSV_OUT"
}

rm -rf TEST_SIRACUSA/build_master

for spec in "${TOPK_MODELS[@]}"; do
  IFS='|' read -r model lower klist override <<<"$spec"
  [ -n "$FILTER" ] && [ "$FILTER" != "${lower}" ] && [ "$FILTER" != "${model,,}" ] && continue

  case "$override" in
    RESNET)    NOGCP_F="$DEFAULT_L2";   COSTK_F="$RESNET_COSTK" ;;
    MOBILENET) NOGCP_F="$MOBILENET_L2"; COSTK_F="$MOBILENET_L2" ;;
    *)         NOGCP_F="$DEFAULT_L2";   COSTK_F="$DEFAULT_COSTK_L2" ;;
  esac

  echo ""
  echo "================================================================"
  echo " $model (topk)"
  echo "================================================================"
  run_one "$model" "nogcp" \
    "./Tests/Models/Training/${model}_UNI_NOGCP/${lower}_train/" \
    "./Tests/Models/Training/${model}_UNI_NOGCP/${lower}_optimizer/" \
    1 "$NOGCP_F"

  IFS=',' read -ra ks <<<"$klist"
  for K in "${ks[@]}"; do
    run_one "$model" "k${K}" \
      "./Tests/Models/Training/${model}_UNI_K${K}/${lower}_train/" \
      "./Tests/Models/Training/${model}_UNI_K${K}/${lower}_optimizer/" \
      10 "$COSTK_F"
  done
done

for spec in "${SIZE_MODELS[@]}"; do
  IFS='|' read -r model lower tlist override <<<"$spec"
  [ -n "$FILTER" ] && [ "$FILTER" != "${lower}" ] && [ "$FILTER" != "${model,,}" ] && continue

  echo ""
  echo "================================================================"
  echo " $model (size-gated)"
  echo "================================================================"
  IFS=',' read -ra ts <<<"$tlist"
  for T in "${ts[@]}"; do
    run_one "$model" "size${T}" \
      "./Tests/Models/Training/${model}_UNI_SIZE${T}/${lower}_train/" \
      "./Tests/Models/Training/${model}_UNI_SIZE${T}/${lower}_optimizer/" \
      10 "$DEFAULT_COSTK_L2"
  done
done

echo ""
echo "DONE. CSV: $CSV_OUT"

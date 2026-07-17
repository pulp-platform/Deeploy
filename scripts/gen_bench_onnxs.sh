#!/usr/bin/env bash
# Generate SGD + Adam training/optimizer ONNXs for the 5 benchmark models
# under the new Deeploy/ tree. Run on the host (env: o4d).
#
# Layout matches what scripts/bench_sgd_vs_adam.sh expects:
#   Tests/Models/Training/<MODEL>_BENCH/<model>_train/network.onnx        (SGD)
#   Tests/Models/Training/<MODEL>_BENCH/<model>_adam_train/network.onnx   (Adam)
#   ... and the *_optimizer siblings.
#
# Usage (host):
#   bash scripts/gen_bench_onnxs.sh
set -euo pipefail

PY="${PY:-/home/ahmet/anaconda3/envs/o4d/bin/python}"
ONNX4D="${ONNX4D:-/home/ahmet/SemesterProject/Onnx4DeeployLatest/Onnx4Deeploy/Onnx4Deeploy.py}"
ROOT="${ROOT:-/home/ahmet/SemesterProject/Deeploy/DeeployTest/Tests/Models/Training}"

N_STEPS="${N_STEPS:-1}"
N_ACCUM="${N_ACCUM:-4}"

# model_id : onnx4deeploy_name : lowercase : extra args
MODELS=(
  "DSCNN:DSCNN:dscnn:"
  "MobileNetV1:MobileNetV2-VWW:mobilenetv1:"
  "ResNet8:ResNet8:resnet8:"
  "CCT_2:CCT:cct_2:--num-layers 2"
  "CCT_3:CCT:cct_3:--num-layers 3"
)

for spec in "${MODELS[@]}"; do
  IFS=":" read -r model_id model_name lower extra <<<"$spec"
  for opt in sgd adam; do
    if [ "$opt" = "adam" ]; then suffix="_adam_train"; else suffix="_train"; fi
    out="$ROOT/${model_id}_BENCH/${lower}${suffix}"
    echo "=== $model_id ($opt) → $out ==="
    mkdir -p "$(dirname "$out")"
    rm -rf "$out" "${out%_train}_optimizer"
    $PY "$ONNX4D" -model "$model_name" -mode train \
      --optimizer "$opt" \
      --n-steps "$N_STEPS" --n-accum "$N_ACCUM" \
      $extra -o "$out" 2>&1 | tail -3
  done
done

echo ""
echo "DONE. Verify with:"
echo "  ls $ROOT/{DSCNN,MobileNetV1,ResNet8,CCT_2,CCT_3}_BENCH/"

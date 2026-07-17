#!/usr/bin/env bash
# Run all CCT-family sweeps (CCT-2, CCT-3, CCT_LoRA L2-mode) in one shot.
set -u
SCRIPT_DIR="$(dirname "$0")"
cd "$SCRIPT_DIR/.."

echo "================================================================"
echo " STAGE 1/2: CCT-2 and CCT-3 (Onnx4Deeploy-generated variants)"
echo "================================================================"
bash "$SCRIPT_DIR/run_cct_n_all.sh"
status_cctn=$?

echo
echo "================================================================"
echo " STAGE 2/2: CCT_LoRA L2-resident sweep"
echo "================================================================"
bash "$SCRIPT_DIR/run_cct_lora_l2_all.sh"
status_lora=$?

echo
echo "================================================================"
echo " Combined CCT sweep done"
echo "  CCT-2/3      : exit=$status_cctn"
echo "  CCT_LoRA L2  : exit=$status_lora"
echo "================================================================"
[[ $status_cctn -eq 0 && $status_lora -eq 0 ]] && exit 0 || exit 1

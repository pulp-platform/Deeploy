#!/usr/bin/env bash
# L2-resident mode for CCT_LoRA (footprint ~256 KB fits easily in 2 MB L2).
python deeployTrainingRunner_tiled_siracusa.py \
  -t ./Tests/Models/Training/CCT_LoRA_COST40/cct_lora_train/ \
  --optimizer-dir ./Tests/Models/Training/CCT_LoRA_COST40/cct_lora_optimizer/ \
  --n-steps 1 --n-accum 4 \
  --l1 128000 --l2 4000000 --defaultMemLevel L2 \
  --tolerance 0.09 -vvv --plotMemAlloc "$@"

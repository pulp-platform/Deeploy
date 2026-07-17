#!/usr/bin/env bash
python deeployTrainingRunner_tiled_siracusa.py \
  -t ./Tests/Models/Training/CCT_LoRA_LN_GELU/cct_lora_train/ \
  --optimizer-dir ./Tests/Models/Training/CCT_LoRA_LN_GELU/cct_lora_optimizer/ \
  --n-steps 1 --n-accum 4 \
  --l1 128000 --l2 2000000 --defaultMemLevel L3 \
  --tolerance 0.09 -vvv --plotMemAlloc "$@"

#!/usr/bin/env bash
python deeployTrainingRunner_tiled_siracusa.py \
  -t ./Tests/Models/Training/CCT_LoRA_SIZE1024/cct_lora_train/ \
  --optimizer-dir ./Tests/Models/Training/CCT_LoRA_SIZE1024/cct_lora_optimizer/ \
  --n-steps 1 --n-accum 4 \
  --l1 128000 --l2 4000000 --defaultMemLevel L2 \
  --tolerance 0.09 -vvv --plotMemAlloc "$@"

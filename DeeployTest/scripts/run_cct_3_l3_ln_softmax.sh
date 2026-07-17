#!/usr/bin/env bash
python deeployTrainingRunner_tiled_siracusa.py \
  -t ./Tests/Models/Training/CCT_3_LN_SOFTMAX/cct_3_train/ \
  --optimizer-dir ./Tests/Models/Training/CCT_3_LN_SOFTMAX/cct_3_optimizer/ \
  --n-steps 1 --n-accum 4 \
  --l1 128000 --defaultMemLevel L3 \
  --tolerance 0.09 -vvv --plotMemAlloc "$@"

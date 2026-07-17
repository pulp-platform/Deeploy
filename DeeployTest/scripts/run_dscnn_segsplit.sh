#!/usr/bin/env bash
python deeployTrainingRunner_tiled_siracusa.py \
  -t ./Tests/Models/Training/DSCNN_SEGSPLIT/dscnn_train/ \
  --optimizer-dir ./Tests/Models/Training/DSCNN_SEGSPLIT/dscnn_optimizer/ \
  --n-steps 1 --n-accum 4 \
  --l1 128000 --l2 2000000 --defaultMemLevel L2 \
  --tolerance 0.09 -vvv --plotMemAlloc "$@"

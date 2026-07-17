#!/usr/bin/env bash
python deeployTrainingRunner_tiled_siracusa.py \
  -t ./Tests/Models/Training/MobileNetV1_COST64/mobilenetv1_train/ \
  --optimizer-dir ./Tests/Models/Training/MobileNetV1_COST64/mobilenetv1_optimizer/ \
  --n-steps 1 --n-accum 4 \
  --l1 128000 --l2 2000000 --defaultMemLevel L3 \
  --tolerance 0.09 -vvv --plotMemAlloc "$@"

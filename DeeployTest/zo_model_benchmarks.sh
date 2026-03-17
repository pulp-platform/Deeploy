#!/bin/bash

# fp32 benchmarks
# LiteCNN
python3 deeployRunner_tiled_gap9.py -t Tests/Models/Noise/LiteCNN --profileTiling -s board > Tests/Models/Noise/LiteCNN.txt 2>&1
python3 deeployRunner_tiled_gap9.py -t Tests/Models/Noise/LiteCNN-Rad --profileTiling -s board > Tests/Models/Noise/LiteCNN-Rad.txt  2>&1
# SleepViT
python3 deeployRunner_tiled_gap9.py -t Tests/Models/Noise/SleepViT --profileTiling -s board > Tests/Models/Noise/SleepViT.txt 2>&1
python3 deeployRunner_tiled_gap9.py -t Tests/Models/Noise/SleepViT-Rad --profileTiling -s board > Tests/Models/Noise/SleepViT-Rad.txt  2>&1

# int8-mixed benchmarks
# LiteCNN
python3 deeployRunner_tiled_gap9.py -t Tests/Models/Noise/QLiteCNN --profileTiling -s board > Tests/Models/Noise/QLiteCNN.txt 2>&1
python3 deeployRunner_tiled_gap9.py -t Tests/Models/Noise/QLiteCNN-RQSRad --profileTiling -s board > Tests/Models/Noise/QLiteCNN-RQSRad.txt  2>&1
# SleepViT
python3 deeployRunner_tiled_gap9.py -t Tests/Models/Noise/QSleepViT --profileTiling -s board > Tests/Models/Noise/QSleepViT.txt 2>&1
python3 deeployRunner_tiled_gap9.py -t Tests/Models/Noise/QSleepViT-RQSRad --profileTiling -s board > Tests/Models/Noise/QSleepViT-RQSRad.txt  2>&1
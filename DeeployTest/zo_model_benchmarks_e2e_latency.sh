#!/bin/bash

# fp32 benchmarks
# LiteCNN
python3 deeployRunner_tiled_gap9.py -t Tests/Models/Noise/LiteCNN -s board > Tests/Models/Noise/LiteCNN_e2e_latency.txt 2>&1
python3 deeployRunner_tiled_gap9.py -t Tests/Models/Noise/LiteCNN-Rad -s board > Tests/Models/Noise/LiteCNN_e2e_latency.txt  2>&1
# SleepViT
python3 deeployRunner_tiled_gap9.py -t Tests/Models/Noise/SleepViT -s board > Tests/Models/Noise/SleepViT_e2e_latency.txt 2>&1
python3 deeployRunner_tiled_gap9.py -t Tests/Models/Noise/SleepViT-Rad -s board > Tests/Models/Noise/SleepViT-Rad_e2e_latency.txt  2>&1 
# int8-mixed benchmarks
# LiteCNN
python3 deeployRunner_tiled_gap9.py -t Tests/Models/Noise/QLiteCNN -s board > Tests/Models/Noise/QLiteCNN_e2e_latency.txt 2>&1
python3 deeployRunner_tiled_gap9.py -t Tests/Models/Noise/QLiteCNN-RQSRad -s board > Tests/Models/Noise/QLiteCNN-RQSRad_e2e_latency.txt  2>&1
# SleepViT
python3 deeployRunner_tiled_gap9.py -t Tests/Models/Noise/QSleepViT -s board > Tests/Models/Noise/QSleepViT_e2e_latency.txt 2>&1
python3 deeployRunner_tiled_gap9.py -t Tests/Models/Noise/QSleepViT-RQSRad -s board > Tests/Models/Noise/QSleepViT-RQSRad_e2e_latency.txt  2>&1
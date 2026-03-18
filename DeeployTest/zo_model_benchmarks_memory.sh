#!/bin/bash

# fp32 benchmarks
# LiteCNN
python3 deeployRunner_tiled_gap9.py -t Tests/Models/Noise/LiteCNN -s board --plotMemAlloc > Tests/Models/Noise/LiteCNN_memory.txt 2>&1
python3 deeployRunner_tiled_gap9.py -t Tests/Models/Noise/LiteCNN-Rad -s board --plotMemAlloc > Tests/Models/Noise/LiteCNN-Rad_memory.txt  2>&1
# SleepViT
python3 deeployRunner_tiled_gap9.py -t Tests/Models/Noise/SleepViT -s board --plotMemAlloc > Tests/Models/Noise/SleepViT_memory.txt 2>&1
python3 deeployRunner_tiled_gap9.py -t Tests/Models/Noise/SleepViT-Rad -s board --plotMemAlloc > Tests/Models/Noise/SleepViT-Rad_memory.txt  2>&1 
# int8-mixed benchmarks
# LiteCNN
python3 deeployRunner_tiled_gap9.py -t Tests/Models/Noise/QLiteCNN -s board --plotMemAlloc > Tests/Models/Noise/QLiteCNN_memory.txt 2>&1
python3 deeployRunner_tiled_gap9.py -t Tests/Models/Noise/QLiteCNN-RQSRad -s board --plotMemAlloc > Tests/Models/Noise/QLiteCNN-RQSRad_memory.txt  2>&1
# SleepViT
python3 deeployRunner_tiled_gap9.py -t Tests/Models/Noise/QSleepViT -s board --plotMemAlloc > Tests/Models/Noise/QSleepViT_memory.txt 2>&1
python3 deeployRunner_tiled_gap9.py -t Tests/Models/Noise/QSleepViT-RQSRad -s board --plotMemAlloc > Tests/Models/Noise/QSleepViT-RQSRad_memory.txt  2>&1
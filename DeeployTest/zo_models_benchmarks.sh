#!/bin/bash

# bias-less models
# python3 deeployRunner_tiled_gap9.py -t Tests/Models/Lite-CNN --profileTiling -s board > Tests/Models/Lite-CNN.txt 2>&1
# python3 deeployRunner_tiled_gap9.py -t Tests/Models/Lite-CNN-ZO --profileTiling -s board > Tests/Models/Lite-CNN-ZO.txt 2>&1
# python3 deeployRunner_tiled_gap9.py -t Tests/Models/LiteCNN-Eggroll --profileTiling -s board > Tests/Models/Lite-CNN-eggroll.txt 2>&1

# python3 deeployRunner_tiled_gap9.py -t Tests/Models/SleepConVit --profileTiling -s board > Tests/Models/SleepConVit.txt 2>&1
# python3 deeployRunner_tiled_gap9.py -t Tests/Models/SleepConVit-ZO --profileTiling -s board > Tests/Models/SleepConVit-ZO.txt 2>&1

# fp32 models with bias & cross-entropy loss
python3 deeployRunner_tiled_gap9.py -t Tests/Models/Noise/LiteCNN --profileTiling -s board > Tests/Models/Noise/LiteCNN.txt 2>&1
python3 deeployRunner_tiled_gap9.py -t Tests/Models/Noise/LiteCNN-Gaussian --profileTiling -s board > Tests/Models/Noise/LiteCNN-Gaussian.txt 2>&1
python3 deeployRunner_tiled_gap9.py -t Tests/Models/Noise/LiteCNN-Uniform --profileTiling -s board > Tests/Models/Noise/LiteCNN-Uniform.txt 2>&1
python3 deeployRunner_tiled_gap9.py -t Tests/Models/Noise/LiteCNN-Rademacher --profileTiling -s board > Tests/Models/Noise/LiteCNN-Rademacher.txt 2>&1
python3 deeployRunner_tiled_gap9.py -t Tests/Models/Noise/LiteCNN-Eggroll --profileTiling -s board > Tests/Models/Noise/LiteCNN-Eggroll.txt 2>&1

#int8 models with bias & cross-entropy loss
python3 deeployRunner_tiled_gap9.py -t Tests/Models/Noise/QLiteCNN --profileTiling -s board > Tests/Models/Noise/QLiteCNN.txt 2>&1
python3 deeployRunner_tiled_gap9.py -t Tests/Models/Noise/QLiteCNN-Gaussian --profileTiling -s board > Tests/Models/Noise/QLiteCNN-Gaussian.txt 2>&1
python3 deeployRunner_tiled_gap9.py -t Tests/Models/Noise/QLiteCNN-Uniform --profileTiling -s board > Tests/Models/Noise/QLiteCNN-Uniform.txt 2>&1
python3 deeployRunner_tiled_gap9.py -t Tests/Models/Noise/QLiteCNN-Rademacher --profileTiling -s board > Tests/Models/Noise/QLiteCNN-Rademacher.txt 2>&1
python3 deeployRunner_tiled_gap9.py -t Tests/Models/Noise/QLiteCNN-Eggroll --profileTiling -s board > Tests/Models/Noise/QLiteCNN-Eggroll.txt 2>&1
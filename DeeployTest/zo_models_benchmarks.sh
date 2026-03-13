#!/bin/bash

# bias-less models
python3 deeployRunner_tiled_gap9.py -t Tests/Models/Lite-CNN --profileTiling -s board > Tests/Models/Lite-CNN.txt 2>&1
python3 deeployRunner_tiled_gap9.py -t Tests/Models/Lite-CNN-ZO --profileTiling -s board > Tests/Models/Lite-CNN-ZO.txt 2>&1
python3 deeployRunner_tiled_gap9.py -t Tests/Models/LiteCNN-Eggroll --profileTiling -s board > Tests/Models/Lite-CNN-eggroll.txt 2>&1

python3 deeployRunner_tiled_gap9.py -t Tests/Models/SleepConVit --profileTiling -s board > Tests/Models/SleepConVit.txt 2>&1
python3 deeployRunner_tiled_gap9.py -t Tests/Models/SleepConVit-ZO --profileTiling -s board > Tests/Models/SleepConVit-ZO.txt 2>&1

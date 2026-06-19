#!/bin/bash

python3 deeployRunner_tiled_gap9.py -t Tests/Kernels/Noise/PerturbNormal --profileTiling -s board > Tests/Kernels/Noise/PerturbNormal.txt 2>&1
python3 deeployRunner_tiled_gap9.py -t Tests/Kernels/Noise/PerturbUniform --profileTiling -s board > Tests/Kernels/Noise/PerturbUniform.txt  2>&1
# Actually needs a different kernel implementation.
python3 deeployRunner_tiled_gap9.py -t Tests/Kernels/Noise/PerturbEggroll --profileTiling -s board > Tests/Kernels/Noise/PerturbEggroll.txt  2>&1

# python3 deeployRunner_tiled_gap9.py -t Tests/Kernels/Noise/PerturbEggroll-Uniform --profileTiling > Tests/Kernels/Noise/PerturbEggroll-Uniform.txt 2>&1
python3 deeployRunner_tiled_gap9.py -t Tests/Kernels/Noise/PerturbTriangle --profileTiling -s board > Tests/Kernels/Noise/PerturbTriangle.txt 2>&1
python3 deeployRunner_tiled_gap9.py -t Tests/Kernels/Noise/PerturbRademacher --profileTiling -s board > Tests/Kernels/Noise/PerturbRademacher.txt 2>&1
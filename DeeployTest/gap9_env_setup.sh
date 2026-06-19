#!/bin/bash

source ../venv/bin/activate
export GVSOC_INSTALL_DIR=/scratch/jansno/Deeploy/install/gap9-sdk/install/workstation
export MINIMALLOC_INSTALL_DIR=/scratch/jansno/Deeploy/install/minimalloc/
export LLVM_INSTALL_DIR="nope"
source ../install/gap9-sdk/configs/gap9_evk_audio.sh
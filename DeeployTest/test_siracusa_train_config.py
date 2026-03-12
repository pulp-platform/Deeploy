# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

PLATFORM_NAME = "Siracusa"
SIMULATOR = "gvsoc"
DEFAULT_CORES = 8

# Training tests for Siracusa (non-tiled, forward + backward + optimizer).
#
# Keys are relative to DeeployTest/Tests/.
# Each entry is a dict with:
#   n_train_steps          -- number of optimizer (weight-update) steps
#   n_accum_steps          -- mini-batches accumulated per optimizer step
#   num_data_inputs        -- number of inputs that change per mini-batch
#   optimizer_dir          -- (optional) relative path to optimizer network dir
#                             (resolved to DeeployTest/Tests/<optimizer_dir>)
TRAINING_TESTS = {
    "Models/Autoencoder_Train/autoencoder_train": {
        "n_train_steps": 1,
        "n_accum_steps": 1,
        "num_data_inputs": 2,
        "optimizer_dir": "Models/Autoencoder_Train/autoencoder_optimizer",
    },
    "Models/MLP_Train/simplemlp_train": {
        "n_train_steps": 1,
        "n_accum_steps": 8,
        "num_data_inputs": 2,
        "optimizer_dir": "Models/MLP_Train/simplemlp_optimizer",
    },
    "Models/Transformer_Train": {
        "n_train_steps": 1,
        "n_accum_steps": 1,
        "num_data_inputs": 2,
    },
    "Models/CCT_Train/CCT2_FT1": {
        "n_train_steps": 1,
        "n_accum_steps": 1,
        "num_data_inputs": 2,
    },
}

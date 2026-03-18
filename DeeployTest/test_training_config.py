# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

SIMULATOR = "gvsoc"
DEFAULT_CORES = 8

# ── Non-tiled model training (forward + backward + optimizer) ─────────────────
# Keys are relative to DeeployTest/Tests/.
TRAINING_TESTS = {
    "Siracusa": {
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
    },
    "GAP9": {
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
        # NOTE: Transformer_Train and CCT_Train disabled — single-output
        # SoftmaxCrossEntropyLoss not yet supported (pre-existing issue).
    },
}

# ── Tiled model training (SBTiler, forward + backward + optimizer) ────────────
# Same test set for Siracusa and GAP9.
TRAINING_TILED_TESTS = {
    "Models/SmallTransformer/tinytransformer_train": {
        "n_train_steps": None,
        "n_accum_steps": 4,
        "num_data_inputs": None,
    },
}

# ── Tiled gradient-operator kernel tests (SBTiler, single backward-pass ops) ──
# Same kernel set for Siracusa and GAP9.
GRAD_TILED_TESTS = {
    "Kernels/FP32/SoftmaxGrad":     [2000],
    "Kernels/FP32/ReluGrad":        [8000],
    "Kernels/FP32/GeLUGrad":        [2000],
    "Kernels/FP32/LayernormGrad":   [4000],
    "Kernels/FP32/MaxpoolGrad":     [2000],
    "Kernels/FP32/AveragePoolGrad": [4000],
    "Kernels/FP32/ConvGrad":        [2000],
    "Kernels/FP32/DWConvGrad":      [8000],
    "Kernels/FP32/PWConvGrad":      [8000],
}

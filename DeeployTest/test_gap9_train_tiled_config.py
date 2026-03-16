# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

PLATFORM_NAME = "GAP9"
SIMULATOR = "gvsoc"
DEFAULT_CORES = 8

# Tiled training tests for GAP9 (SBTiler, forward + backward + optimizer).
#
# Keys are relative to DeeployTest/Tests/.
# Each entry is a dict with:
#   n_train_steps          -- number of optimizer (weight-update) steps; None = auto-detect
#   n_accum_steps          -- mini-batches accumulated per optimizer step; None = auto-detect
#   num_data_inputs        -- number of inputs that change per mini-batch; None = auto-detect
#   optimizer_dir          -- (optional) relative path to optimizer network dir
TRAINING_TILED_TESTS = {
    "Models/SmallTransformer/tinytransformer_train": {
        "n_train_steps": None,
        "n_accum_steps": 4,
        "num_data_inputs": None,
    },
}

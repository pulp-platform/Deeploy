# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

PLATFORM_NAME = "Siracusa"
SIMULATOR = "gvsoc"
DEFAULT_CORES = 8
DEFAULT_N_TRAIN_STEPS = 1
DEFAULT_N_ACCUM_STEPS = 1

# Training tests: test_name -> (n_train_steps, n_accum_steps, training_num_data_inputs)
# test_name is relative to DeeployTest/Tests/
# NOTE: simplemlp_train uses an external path — handled specially in test function
TRAINING_TESTS = {
    "Models/SimpleMLP_Train": (1, 1, 2),
    "Models/Transformer_Train": (1, 1, 2),
    "Models/CCT_Train/CCT2_FT1": (1, 1, 2),
}

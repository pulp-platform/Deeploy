# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from Deeploy.DeeployTypes import NodeTemplate

# Forward pass (training mode): BatchNormInternal
# Inputs:  X, gamma, beta, running_mean, running_var
# Outputs: Y, saved_mean, saved_inv_std  (updated_running_mean/var have no consumers)
batchNormInternalTemplate = NodeTemplate("""
// BatchNormInternal (Name: ${nodeName}, Op: ${nodeOp})
PULP_BatchNormInternal_fp32(
    ${data_in},
    ${scale},
    ${bias},
    ${running_mean},
    ${running_var},
    ${data_out},
    ${saved_mean},
    ${saved_inv_std},
    ${N},
    ${C},
    ${H_in},
    ${W_in},
    ${epsilon}f,
    ${momentum}f
);
""")

# Backward pass: BatchNormalizationGrad
# Inputs:  dY, X, gamma, saved_mean, saved_inv_std
# Outputs: dX, dgamma, dbeta
batchNormGradTemplate = NodeTemplate("""
// BatchNormalizationGrad (Name: ${nodeName}, Op: ${nodeOp})
PULP_BatchNormGrad_fp32(
    ${dY},
    ${X},
    ${gamma},
    ${saved_mean},
    ${saved_inv_std},
    ${dX},
    ${dgamma},
    ${dbeta},
    ${N},
    ${C},
    ${H_in},
    ${W_in},
    ${epsilon}f
);
""")

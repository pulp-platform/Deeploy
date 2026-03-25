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

# Split BN forward: WelfordReduce
welfordReduceTemplate = NodeTemplate("""
// WelfordReduce (Name: ${nodeName}, Op: ${nodeOp})
PULP_WelfordReduce_fp32(
    ${data_in},
    ${saved_mean},
    ${saved_inv_std},
    ${N},
    ${C},
    ${H_in},
    ${W_in},
    ${epsilon}f
);
""")

# Split BN forward: ChannelNormalize
channelNormalizeTemplate = NodeTemplate("""
// ChannelNormalize (Name: ${nodeName}, Op: ${nodeOp})
PULP_ChannelNormalize_fp32(
    ${data_in},
    ${saved_mean},
    ${saved_inv_std},
    ${gamma},
    ${beta},
    ${data_out},
    ${N},
    ${C},
    ${H_in},
    ${W_in}
);
""")

# Split BN backward: BNGradReduce
bnGradReduceTemplate = NodeTemplate("""
// BNGradReduce (Name: ${nodeName}, Op: ${nodeOp})
PULP_BNGradReduce_fp32(
    ${dY},
    ${X},
    ${saved_mean},
    ${saved_inv_std},
    ${dgamma},
    ${dbeta},
    ${N},
    ${C},
    ${H_in},
    ${W_in}
);
""")

# Split BN backward: BNGradNormalize
bnGradNormalizeTemplate = NodeTemplate("""
// BNGradNormalize (Name: ${nodeName}, Op: ${nodeOp})
PULP_BNGradNormalize_fp32(
    ${dY},
    ${X},
    ${saved_mean},
    ${saved_inv_std},
    ${gamma},
    ${dgamma},
    ${dbeta},
    ${dX},
    ${N},
    ${C},
    ${H_in},
    ${W_in},
    ${N_total_inv}f
);
""")

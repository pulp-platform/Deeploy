# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from Deeploy.DeeployTypes import NodeTemplate

# Forward: GlobalAveragePool
# Inputs:  data_in [N, C, H, W]
# Outputs: data_out [N, C, 1, 1]  (stored as N*C elements)
globalAveragePoolTemplate = NodeTemplate("""
// GlobalAveragePool (Name: ${nodeName}, Op: ${nodeOp})
PULP_GlobalAveragePool_fp32(
    ${data_in},
    ${data_out},
    ${batch},
    ${channels},
    ${dim_im_in_x},
    ${dim_im_in_y}
);
""")

# Backward: GlobalAveragePoolGrad
# Inputs:  dY [N, C, 1, 1]  (stored as N*C elements)
# Outputs: dX [N, C, H, W]
globalAveragePoolGradTemplate = NodeTemplate("""
// GlobalAveragePoolGrad (Name: ${nodeName}, Op: ${nodeOp})
PULP_GlobalAveragePoolGrad_fp32(
    ${dY},
    ${dX},
    ${batch},
    ${channels},
    ${H},
    ${W}
);
""")

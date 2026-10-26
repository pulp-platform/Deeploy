# SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from Deeploy.DeeployTypes import NodeTemplate

referenceTemplate = NodeTemplate("""
<%
batchOffsetIn  = ch_im_in  * dim_im_in_x  * dim_im_in_y
batchOffsetOut = ch_im_out * dim_im_out_x * dim_im_out_y
%>
// 2D FP32 Conv (Name: ${nodeName}, Op: ${nodeOp})
// Carfield_Conv2D_tiled stages the operands in L1 through the system iDMA;
// Carfield_Conv is the naive DDR-resident reference. Same signature, so
// swapping the identifier back is the whole revert.
for (uint32_t n = 0; n < ${batch}; ++n) {
    Carfield_Conv2D_tiled(${data_in} + n * ${batchOffsetIn},
                          ${weight},
                          ${bias},
                          ${data_out} + n * ${batchOffsetOut},
                          ${ch_im_out}, ${ch_im_in},
                          ${dim_im_in_x}, ${dim_im_in_y},
                          ${dim_kernel_x}, ${dim_kernel_y},
                          core_id, numThreads);
}
""")

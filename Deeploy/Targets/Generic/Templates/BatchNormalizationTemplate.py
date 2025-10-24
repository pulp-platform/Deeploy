# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from Deeploy.DeeployTypes import NodeTemplate

referenceTemplate = NodeTemplate("""
// BatchNorm (Name: ${nodeName}, Op: ${nodeOp})
BEGIN_SINGLE_CORE
    BatchNorm_fp32(
        ${data_in}, ${scale}, ${bias}, ${mean}, ${variance},
        ${data_out}, ${batch_size}, ${channel_size}, ${window_size}
    );
END_SINGLE_CORE
""")

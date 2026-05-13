# SPDX-FileCopyrightText: 2021 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from Deeploy.DeeployTypes import NodeTemplate

referenceTemplate = NodeTemplate("""
// Add (Name: ${nodeName}, Op: ${nodeOp})
BEGIN_SINGLE_CORE
    for (uint32_t i = 0; i < ${size}; i++) {
        ${data_out}[i] = ceilf(${data_in}[i]);
    }
END_SINGLE_CORE
""")

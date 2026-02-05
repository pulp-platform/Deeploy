# SPDX-FileCopyrightText: 2022 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from Deeploy.DeeployTypes import NodeTemplate

referenceTemplate = NodeTemplate("""
// Float Mul (Name: ${nodeName}, Op: ${nodeOp})
BEGIN_SINGLE_CORE
    for (uint32_t i=0;i<${size};i++){
        ${C}[i] = ${A}[i] * ${B}[0];
    }
END_SINGLE_CORE
""")

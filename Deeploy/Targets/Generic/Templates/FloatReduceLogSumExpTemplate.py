# SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from Deeploy.DeeployTypes import NodeTemplate

referenceTemplate = NodeTemplate("""
// ReduceLogSumExp (Name: ${nodeName}, Op: ${nodeOp})
SINGLE_CORE ReduceLogSumExp_fp32_fp32(${data_in}, ${data_out}, ${outerSize}, ${axisLength}, ${innerSize});
""")

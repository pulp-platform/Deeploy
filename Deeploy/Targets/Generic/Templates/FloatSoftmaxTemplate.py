# SPDX-FileCopyrightText: 2021 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from Deeploy.DeeployTypes import NodeTemplate

referenceTemplate = NodeTemplate("""
// Softmax (Name: ${nodeName}, Op: ${nodeOp})
SINGLE_CORE Softmax_fp${data_in_type.referencedType.typeWidth}_fp${data_out_type.referencedType.typeWidth}(${data_in}, ${data_out}, ${size}, ${lastDimLength});
""")

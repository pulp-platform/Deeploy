# SPDX-FileCopyrightText: 2023 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from Deeploy.DeeployTypes import NodeTemplate

referenceTemplate = NodeTemplate("""
// Division (Name: ${nodeName}, Op: ${nodeOp})
SINGLE_CORE Div_fp${input1_type.referencedType.typeWidth}_fp${input2_type.referencedType.typeWidth}_fp${output_type.referencedType.typeWidth}(${input1}, ${input2}, ${output}, ${size});
""")

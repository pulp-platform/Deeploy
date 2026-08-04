# SPDX-FileCopyrightText: 2023 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from Deeploy.DeeployTypes import NodeTemplate

referenceTemplate = NodeTemplate("""
// Division (Name: ${nodeName}, Op: ${nodeOp})
SINGLE_CORE Div_fp${A_type.referencedType.typeWidth}_fp${B_type.referencedType.typeWidth}_fp${C_type.referencedType.typeWidth}(${A}, ${B}, ${C}, ${size});
""")

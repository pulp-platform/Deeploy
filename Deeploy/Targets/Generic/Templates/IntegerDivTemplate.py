# SPDX-FileCopyrightText: 2022 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from Deeploy.DeeployTypes import NodeTemplate

referenceTemplate = NodeTemplate("""
// Integer Division (Name: ${nodeName}, Op: ${nodeOp})
SINGLE_CORE Div_s${A_type.referencedType.typeWidth}_s${B_type.referencedType.typeWidth}(${A}, ${B}, ${sizeA}, ${sizeB}, ${nomStep}, ${denomStep}, ${C}, ${Delta}, ${eps}, ${eta});
""")

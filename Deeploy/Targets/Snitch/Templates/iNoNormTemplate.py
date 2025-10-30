# SPDX-FileCopyrightText: 2024 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from Deeploy.CommonExtensions.NodeTemplate import ElementwiseTemplate

referenceTemplate = ElementwiseTemplate("""
// iNoNorm (Name: ${nodeName}, Op: ${nodeOp})
SnitchiNoNorm_s${data_in_type.referencedType.typeWidth}_s${data_out_type.referencedType.typeWidth}(${data_in}, ${data_out}, ${weights}, ${bias}, ${size}, ${mul}, ${log2D});
""")

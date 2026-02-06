# SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, List, Tuple

from Deeploy.DeeployTypes import NetworkContext, NodeTemplate, OperatorRepresentation


class FloatMulTemplate(NodeTemplate):
    """Template for FP32 Mul operation with dynamic template selection."""

    def __init__(self, templateStr):
        super().__init__(templateStr)

    def alignToContext(self, ctxt: NetworkContext,
                       operatorRepresentation: OperatorRepresentation) -> Tuple[NetworkContext, Dict, List[str]]:

        # Check if scalar broadcasting
        is_scalar = operatorRepresentation.get('is_scalar', False)

        # Dynamically select template based on is_scalar flag
        if is_scalar:
            # Use scalar broadcasting version
            self.templateStr = FloatMulScalarTemplateStr
        else:
            # Use element-wise version
            self.templateStr = FloatMulTemplateStr

        return ctxt, operatorRepresentation, []


# Template for element-wise multiplication
# Note: MulParser uses A, B, C for input1, input2, output respectively
FloatMulTemplateStr = r"""
Mul_fp32(${A}, ${B}, ${C}, ${size});
"""

# Template for scalar broadcasting (optimized)
FloatMulScalarTemplateStr = r"""
{
    float32_t scalar = ${B}[0];
    Mul_fp32_scalar(${A}, scalar, ${C}, ${size});
}
"""

# Create reference template with default (element-wise)
referenceTemplate = FloatMulTemplate(FloatMulTemplateStr)

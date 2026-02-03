# SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, List, Tuple

from Deeploy.DeeployTypes import NetworkContext, NodeTemplate, OperatorRepresentation


class FloatDivTemplate(NodeTemplate):
    """Template for FP32 Div operation with dynamic template selection."""

    def __init__(self, templateStr):
        super().__init__(templateStr)

    def alignToContext(self, ctxt: NetworkContext,
                       operatorRepresentation: OperatorRepresentation) -> Tuple[NetworkContext, Dict, List[str]]:

        # Check if scalar broadcasting
        is_scalar = operatorRepresentation.get('is_scalar', False)

        # Dynamically select template based on is_scalar flag
        if is_scalar:
            # Use scalar broadcasting version
            self.templateStr = FloatDivScalarTemplateStr
        else:
            # Use element-wise version
            self.templateStr = FloatDivTemplateStr

        return ctxt, operatorRepresentation, []


# Template for element-wise division
FloatDivTemplateStr = r"""
Div_fp32(${input1}, ${input2}, ${output}, ${size});
"""

# Template for scalar broadcasting (optimized)
FloatDivScalarTemplateStr = r"""
{
    float32_t scalar = ${input2}[0];
    Div_fp32_scalar(${input1}, scalar, ${output}, ${size});
}
"""

# Create reference template with default (element-wise)
referenceTemplate = FloatDivTemplate(FloatDivTemplateStr)

# SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, List, Tuple

from mako.template import Template as MakoTemplate

from Deeploy.DeeployTypes import NetworkContext, NodeTemplate, OperatorRepresentation


class FloatMulTemplate(NodeTemplate):
    """Template for FP32 Mul operation with dynamic template selection."""

    def __init__(self, templateStr):
        super().__init__(templateStr)

    def alignToContext(self, ctxt: NetworkContext,
                       operatorRepresentation: OperatorRepresentation) -> Tuple[NetworkContext, Dict, List[str]]:

        # Check if scalar broadcasting
        is_scalar = operatorRepresentation.get('is_scalar', False)

        # IMPORTANT: Must recompile self.template (Mako Template object),
        # not just assign self.templateStr. NodeTemplate.generate() uses
        # the pre-compiled self.template, not self.templateStr.
        if is_scalar:
            self.template = MakoTemplate(FloatMulScalarTemplateStr, strict_undefined = True)
        else:
            self.template = MakoTemplate(FloatMulTemplateStr, strict_undefined = True)

        return ctxt, operatorRepresentation, []


# Template for element-wise multiplication
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

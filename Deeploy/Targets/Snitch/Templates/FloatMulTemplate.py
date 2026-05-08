# SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from Deeploy.DeeployTypes import NodeTemplate

referenceTemplate = NodeTemplate(r"""
% if is_scalar:
{
    float32_t scalar = ${B}[0];
    Mul_fp32_scalar(${A}, scalar, ${C}, ${size});
}
% else:
Mul_fp32(${A}, ${B}, ${C}, ${size});
% endif
""")

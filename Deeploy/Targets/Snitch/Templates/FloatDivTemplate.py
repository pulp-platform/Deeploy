# SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from Deeploy.DeeployTypes import NodeTemplate

referenceTemplate = NodeTemplate(r"""
Div_fp32(${A}, ${B}, ${C}, ${size}, ${1 if is_scalar else 0});
""")

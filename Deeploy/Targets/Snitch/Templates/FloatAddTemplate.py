# SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from Deeploy.DeeployTypes import NodeTemplate

# Plain multi-core element-wise add. Works regardless of where operands
# live (L1/L2), so it is the default reference path.
referenceTemplate = NodeTemplate(r"""
Add_fp32(${data_in_1}, ${data_in_2}, ${data_out}, ${size}, ${1 if is_scalar else 0});
""")

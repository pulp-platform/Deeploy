# ----------------------------------------------------------------------
# File: iLeakyReLUTemplate.py
#
# SoCDAML Part III - TA reference solution.
# Mako template that emits the call to the PULP iLeakyReLU C kernel.
#
# Drop this file into:
#   Deeploy/Targets/PULPOpen/Templates/iLeakyReLUTemplate.py
# ----------------------------------------------------------------------
# SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from Deeploy.DeeployTypes import NodeTemplate


class _iLeakyReLUTemplate(NodeTemplate):

    def __init__(self, templateStr):
        super().__init__(templateStr)


referenceTemplate = _iLeakyReLUTemplate("""
// iLeakyReLU (Name: ${nodeName}, Op: ${nodeOp})
PULPiLeakyReLU_i8_i8(${data_in}, ${data_out}, ${size}, ${mul}, ${shift});
""")

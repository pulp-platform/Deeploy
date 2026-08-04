# ----------------------------------------------------------------------
# File: iLeakyReLUTemplate.py  (SoCDAML Part III - Step 4 skeleton)
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


# TODO(student): fill in the Mako template body so it emits a single
# call to your C kernel:
#
#     PULPiLeakyReLU_i8_i8(<data_in>, <data_out>, <size>, <mul>, <shift>);
#
# All five `${...}` substitutions correspond to keys you populated in
# the parser (or that Deeploy fills automatically for tensor names).
referenceTemplate = _iLeakyReLUTemplate("""
// iLeakyReLU (Name: ${nodeName}, Op: ${nodeOp})
// TODO(student): emit the kernel call here.
""")

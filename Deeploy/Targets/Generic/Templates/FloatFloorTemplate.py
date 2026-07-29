# SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0
from Deeploy.Targets.Generic.Templates.UnaryTemplate import _UnaryTemplate

referenceTemplate = _UnaryTemplate("""
// Floor (Name: ${nodeName}, Op: ${nodeOp})
Floor_fp${type_width}_fp${type_width}(${data_in}, ${data_out}, ${size});
""")

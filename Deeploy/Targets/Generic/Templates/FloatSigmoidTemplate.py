# SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0
from Deeploy.Targets.Generic.Templates.FloatUnaryTemplate import _FloatUnaryTemplate

referenceTemplate = _FloatUnaryTemplate("""
// Sigmoid (Name: ${nodeName}, Op: ${nodeOp})
Sigmoid_fp${type_width}_fp${type_width}(${data_in}, ${data_out}, ${size});
""")

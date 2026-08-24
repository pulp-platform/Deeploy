# SPDX-FileCopyrightText: 2023 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0
from Deeploy.Targets.Generic.Templates.UnaryTemplate import _UnaryTemplate

referenceTemplate = _UnaryTemplate("""
// Relu (Name: ${nodeName}, Op: ${nodeOp})
SINGLE_CORE Relu_fp${type_width}_fp${type_width}(${data_in}, ${data_out}, ${size});
""")

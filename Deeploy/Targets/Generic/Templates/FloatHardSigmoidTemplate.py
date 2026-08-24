# SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0
from Deeploy.Targets.Generic.Templates.UnaryTemplate import _UnaryTemplate

referenceTemplate = _UnaryTemplate("""
// HardSigmoid (Name: ${nodeName}, Op: ${nodeOp})
HardSigmoid_fp${type_width}_fp${type_width}(${data_in}, ${data_out}, ${alpha}, ${beta}, ${size});
""")

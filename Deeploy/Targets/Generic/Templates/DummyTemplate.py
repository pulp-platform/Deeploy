# SPDX-FileCopyrightText: 2021 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from Deeploy.DeeployTypes import NodeTemplate

referenceTemplate = NodeTemplate("""
// Dummy (Name: ${nodeName}, Op: ${nodeOp})
SINGLE_CORE void DummyOP(${data_in}, ${data_out}, ${size});
""")

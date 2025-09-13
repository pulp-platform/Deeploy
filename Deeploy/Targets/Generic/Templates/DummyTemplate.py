# Copyright (C) 2025, ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

from Deeploy.DeeployTypes import NodeTemplate

referenceTemplate = NodeTemplate("""
// Dummy (Name: ${nodeName}, Op: ${nodeOp})
SINGLE_CORE void DummyOP(${data_in}, ${data_out}, ${size});
""")

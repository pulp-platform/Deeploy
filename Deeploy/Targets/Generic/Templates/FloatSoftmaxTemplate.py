# Copyright (C) 2025, ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

from Deeploy.DeeployTypes import NodeTemplate

referenceTemplate = NodeTemplate("""
// Softmax (Name: ${nodeName}, Op: ${nodeOp})
SINGLE_CORE Softmax_fp${data_in_type.referencedType.typeWidth}_fp${data_out_type.referencedType.typeWidth}(${data_in}, ${data_out}, ${size}, ${lastDimLength});
""")

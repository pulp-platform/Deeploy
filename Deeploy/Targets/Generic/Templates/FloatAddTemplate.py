# Copyright (C) 2025, ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

from Deeploy.DeeployTypes import NodeTemplate

referenceTemplate = NodeTemplate("""
// Add (Name: ${nodeName}, Op: ${nodeOp})
BEGIN_SINGLE_CORE
    for (uint32_t i=0;i<${size};i++){
        ${data_out}[i] = ${data_in_1}[i] + ${data_in_2}[i];
    }
END_SINGLE_CORE
""")

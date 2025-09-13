# Copyright (C) 2025, ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

from Deeploy.DeeployTypes import NodeTemplate

referenceTemplate = NodeTemplate("""
// Gather (Name: ${nodeName}, Op: ${nodeOp})
<%
width = int(data_in_type.referencedType.typeWidth/8)
%>
BEGIN_SINGLE_CORE
for (uint32_t i=0; i<${batch}; ++i) {
    memcpy(${data_out} + i * ${axis_length}, ${data_in} + i * ${batch_length} + ${index} * ${axis_length}, ${axis_length} * ${width});
}
END_SINGLE_CORE
""")

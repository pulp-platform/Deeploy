# SPDX-FileCopyrightText: 2023 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from Deeploy.DeeployTypes import NodeTemplate

referenceTemplate = NodeTemplate("""
// Float Layernorm (Name: ${nodeName}, Op: ${nodeOp})
PULP_Layernorm_fp${data_in_type.referencedType.typeWidth}_fp${data_out_type.referencedType.typeWidth}(
    ${data_in},
    ${data_out},
    ${weight},
    ${bias},
    ${epsilon},
    ${size},
    ${lastDimLength}
);
""")
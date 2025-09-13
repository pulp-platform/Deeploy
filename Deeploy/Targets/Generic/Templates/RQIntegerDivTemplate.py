# Copyright (C) 2025, ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

from Deeploy.DeeployTypes import NodeTemplate

referenceTemplate = NodeTemplate("""
// RQIntegerDiv (Name: ${nodeName}, Op: ${nodeOp})
SINGLE_CORE RQDiv_s${A_type.referencedType.typeWidth}_s${C_type.referencedType.typeWidth}(${A}, ${B}, ${sizeA}, ${sizeB}, ${nomStep}, ${denomStep}, ${C}, ${Delta}, ${eps}, ${eta}, *${requant_mul}, *${requant_add}, *${requant_div});
""")

# Copyright (C) 2025, ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

from Deeploy.DeeployTypes import NodeTemplate


class _iNoNormTemplate(NodeTemplate):

    def __init__(self, templateStr):
        super().__init__(templateStr)


referenceTemplate = _iNoNormTemplate("""
// iNoNorm (Name: ${nodeName}, Op: ${nodeOp})
SnitchiNoNorm_s${data_in_type.referencedType.typeWidth}_s${data_out_type.referencedType.typeWidth}(${data_in}, ${data_out}, ${weights}, ${bias}, ${size}, ${mul}, ${log2D});
""")

# Copyright (C) 2025, ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, Tuple

from Deeploy.DeeployTypes import NetworkContext, NodeTemplate, OperatorRepresentation


class _iLayerNormTemplate(NodeTemplate):

    def __init__(self, templateStr):
        super().__init__(templateStr)

    def alignToContext(self, ctxt: NetworkContext,
                       operatorRepresentation: OperatorRepresentation) -> Tuple[NetworkContext, Dict]:

        data_in = ctxt.lookup(operatorRepresentation['data_in'])
        operatorRepresentation['input_offset'] = (data_in._signed == 0) * int(data_in.nLevels / 2)

        return ctxt, operatorRepresentation, []


referenceTemplate = _iLayerNormTemplate("""
// iLayernorm (Name: ${nodeName}, Op: ${nodeOp})
SINGLE_CORE Layernorm_s${data_in_type.referencedType.typeWidth}_s${data_out_type.referencedType.typeWidth}(${data_in}, ${data_out}, ${weight}, ${bias}, ${input_offset}, ${size}, ${lastDimLength}, ${log2D});
""")

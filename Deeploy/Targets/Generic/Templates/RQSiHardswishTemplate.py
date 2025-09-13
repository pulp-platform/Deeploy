# Copyright (C) 2025, ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, List, Tuple

from Deeploy.DeeployTypes import NetworkContext, NodeTemplate, OperatorRepresentation


class _RQSiHardswishTemplate(NodeTemplate):

    def __init__(self, templateStr):
        super().__init__(templateStr)

    def alignToContext(self, ctxt: NetworkContext,
                       operatorRepresentation: OperatorRepresentation) -> Tuple[NetworkContext, Dict, List[str]]:

        data_in = ctxt.lookup(operatorRepresentation['data_in'])
        data_out = ctxt.lookup(operatorRepresentation['data_out'])
        operatorRepresentation['input_offset'] = (data_in._signed == 0) * int(data_in.nLevels / 2)
        operatorRepresentation['output_offset'] = (data_out._signed == 0) * int(data_out.nLevels / 2)

        return ctxt, operatorRepresentation, []


referenceTemplate = _RQSiHardswishTemplate("""
// RequantizediHardswish (Name: ${nodeName}, Op: ${nodeOp})
SINGLE_CORE RQiHardswish_s${data_in_type.referencedType.typeWidth}_s${data_out_type.referencedType.typeWidth}(${data_in}, ${data_out}, ${size}, ${one_over_six}, ${three}, ${six}, ${input_offset}, ${output_offset}, ${mul}, ${add}, ${shift});
""")

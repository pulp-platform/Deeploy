# Copyright (C) 2025, ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, List, Tuple

from Deeploy.DeeployTypes import NetworkContext, NodeTemplate, OperatorRepresentation

from .CMSISUtils import bindFCParams


class _GEMM_8_Template(NodeTemplate):

    def __init__(self, templateStr):
        super().__init__(templateStr)

    def alignToContext(self, ctxt: NetworkContext,
                       operatorRepresentation: OperatorRepresentation) -> Tuple[NetworkContext, Dict, List[str]]:

        # Hoist the structs to the global ctxt
        data_in = ctxt.lookup(operatorRepresentation['A'])
        weight = ctxt.lookup(operatorRepresentation['B'])

        ctxt, operatorRepresentation, nameList = bindFCParams(ctxt, operatorRepresentation['data_out'],
                                                              operatorRepresentation['mul'],
                                                              operatorRepresentation['shift'], data_in, weight,
                                                              operatorRepresentation)

        return ctxt, operatorRepresentation, nameList


Linear_8_Template = _GEMM_8_Template("""
// GEMM
int8_t* ref_${data_out}_${A} = ${A};
int8_t* ref_${data_out}_${B} = ${B};
int8_t* ref_${data_out}_${data_out} = ${data_out};
for(int i=0;i<${batch};i++){
    arm_fully_connected_s8(&${ctxt}, &${fc_params}, &${quant_params}, &${input_dims}, ref_${data_out}_${A}, &${filter_dims}, ref_${data_out}_${B}, &${bias_dims}, ${C}, &${output_dims}, ref_${data_out}_${data_out});
    ref_${data_out}_${A} += ${M} * ${N};
    ref_${data_out}_${B} += ${N} * ${O};
    ref_${data_out}_${data_out} += ${M} * ${O};
}
""")


class _GEMM_16_Template(NodeTemplate):

    def __init__(self, templateStr):
        super().__init__(templateStr)

    def alignToContext(self, ctxt: NetworkContext,
                       operatorRepresentation: OperatorRepresentation) -> Tuple[NetworkContext, Dict, List[str]]:
        # Hoist the structs to the global ctxt
        data_in = ctxt.lookup(operatorRepresentation['A'])
        weight = ctxt.lookup(operatorRepresentation['B'])

        ctxt, operatorRepresentation, nameList = bindFCParams(ctxt, operatorRepresentation['data_out'],
                                                              operatorRepresentation['mul'],
                                                              operatorRepresentation['shift'], data_in, weight,
                                                              operatorRepresentation)

        return ctxt, operatorRepresentation, nameList


Linear_16_Template = _GEMM_16_Template("""
// FC
arm_fully_connected_s16(&${ctxt}, &${fc_params}, &${quant_params}, &${input_dims}, ${A}, &${filter_dims}, ${B}, &${bias_dims}, ${C}, &${output_dims}, ${data_out});
""")

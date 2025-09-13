# Copyright (C) 2025, ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, List, Tuple, Union

from ortools.constraint_solver.pywrapcp import IntVar

from Deeploy.DeeployTypes import NetworkContext, NodeTemplate, OperatorRepresentation


class iSoftmaxPreAllocatedBuffTemplate(NodeTemplate):

    @staticmethod
    def computeTransientBuffersSize(
            ctxt: NetworkContext,
            operatorRepresentation: OperatorRepresentation) -> List[Tuple[str, Union[int, IntVar]]]:

        lastDimBuffer_dim = 8 * 4 * operatorRepresentation['lastDimLength']
        lastDimBuffer_name = operatorRepresentation['nodeName'] + "_lastDimBuffer"
        return [(lastDimBuffer_name, lastDimBuffer_dim)]

    def hoistTransientBuffers(self, ctxt: NetworkContext,
                              operatorRepresentation: OperatorRepresentation) -> Tuple[NetworkContext, Dict, List[str]]:
        lastDimBuffer_name, lastDimBuffer_dim = iSoftmaxPreAllocatedBuffTemplate.computeTransientBuffersSize(
            ctxt, operatorRepresentation)[0]
        ctxt.hoistTransientBuffer(lastDimBuffer_name, lastDimBuffer_dim)

        operatorRepresentation['lastDimBuffer'] = lastDimBuffer_name
        operatorRepresentation['lastDimBufferSize'] = lastDimBuffer_dim
        return ctxt, operatorRepresentation, [lastDimBuffer_name]

    def alignToContext(self, ctxt: NetworkContext,
                       operatorRepresentation: OperatorRepresentation) -> Tuple[NetworkContext, Dict, List[str]]:

        signedI = ctxt.lookup(operatorRepresentation['data_in'])._type.referencedType.typeMin < 0
        signedO = ctxt.lookup(operatorRepresentation['data_out'])._type.referencedType.typeMin < 0

        operatorRepresentation['input_signed'] = signedI
        operatorRepresentation['output_signed'] = signedO

        return ctxt, operatorRepresentation, []

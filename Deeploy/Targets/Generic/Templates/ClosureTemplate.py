# Copyright (C) 2025, ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

from Deeploy.DeeployTypes import NodeTemplate, OperatorRepresentation


class ClosureTemplate(NodeTemplate, OperatorRepresentation):

    def __init__(self, templateStr):
        super().__init__(templateStr)


referenceTemplate = ClosureTemplate("""
void ${nodeName}_closure(void* {nodeName}_args){
${nodeName}_args_t* args = (${nodeName}_args_t*) {nodeName}_args;
% for argName, argType in closureStructArgs.items():
${argType.typeName} ${argName} = args->${argName};
% endfor
${functionCall}
}
""")

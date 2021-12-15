# ----------------------------------------------------------------------
#
# File: ClosureTemplate.py
#
# Last edited: 15.03.2023
#
# Copyright (C) 2023, ETH Zurich and University of Bologna.
#
# Author: Moritz Scherer, ETH Zurich
#
# ----------------------------------------------------------------------
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the License); you may
# not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an AS IS BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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

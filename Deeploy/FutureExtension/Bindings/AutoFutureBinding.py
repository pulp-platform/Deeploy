# ----------------------------------------------------------------------
#
# File: PULPDMAFutureBinding.py
#
# Last edited: 08.06.2023
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

from typing import Optional

from Deeploy.DeeployTypes import CodeTransformation, NetworkContext, NodeTemplate, NodeTypeChecker
from Deeploy.FutureExtension.Bindings.FutureBinding import FutureBinding
from Deeploy.FutureExtension.Future import Future


class AutoFutureBinding(FutureBinding):

    def __init__(self,
                 typeChecker: NodeTypeChecker,
                 template: NodeTemplate,
                 codeTransformer: CodeTransformation,
                 stateReferenceType: Optional = None):
        super().__init__(typeChecker, template, codeTransformer)

        futureOutputs = [idx for idx, output in enumerate(self.typeChecker.output_types) if issubclass(output, Future)]

        if len(futureOutputs) > 1:
            raise Exception(f"{self} assigns more than one future output!")

        if len(futureOutputs) == 1:
            self.stateReferenceType = self.typeChecker.output_types[futureOutputs[0]].stateReferenceType

        self.futureOutputs = futureOutputs

    def assignStateReferenceElement(self, ctxt) -> NetworkContext:

        if len(self.futureOutputs) > 1:
            raise Exception(f"{self} assigns more than one future output!")

        if len(self.futureOutputs) == 0:
            return ctxt

        for codeSnippet in self.executionBlock.codeSnippets:
            operatorRepresentation = codeSnippet.operatorRepresentation

            stateElementCandidates = []
            for key, value in operatorRepresentation.items():
                if type(value) == str and (ctxt.is_local(value) or ctxt.is_global(value)):
                    reference = ctxt.lookup(value)
                    if isinstance(reference._instance,
                                  self.stateReferenceType) and reference not in stateElementCandidates:
                        stateElementCandidates.append(reference)

            if len(stateElementCandidates) == 1:
                print(f"WARNING: Automagically assigning state Element of {self}")
                for key, value in operatorRepresentation.items():
                    if type(value) == str and (ctxt.is_local(value) or ctxt.is_global(value)):
                        reference = ctxt.lookup(value)
                        if issubclass(reference._type, Future) and not hasattr(reference._instance, "stateReference"):
                            reference._instance.assignStateReference(stateElementCandidates[0], ctxt)

            else:
                raise Exception(f"Can't assign a unique state element to {self} automagically!")

        return ctxt

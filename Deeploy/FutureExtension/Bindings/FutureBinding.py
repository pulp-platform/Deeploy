# ----------------------------------------------------------------------
#
# File: DMABinding.py
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

from Deeploy.DeeployTypes import CodeGenVerbosity, CodeTransformation, NetworkContext, NodeBinding, NodeTemplate, \
    NodeTypeChecker, _NoVerbosity


class FutureBinding(NodeBinding):

    def __init__(self,
                 typeChecker: NodeTypeChecker,
                 template: NodeTemplate,
                 codeTransformer: CodeTransformation,
                 stateReference: Optional = None):
        super().__init__(typeChecker, template, codeTransformer)
        self.stateReference = stateReference

    def assignStateReferenceElement(self, ctxt: NetworkContext) -> NetworkContext:
        return ctxt

    def codeTransform(self, ctxt: NetworkContext, verbose: CodeGenVerbosity = _NoVerbosity) -> NetworkContext:
        ctxt = self.assignStateReferenceElement(ctxt)
        ctxt = super().codeTransform(ctxt, verbose)

        return ctxt

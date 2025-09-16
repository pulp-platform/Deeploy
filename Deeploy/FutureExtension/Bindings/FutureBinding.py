# SPDX-FileCopyrightText: 2023 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

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

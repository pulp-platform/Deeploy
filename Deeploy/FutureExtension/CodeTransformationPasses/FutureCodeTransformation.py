# ----------------------------------------------------------------------
#
# File: Future.py
#
# Last edited: 12.06.2023
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

from typing import List, Tuple

from Deeploy.CommonExtensions.CodeTransformationPasses.IntrospectiveCodeTransformation import \
    IntrospectiveCodeTransformationMixIn
from Deeploy.DeeployTypes import CodeGenVerbosity, CodeTransformationPass, ExecutionBlock, NetworkContext, _NoVerbosity
from Deeploy.FutureExtension.Future import Future


class FutureGeneration(CodeTransformationPass, IntrospectiveCodeTransformationMixIn):

    def apply(self,
              ctxt: NetworkContext,
              executionBlock: ExecutionBlock,
              name: str,
              verbose: CodeGenVerbosity = _NoVerbosity) -> Tuple[NetworkContext, ExecutionBlock]:
        ctxt, executionBlock = self._dispatchFutures(ctxt, executionBlock, name)
        ctxt, executionBlock = self._resolveFutures(ctxt, executionBlock, name)
        return ctxt, executionBlock

    def _extractFutureArgs(self, ctxt: NetworkContext, executionBlock: ExecutionBlock) -> List[str]:
        futures = []
        dynamicReferences = self.extractDynamicReferences(ctxt, executionBlock, unrollStructs = True)
        references = [ctxt.lookup(key) for key in dynamicReferences]
        futureReferences = [ref for ref in references if issubclass(ref._type, Future)]

        for reference in futureReferences:
            if not hasattr(reference._instance, "stateReference"):
                raise Exception(f"Buffer {reference} is a Future type but has no state element!")
            if reference._deploy:
                futures.append(reference)

        return futures

    def _resolveFutures(self, ctxt: NetworkContext, executionBlock: ExecutionBlock,
                        name: str) -> Tuple[NetworkContext, ExecutionBlock]:

        futures = self._extractFutureArgs(ctxt, executionBlock)
        for reference in futures:
            stateReference = reference._instance.stateReference.name
            # SCHEREMO: Late resolve if we are in the output of the network
            if reference._users == []:
                executionBlock.addRight(reference._type.resolveCheckTemplate, {
                    **reference._bufferRepresentation(),
                    **reference._instance._bufferRepresentation()
                })
                if name not in ctxt.lookup(stateReference)._users:
                    ctxt.lookup(stateReference)._users.append(name)

            # SCHEREMO: Early resolve if we are the first user - otherwise it has been resolved already (in a static scheduler)!
            elif name == reference._users[0]:
                executionBlock.addLeft(reference._type.resolveCheckTemplate, {
                    **reference._bufferRepresentation(),
                    **reference._instance._bufferRepresentation()
                })
                if name not in ctxt.lookup(stateReference)._users:
                    ctxt.lookup(stateReference)._users.append(name)
        return ctxt, executionBlock

    def _dispatchFutures(self, ctxt: NetworkContext, executionBlock: ExecutionBlock,
                         name: str) -> Tuple[NetworkContext, ExecutionBlock]:
        futures = self._extractFutureArgs(ctxt, executionBlock)
        for reference in futures:
            stateReference = reference._instance.stateReference.name
            # Dispatch iff we are not a user, i.e. we are the producer
            if name not in reference._users:
                executionBlock.addLeft(reference._type.dispatchCheckTemplate, {
                    **reference._bufferRepresentation(),
                    **reference._instance._bufferRepresentation()
                })
                if name not in ctxt.lookup(stateReference)._users:
                    ctxt.lookup(stateReference)._users.append(name)
        return ctxt, executionBlock

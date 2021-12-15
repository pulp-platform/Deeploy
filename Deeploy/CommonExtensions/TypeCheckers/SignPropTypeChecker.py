# ----------------------------------------------------------------------
#
# File: SignPropChecker.py
#
# Last edited: 19.05.2023
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

from typing import List, Optional

import onnx_graphsurgeon as gs

from Deeploy.DeeployTypes import ConstantBuffer, NetworkContext, NodeTypeChecker, OperatorRepresentation, VariableBuffer


class SignPropTypeChecker(NodeTypeChecker):

    def _inferNumLevels(self, inputs: List[VariableBuffer],
                        operatorRepresentation: OperatorRepresentation) -> Optional[List[int]]:
        return None

    def _inferSignedness(self, inputs: List[VariableBuffer],
                         operatorRepresentation: OperatorRepresentation) -> Optional[List[int]]:
        return None

    def typeInferGlobalCtxt(self, ctxt: NetworkContext, node: gs.Node) -> NetworkContext:
        ctxt = super().typeInferGlobalCtxt(ctxt, node)

        for inputNode, _type in zip(node.inputs, self.input_types):
            if isinstance(ctxt.lookup(inputNode.name), ConstantBuffer):
                reference = ctxt.lookup(inputNode.name)
                if not _type.referencedType.checkPromotion(reference.values):
                    raise Exception(f"Can't cast {reference} to {_type}!")

                reference.nLevels = reference.values.max() - reference.values.min()
                reference._signed = _type.referencedType.typeMin < 0

        return ctxt

    def typeInferOutput(self, ctxt: NetworkContext, node: gs.Node,
                        operatorRepresentation: OperatorRepresentation) -> NetworkContext:
        ctxt = super().typeInferOutput(ctxt, node, operatorRepresentation)

        inputs = [ctxt.lookup(inputNode.name) for inputNode in node.inputs]
        outputs = [ctxt.lookup(outputNode.name) for outputNode in node.outputs]

        signProp = all([hasattr(_input, "_signed") and hasattr(_input, "nLevels") for _input in inputs])

        if signProp:
            nLevels = self._inferNumLevels(inputs, operatorRepresentation)
            signedness = self._inferSignedness(inputs, operatorRepresentation)

            for obj, nLevels, sign in zip(outputs, nLevels, signedness):
                obj.nLevels = nLevels
                obj._signed = sign

        return ctxt

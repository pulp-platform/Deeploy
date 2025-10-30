# SPDX-FileCopyrightText: 2023 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod
from typing import List

import onnx_graphsurgeon as gs

from Deeploy.AbstractDataTypes import IntegerImmediate
from Deeploy.DeeployTypes import ConstantBuffer, NetworkContext, NodeTypeChecker, OperatorRepresentation, VariableBuffer
from Deeploy.Logging import DEFAULT_LOGGER as log


class SignPropTypeChecker(NodeTypeChecker, ABC):

    @abstractmethod
    def _inferNumLevels(self, inputs: List[VariableBuffer],
                        operatorRepresentation: OperatorRepresentation) -> List[int]:
        pass

    @abstractmethod
    def _inferSignedness(self, inputs: List[VariableBuffer],
                         operatorRepresentation: OperatorRepresentation) -> List[bool]:
        pass

    def typeInferGlobalCtxt(self, ctxt: NetworkContext, node: gs.Node) -> NetworkContext:
        ctxt = super().typeInferGlobalCtxt(ctxt, node)

        for tensor, _type in zip(node.inputs, self.input_types):
            buffer = ctxt.lookup(tensor.name)
            if isinstance(buffer, ConstantBuffer):
                refTy = _type.referencedType
                assert issubclass(refTy, IntegerImmediate)
                if not refTy.checkPromotion(buffer.values):
                    raise ValueError(f"Can't cast {buffer} to {refTy}!")
                buffer.nLevels = buffer.values.max() - buffer.values.min()
                buffer._signed = refTy.typeMin < 0

        return ctxt

    def typeInferOutput(self, ctxt: NetworkContext, node: gs.Node,
                        operatorRepresentation: OperatorRepresentation) -> NetworkContext:
        ctxt = super().typeInferOutput(ctxt, node, operatorRepresentation)

        inputs = [ctxt.lookup(inputNode.name) for inputNode in node.inputs]
        outputs = [ctxt.lookup(outputNode.name) for outputNode in node.outputs]

        nLevels = self._inferNumLevels(inputs, operatorRepresentation)
        signedness = self._inferSignedness(inputs, operatorRepresentation)

        for obj, nLevels, sign in zip(outputs, nLevels, signedness):
            assert isinstance(obj, VariableBuffer)
            obj.nLevels = nLevels
            obj._signed = sign
            refTy = obj._type.referencedType
            if issubclass(refTy, IntegerImmediate) and not refTy.fitsNumLevels(nLevels):
                log.warning(
                    f"{obj.name} has {nLevels} levels, but {refTy.typeName} only supports {refTy.nLevels} levels.")

        return ctxt

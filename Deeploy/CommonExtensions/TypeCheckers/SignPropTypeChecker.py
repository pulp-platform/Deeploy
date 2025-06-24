# SPDX-FileCopyrightText: 2023 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from typing import List, Optional

import onnx_graphsurgeon as gs

from Deeploy.AbstractDataTypes import IntegerImmediate
from Deeploy.DeeployTypes import ConstantBuffer, NetworkContext, NodeTypeChecker, OperatorRepresentation, VariableBuffer
from Deeploy.Logging import DEFAULT_LOGGER as log


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

            if nLevels is None or signedness is None:
                return ctxt
            for obj, nLevel, sign in zip(outputs, nLevels, signedness):
                obj.nLevels = nLevel
                obj._signed = sign
        else:
            if issubclass(obj._type.referencedType, IntegerImmediate) and not obj._type.fitsNumLevels(nLevel):
                log.warning(
                    f"{obj.name} has {nLevel} levels, but {obj._type.referencedType.typeName} only supports {obj._type.referencedType.nLevels} levels."
                )

            if issubclass(obj._type.referencedType, IntegerImmediate) and not obj._type.fitsNumLevels(nLevel):
                log.warning(
                    f"{obj.name} has {nLevel} levels, but {obj._type.referencedType.typeName} only supports {obj._type.referencedType.nLevels} levels."
                )

        return ctxt

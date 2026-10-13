# SPDX-FileCopyrightText: 2021 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from typing import List, Optional, Sequence, Type

import numpy as np

from Deeploy.AbstractDataTypes import FloatImmediate, Pointer
from Deeploy.CommonExtensions.TypeCheckers.SignPropTypeChecker import SignPropTypeChecker
from Deeploy.DeeployTypes import ConstantBuffer, OperatorRepresentation, VariableBuffer


class ConcatChecker(SignPropTypeChecker):

    def __init__(self, input_types: Sequence[Type[Pointer]], output_types: Sequence[Type[Pointer]]):
        super().__init__(input_types, output_types)

    def _inferNumLevels(self, inputs: List[VariableBuffer],
                        operatorRepresentation: OperatorRepresentation) -> Optional[List[int]]:

        maxNLevel = max(i.nLevels for i in inputs)

        return [maxNLevel]

    def _inferSignedness(self, inputs: List[VariableBuffer],
                         operatorRepresentation: OperatorRepresentation) -> Optional[List[bool]]:
        assert (all([_inp._signed == True for _inp in inputs]) or all(
            [[_inp._signed == False for _inp in inputs]])), "Some inputs in concat operation have different signs!"

        if inputs[0]._signed:
            return [True]
        else:
            return [False]


class SliceChecker(SignPropTypeChecker):

    def __init__(self, input_types: Sequence[Type[Pointer]], output_types: Sequence[Type[Pointer]]):
        super().__init__(input_types, output_types)

    def _inferNumLevels(self, inputs: List[VariableBuffer],
                        operatorRepresentation: OperatorRepresentation) -> Optional[List[int]]:
        return [inputs[0].nLevels]

    def _inferSignedness(self, inputs: List[VariableBuffer],
                         operatorRepresentation: OperatorRepresentation) -> Optional[List[bool]]:
        if inputs[0]._signed:
            return [True]
        else:
            return [False]


class TransposeChecker(SignPropTypeChecker):

    def __init__(self, input_types: Sequence[Type[Pointer]], output_types: Sequence[Type[Pointer]]):
        super().__init__(input_types, output_types)

    def _inferNumLevels(self, inputs: List[VariableBuffer],
                        operatorRepresentation: OperatorRepresentation) -> Optional[List[int]]:
        return [inputs[0].nLevels]

    def _inferSignedness(self, inputs: List[VariableBuffer],
                         operatorRepresentation: OperatorRepresentation) -> Optional[List[bool]]:
        if inputs[0]._signed:
            return [True]
        else:
            return [False]


class PadChecker(SignPropTypeChecker):

    def __init__(self, input_types: Sequence[Type[Pointer]], output_types: Sequence[Type[Pointer]]):
        super().__init__(input_types, output_types)

    def _inferNumLevels(self, inputs: List[VariableBuffer],
                        operatorRepresentation: OperatorRepresentation) -> List[int]:
        return [inputs[0].nLevels]

    def _inferSignedness(self, inputs: List[VariableBuffer],
                         operatorRepresentation: OperatorRepresentation) -> List[bool]:
        if inputs[0]._signed:
            return [True]
        else:
            return [False]


class AddChecker(SignPropTypeChecker):

    def __init__(self, input_types: Sequence[Type[Pointer]], output_types: Sequence[Type[Pointer]]):
        super().__init__(input_types, output_types)

    def _inferNumLevels(self, inputs: List[VariableBuffer],
                        operatorRepresentation: OperatorRepresentation) -> List[int]:
        return [inputs[0].nLevels + inputs[1].nLevels]

    def _inferSignedness(self, inputs: List[VariableBuffer],
                         operatorRepresentation: OperatorRepresentation) -> List[bool]:
        if inputs[0]._signed or isinstance(inputs[1], ConstantBuffer):
            return [True]
        else:
            return [False]


class GatherChecker(SignPropTypeChecker):

    def __init__(self, input_types: Sequence[Type[Pointer]], output_types: Sequence[Type[Pointer]]):
        super().__init__(input_types, output_types)

    def _inferNumLevels(self, inputs: List[VariableBuffer],
                        operatorRepresentation: OperatorRepresentation) -> List[int]:
        return [inputs[0].nLevels]

    def _inferSignedness(self, inputs: List[VariableBuffer],
                         operatorRepresentation: OperatorRepresentation) -> List[bool]:
        if inputs[0]._signed:
            return [True]
        else:
            return [False]


class ReshapeChecker(SignPropTypeChecker):

    def __init__(self, input_types: Sequence[Type[Pointer]], output_types: Sequence[Type[Pointer]]):
        super().__init__(input_types, output_types)

    def _inferNumLevels(self, inputs: List[VariableBuffer],
                        operatorRepresentation: OperatorRepresentation) -> List[int]:
        return [inputs[0].nLevels]

    def _inferSignedness(self, inputs: List[VariableBuffer],
                         operatorRepresentation: OperatorRepresentation) -> List[bool]:
        if inputs[0]._signed:
            return [True]
        else:
            return [False]


class MHSAChecker(SignPropTypeChecker):

    def __init__(self, input_types: Sequence[Type[Pointer]], output_types: Sequence[Type[Pointer]]):
        super().__init__(input_types, output_types)

    def _inferNumLevels(self, inputs: List[VariableBuffer],
                        operatorRepresentation: OperatorRepresentation) -> List[int]:
        return [operatorRepresentation['n_levels']]

    def _inferSignedness(self, inputs: List[VariableBuffer],
                         operatorRepresentation: OperatorRepresentation) -> List[bool]:
        return [True]


class CLCAChecker(SignPropTypeChecker):

    def __init__(self, input_types: Sequence[Type[Pointer]], output_types: Sequence[Type[Pointer]]):
        super().__init__(input_types, output_types)

    def _inferNumLevels(self, inputs: List[VariableBuffer],
                        operatorRepresentation: OperatorRepresentation) -> List[int]:
        return [2**(self.input_types[0].referencedType.typeWidth)]

    def _inferSignedness(self, inputs: List[VariableBuffer],
                         operatorRepresentation: OperatorRepresentation) -> List[bool]:
        return [True]


class AndChecker(SignPropTypeChecker):

    def __init__(self, input_types, output_types):
        super().__init__(input_types, output_types)

    def _inferNumLevels(self, inputs, operatorRepresentation):
        return [2]

    def _inferSignedness(self, inputs, operatorRepresentation):
        return [True]


class NotChecker(SignPropTypeChecker):

    def __init__(self, input_types: Sequence[Type[Pointer]], output_types: Sequence[Type[Pointer]]):
        super().__init__(input_types, output_types)

    def _inferNumLevels(self, inputs: List[VariableBuffer],
                        operatorRepresentation: OperatorRepresentation) -> List[int]:
        return [2]

    def _inferSignedness(self, inputs: List[VariableBuffer],
                         operatorRepresentation: OperatorRepresentation) -> List[bool]:
        return [True]


## IsInf Type Checking
class IsInfChecker(SignPropTypeChecker):

    def __init__(self, input_types: Sequence[Type[Pointer]], output_types: Sequence[Type[Pointer]]):
        super().__init__(input_types, output_types)

    def _inferNumLevels(self, inputs: List[VariableBuffer],
                        operatorRepresentation: OperatorRepresentation) -> List[int]:
        return [2]

    def _inferSignedness(self, inputs: List[VariableBuffer],
                         operatorRepresentation: OperatorRepresentation) -> List[bool]:
        return [True]


class IsNanChecker(SignPropTypeChecker):

    def __init__(self, input_types: Sequence[Type[Pointer]], output_types: Sequence[Type[Pointer]]):
        super().__init__(input_types, output_types)

    def _inferNumLevels(self, inputs: List[VariableBuffer],
                        operatorRepresentation: OperatorRepresentation) -> List[int]:
        return [2]

    def _inferSignedness(self, inputs: List[VariableBuffer],
                         operatorRepresentation: OperatorRepresentation) -> List[bool]:
        return [True]


## Doing the same thing add does (types are the same!)
## May need modifications in the future
class GreaterChecker(AddChecker):

    def __init__(self, input_types, output_types):
        super().__init__(input_types, output_types)


class EqualChecker(AddChecker):

    def __init__(self, input_types, output_types):
        super().__init__(input_types, output_types)


class ScatterNDChecker(SignPropTypeChecker):

    def __init__(self, input_types: Sequence[Type[Pointer]], output_types: Sequence[Type[Pointer]]):
        super().__init__(input_types, output_types)

    def _inferNumLevels(self, inputs: List[VariableBuffer],
                        operatorRepresentation: OperatorRepresentation) -> List[int]:
        return [2**(self.input_types[0].referencedType.typeWidth)]

    def _inferSignedness(self, inputs: List[VariableBuffer],
                         operatorRepresentation: OperatorRepresentation) -> List[bool]:
        if inputs[0]._signed:
            return [True]
        else:
            return [False]


class CosChecker(SignPropTypeChecker):

    def __init__(self, input_types: Sequence[Type[Pointer]], output_types: Sequence[Type[Pointer]]):
        super().__init__(input_types, output_types)

    def _inferNumLevels(self, inputs: List[VariableBuffer],
                        operatorRepresentation: OperatorRepresentation) -> List[int]:
        return [2**(self.input_types[0].referencedType.typeWidth)]

    def _inferSignedness(self, inputs: List[VariableBuffer],
                         operatorRepresentation: OperatorRepresentation) -> List[bool]:
        return [True]


class SinChecker(SignPropTypeChecker):

    def __init__(self, input_types: Sequence[Type[Pointer]], output_types: Sequence[Type[Pointer]]):
        super().__init__(input_types, output_types)

    def _inferNumLevels(self, inputs: List[VariableBuffer],
                        operatorRepresentation: OperatorRepresentation) -> List[int]:
        return [2**(self.input_types[0].referencedType.typeWidth)]

    def _inferSignedness(self, inputs: List[VariableBuffer],
                         operatorRepresentation: OperatorRepresentation) -> List[bool]:
        return [True]


class ExpandChecker(SignPropTypeChecker):

    def __init__(self, input_types: Sequence[Type[Pointer]], output_types: Sequence[Type[Pointer]]):
        super().__init__(input_types, output_types)

    def _inferNumLevels(self, inputs: List[VariableBuffer],
                        operatorRepresentation: OperatorRepresentation) -> List[int]:
        return [2**(self.input_types[0].referencedType.typeWidth)]

    def _inferSignedness(self, inputs: List[VariableBuffer],
                         operatorRepresentation: OperatorRepresentation) -> Optional[List[bool]]:
        if inputs[0]._signed:
            return [True]
        else:
            return [False]


class GatherElementsChecker(SignPropTypeChecker):

    def __init__(self, input_types: Sequence[Type[Pointer]], output_types: Sequence[Type[Pointer]]):
        super().__init__(input_types, output_types)

    def _inferNumLevels(self, inputs: List[VariableBuffer],
                        operatorRepresentation: OperatorRepresentation) -> List[int]:
        return [2**(self.input_types[0].referencedType.typeWidth)]

    def _inferSignedness(self, inputs: List[VariableBuffer],
                         operatorRepresentation: OperatorRepresentation) -> Optional[List[bool]]:
        if inputs[0]._signed:
            return [True]
        else:
            return [False]


class TileChecker(SignPropTypeChecker):

    def __init__(self, input_types: Sequence[Type[Pointer]], output_types: Sequence[Type[Pointer]]):
        super().__init__(input_types, output_types)

    def _inferNumLevels(self, inputs: List[VariableBuffer],
                        operatorRepresentation: OperatorRepresentation) -> List[int]:
        return [2**(self.input_types[0].referencedType.typeWidth)]

    def _inferSignedness(self, inputs: List[VariableBuffer],
                         operatorRepresentation: OperatorRepresentation) -> Optional[List[bool]]:
        if inputs[0]._signed:
            return [True]
        else:
            return [False]


class TopKChecker(SignPropTypeChecker):

    def __init__(self, input_types: Sequence[Type[Pointer]], output_types: Sequence[Type[Pointer]]):
        super().__init__(input_types, output_types)

    def _inferNumLevels(
            self,
            inputs: List[VariableBuffer],  #OMG THIS IS ACTUALLY USEFUL
            operatorRepresentation: OperatorRepresentation) -> List[int]:
        return [2**(self.input_types[0].referencedType.typeWidth), 2**(self.input_types[1].referencedType.typeWidth)]

    #ACTUALLY, only the inputs get automaticaly assigned a nLevels and signed
    #HERE is where the outputs get the samle treatment,
    # meaning EVERY SINGLE output needs this
    def _inferSignedness(self, inputs: List[VariableBuffer],
                         operatorRepresentation: OperatorRepresentation) -> Optional[List[bool]]:
        if inputs[0]._signed:
            return [True, True]  #indices shouldn't have to deal with an offset of 2**63
        else:
            return [False, True]


class LogChecker(SignPropTypeChecker):

    def __init__(self, input_types: Sequence[Type[Pointer]], output_types: Sequence[Type[Pointer]]):
        super().__init__(input_types, output_types)

    def _inferNumLevels(self, inputs: List[VariableBuffer],
                        operatorRepresentation: OperatorRepresentation) -> List[int]:
        return [2**(self.input_types[0].referencedType.typeWidth)]

    def _inferSignedness(self, inputs: List[VariableBuffer],
                         operatorRepresentation: OperatorRepresentation) -> List[bool]:
        return [True]


class ErfChecker(SignPropTypeChecker):

    def __init__(self, input_types: Sequence[Type[Pointer]], output_types: Sequence[Type[Pointer]]):
        super().__init__(input_types, output_types)

    def _inferNumLevels(self, inputs: List[VariableBuffer],
                        operatorRepresentation: OperatorRepresentation) -> List[int]:
        return [2**(self.input_types[0].referencedType.typeWidth)]

    def _inferSignedness(self, inputs: List[VariableBuffer],
                         operatorRepresentation: OperatorRepresentation) -> List[bool]:
        return [True]


# class WhereChecker(NodeTypeChecker):

#     def __init__(self, input_types: Sequence[Type[Pointer]], output_types: Sequence[Type[Pointer]]):
#         super().__init__(input_types, output_types)

#     def _inferNumLevels(self, inputs: List[VariableBuffer],
#                         operatorRepresentation: OperatorRepresentation) -> List[int]:
#         return [2**(self.input_types[1].referencedType.typeWidth)]

#     def _inferSignedness(self, inputs: List[VariableBuffer],
#                          operatorRepresentation: OperatorRepresentation) -> List[bool]:
#         if inputs[1]._signed or inputs[2]._signed:
#             return [True]
#         else :
#             return [False]

#     #Actually implement boolean type instead of this
#     def typeInferGlobalCtxt(self, ctxt: NetworkContext, node: gs.Node) -> NetworkContext:
#         ctxt = super().typeInferGlobalCtxt(ctxt, node)

#         for inputNode, _type in zip(node.inputs, self.input_types):
#             if isinstance(ctxt.lookup(inputNode.name), ConstantBuffer):
#                 reference = ctxt.lookup(inputNode.name)
#                 if not _type.referencedType.checkPromotion(reference.values):
#                     raise Exception(f"Can't cast {reference} to {_type}!")
#                 #print(reference)
#                 try :
#                     reference.nLevels = reference.values.max() - reference.values.min()
#                     reference._signed = _type.referencedType.typeMin < 0
#                 except:#because the values are boolean, True minus False is undefined
#                     reference.nLevels=2
#                     reference._signed=True
#         return ctxt

#     #copied from SignProp
#     def typeInferOutput(self, ctxt: NetworkContext, node: gs.Node,
#                         operatorRepresentation: OperatorRepresentation) -> NetworkContext:
#         ctxt = super().typeInferOutput(ctxt, node, operatorRepresentation)

#         inputs = [ctxt.lookup(inputNode.name) for inputNode in node.inputs]
#         outputs = [ctxt.lookup(outputNode.name) for outputNode in node.outputs]

#         signProp = all([hasattr(_input, "_signed") and hasattr(_input, "nLevels") for _input in inputs])

#         if signProp:
#             nLevels = self._inferNumLevels(inputs, operatorRepresentation)
#             signedness = self._inferSignedness(inputs, operatorRepresentation)

#             for obj, nLevels, sign in zip(outputs, nLevels, signedness):
#                 obj.nLevels = nLevels
#                 obj._signed = sign

#         return ctxt


class WhereChecker(SignPropTypeChecker):

    def __init__(self, input_types: Sequence[Type[Pointer]], output_types: Sequence[Type[Pointer]]):
        super().__init__(input_types, output_types)

    def _inferNumLevels(self, inputs: List[VariableBuffer],
                        operatorRepresentation: OperatorRepresentation) -> List[int]:
        return [2**(self.input_types[0].referencedType.typeWidth)]

    def _inferSignedness(self, inputs: List[VariableBuffer],
                         operatorRepresentation: OperatorRepresentation) -> List[bool]:
        return [True]


class ModChecker(SignPropTypeChecker):

    def __init__(self, input_types: Sequence[Type[Pointer]], output_types: Sequence[Type[Pointer]]):
        super().__init__(input_types, output_types)

    def _inferNumLevels(self, inputs: List[VariableBuffer],
                        operatorRepresentation: OperatorRepresentation) -> List[int]:
        return [2**(self.input_types[0].referencedType.typeWidth)]

    def _inferSignedness(self, inputs: List[VariableBuffer],
                         operatorRepresentation: OperatorRepresentation) -> List[bool]:
        return [True]


class ResizeChecker(SignPropTypeChecker):

    def __init__(self, input_types: Sequence[Type[Pointer]], output_types: Sequence[Type[Pointer]]):
        super().__init__(input_types, output_types)

    def _inferNumLevels(self, inputs: List[VariableBuffer],
                        operatorRepresentation: OperatorRepresentation) -> List[int]:
        return [2**(self.input_types[0].referencedType.typeWidth)]

    def _inferSignedness(self, inputs: List[VariableBuffer],
                         operatorRepresentation: OperatorRepresentation) -> List[bool]:
        if inputs[0]._signed:
            return [True]
        else:
            return [False]


class GridSampleChecker(SignPropTypeChecker):

    def __init__(self, input_types: Sequence[Type[Pointer]], output_types: Sequence[Type[Pointer]]):
        super().__init__(input_types, output_types)

    def _inferNumLevels(self, inputs: List[VariableBuffer],
                        operatorRepresentation: OperatorRepresentation) -> List[int]:
        return [2**(self.input_types[0].referencedType.typeWidth)]

    def _inferSignedness(self, inputs: List[VariableBuffer],
                         operatorRepresentation: OperatorRepresentation) -> List[bool]:
        if inputs[0]._signed:
            return [True]
        else:
            return [False]


class CastChecker(SignPropTypeChecker):

    def __init__(self, input_types: Sequence[Type[Pointer]], output_types: Sequence[Type[Pointer]]):
        super().__init__(input_types, output_types)

    def _inferNumLevels(self, inputs: List[VariableBuffer],
                        operatorRepresentation: OperatorRepresentation) -> List[int]:
        return [2**(self.input_types[0].referencedType.typeWidth)]  #[inputs[0].nLevels] id less stable?

    def _inferSignedness(self, inputs: List[VariableBuffer],
                         operatorRepresentation: OperatorRepresentation) -> List[bool]:
        #print("Checker says")
        #print(self.output_types[0].typeName)
        #print(self.output_types[0].referencedType.typeName)

        #We will trust the OUTPUT
        if self.output_types[0].referencedType.typeName == "float32_t":  #add other float datatypes
            return [True]
        if inputs[0]._signed:  #NOT correct if we do unsigned int to signed float,
            return [True]
        else:
            return [False]


class SplitChecker(SignPropTypeChecker):

    def __init__(self, input_types: Sequence[Type[Pointer]], output_types: Sequence[Type[Pointer]]):
        super().__init__(input_types, output_types)

    def _inferNumLevels(self, inputs: List[VariableBuffer],
                        operatorRepresentation: OperatorRepresentation) -> List[int]:
        return [
            2**(self.input_types[0].referencedType.typeWidth), 2**(self.input_types[0].referencedType.typeWidth),
            2**(self.input_types[0].referencedType.typeWidth)
        ]

    def _inferSignedness(self, inputs: List[VariableBuffer],
                         operatorRepresentation: OperatorRepresentation) -> List[bool]:
        if inputs[0]._signed:
            return [True, True, True]
        else:
            return [False, False, False]


class IdentityCastChecker(SignPropTypeChecker):

    def __init__(self, input_types: Sequence[Type[Pointer]], output_types: Sequence[Type[Pointer]]):
        super().__init__(input_types, output_types)

    def _inferNumLevels(self, inputs: List[VariableBuffer],
                        operatorRepresentation: OperatorRepresentation) -> List[int]:
        return [inputs[0].nLevels]

    def _inferSignedness(self, inputs: List[VariableBuffer],
                         operatorRepresentation: OperatorRepresentation) -> List[bool]:
        if inputs[0]._signed:
            return [True]
        else:
            return [False]


## TODO: Modify. It may not be correct


class LinearAttentionChecker(SignPropTypeChecker):

    def __init__(self, input_types: Sequence[Type[Pointer]], output_types: Sequence[Type[Pointer]]):
        super().__init__(input_types, output_types)

    def _inferNumLevels(self, inputs: List[VariableBuffer],
                        operatorRepresentation: OperatorRepresentation) -> List[int]:
        return [2**(self.input_types[0].referencedType.typeWidth)]

    def _inferSignedness(self, inputs: List[VariableBuffer],
                         operatorRepresentation: OperatorRepresentation) -> List[bool]:
        return [True]


class GEMMChecker(SignPropTypeChecker):

    def __init__(self, input_types: Sequence[Type[Pointer]], output_types: Sequence[Type[Pointer]]):
        super().__init__(input_types, output_types)

    def _inferNumLevels(self, inputs: List[VariableBuffer],
                        operatorRepresentation: OperatorRepresentation) -> List[int]:
        return [
            2**((self.input_types[0].referencedType.typeWidth) * 2) *
            inputs[0].shape[-1 - operatorRepresentation['transA']]
        ]

    def _inferSignedness(self, inputs: List[VariableBuffer],
                         operatorRepresentation: OperatorRepresentation) -> List[bool]:
        return [True]


class LayerNormChecker(SignPropTypeChecker):

    def __init__(self, input_types: Sequence[Type[Pointer]], output_types: Sequence[Type[Pointer]]):
        super().__init__(input_types, output_types)

    def _inferNumLevels(self, inputs: List[VariableBuffer],
                        operatorRepresentation: OperatorRepresentation) -> List[int]:
        return [2**(self.input_types[0].referencedType.typeWidth)]

    def _inferSignedness(self, inputs: List[VariableBuffer],
                         operatorRepresentation: OperatorRepresentation) -> List[bool]:
        return [True]


class MulChecker(SignPropTypeChecker):

    def __init__(self, input_types: Sequence[Type[Pointer]], output_types: Sequence[Type[Pointer]]):
        super().__init__(input_types, output_types)

    def _inferNumLevels(self, inputs: List[VariableBuffer],
                        operatorRepresentation: OperatorRepresentation) -> List[int]:
        return [2**(self.input_types[1].typeWidth)]

    def _inferSignedness(self, inputs: List[VariableBuffer],
                         operatorRepresentation: OperatorRepresentation) -> List[bool]:
        if inputs[0]._signed or isinstance(inputs[1], ConstantBuffer):
            return [True]
        else:
            return [False]


class DivChecker(SignPropTypeChecker):

    def __init__(self, input_types: Sequence[Type[Pointer]], output_types: Sequence[Type[Pointer]]):
        super().__init__(input_types, output_types)

    def _inferNumLevels(self, inputs: List[VariableBuffer],
                        operatorRepresentation: OperatorRepresentation) -> List[int]:
        return [2**(self.output_types[0].referencedType.typeWidth)]

    def _inferSignedness(self, inputs: List[VariableBuffer],
                         operatorRepresentation: OperatorRepresentation) -> List[bool]:
        if inputs[0]._signed or isinstance(inputs[1], ConstantBuffer):
            return [True]
        else:
            return [False]


class RQIntegerDivChecker(SignPropTypeChecker):

    def __init__(self, input_types: Sequence[Type[Pointer]], output_types: Sequence[Type[Pointer]]):
        super().__init__(input_types, output_types)

    def _inferNumLevels(self, inputs: List[VariableBuffer],
                        operatorRepresentation: OperatorRepresentation) -> List[int]:
        return [2**(self.output_types[0].referencedType.typeWidth)]

    def _inferSignedness(self, inputs: List[VariableBuffer],
                         operatorRepresentation: OperatorRepresentation) -> List[bool]:
        if inputs[0]._signed or isinstance(inputs[1], ConstantBuffer):
            return [True]
        else:
            return [False]


class MatMulChecker(SignPropTypeChecker):

    def __init__(self, input_types: Sequence[Type[Pointer]], output_types: Sequence[Type[Pointer]]):
        super().__init__(input_types, output_types)

    def _inferNumLevels(self, inputs: List[VariableBuffer],
                        operatorRepresentation: OperatorRepresentation) -> List[int]:
        return [np.max(inputs[0].shape) * np.max(inputs[1].shape) * 2**(self.input_types[0].referencedType.typeWidth)]

    def _inferSignedness(self, inputs: List[VariableBuffer],
                         operatorRepresentation: OperatorRepresentation) -> List[bool]:
        # WIESEP: Hack because previous kernel implementation assumed signed to always be true.
        return [True]


class RQMatMulChecker(SignPropTypeChecker):

    def __init__(self, input_types: Sequence[Type[Pointer]], output_types: Sequence[Type[Pointer]]):
        super().__init__(input_types, output_types)

    def _inferNumLevels(self, inputs: List[VariableBuffer],
                        operatorRepresentation: OperatorRepresentation) -> List[int]:
        return [operatorRepresentation['n_levels']]

    def _inferSignedness(self, inputs: List[VariableBuffer],
                         operatorRepresentation: OperatorRepresentation) -> List[bool]:
        return [bool(operatorRepresentation["signed"])]


class RQGEMMChecker(SignPropTypeChecker):

    def __init__(self, input_types: Sequence[Type[Pointer]], output_types: Sequence[Type[Pointer]]):
        super().__init__(input_types, output_types)

    def _inferNumLevels(self, inputs: List[VariableBuffer],
                        operatorRepresentation: OperatorRepresentation) -> List[int]:
        return [operatorRepresentation['n_levels']]

    def _inferSignedness(self, inputs: List[VariableBuffer],
                         operatorRepresentation: OperatorRepresentation) -> List[bool]:
        return [bool(operatorRepresentation["signed"])]


class ReduceMeanChecker(SignPropTypeChecker):

    def __init__(self, input_types: Sequence[Type[Pointer]], output_types: Sequence[Type[Pointer]]):
        super().__init__(input_types, output_types)

    def _inferNumLevels(self, inputs: List[VariableBuffer],
                        operatorRepresentation: OperatorRepresentation) -> List[int]:
        return [2**(self.input_types[0].referencedType.typeWidth)]

    def _inferSignedness(self, inputs: List[VariableBuffer],
                         operatorRepresentation: OperatorRepresentation) -> List[bool]:
        if inputs[0]._signed:
            return [True]
        else:
            return [False]


class ReduceSumChecker(SignPropTypeChecker):

    def __init__(self, input_types: Sequence[Type[Pointer]], output_types: Sequence[Type[Pointer]]):
        super().__init__(input_types, output_types)

    def _inferNumLevels(self, inputs: List[VariableBuffer],
                        operatorRepresentation: OperatorRepresentation) -> List[int]:
        return [operatorRepresentation['axisLength'] * 2**(self.input_types[0].referencedType.typeWidth)]

    def _inferSignedness(self, inputs: List[VariableBuffer],
                         operatorRepresentation: OperatorRepresentation) -> List[bool]:
        if inputs[0]._signed:
            return [True]
        else:
            return [False]


class ReluChecker(SignPropTypeChecker):

    def __init__(self, input_types: Sequence[Type[Pointer]], output_types: Sequence[Type[Pointer]]):
        super().__init__(input_types, output_types)

    def _inferNumLevels(self, inputs, operatorRepresentation):
        return [2**(self.input_types[0].referencedType.typeWidth)]

    def _inferSignedness(self, inputs: List[VariableBuffer],
                         operatorRepresentation: OperatorRepresentation) -> List[bool]:
        return [False]


class SoftmaxChecker(SignPropTypeChecker):

    def __init__(self, input_types: Sequence[Type[Pointer]], output_types: Sequence[Type[Pointer]]):
        super().__init__(input_types, output_types)

    def _inferNumLevels(self, inputs: List[VariableBuffer],
                        operatorRepresentation: OperatorRepresentation) -> List[int]:
        return [2**(self.input_types[0].referencedType.typeWidth)]

    def _inferSignedness(self, inputs: List[VariableBuffer],
                         operatorRepresentation: OperatorRepresentation) -> List[bool]:
        return [False]


class iNoNormChecker(SignPropTypeChecker):

    def __init__(self, input_types: Sequence[Type[Pointer]], output_types: Sequence[Type[Pointer]]):
        super().__init__(input_types, output_types)

    def _inferNumLevels(self, inputs: List[VariableBuffer],
                        operatorRepresentation: OperatorRepresentation) -> List[int]:
        return [2**(4 * self.input_types[0].referencedType.typeWidth)]

    def _inferSignedness(self, inputs: List[VariableBuffer],
                         operatorRepresentation: OperatorRepresentation) -> List[bool]:
        if inputs[0]._signed:
            return [True]
        else:
            return [False]


class GELUChecker(SignPropTypeChecker):

    def __init__(self, input_types: Sequence[Type[Pointer]], output_types: Sequence[Type[Pointer]]):
        super().__init__(input_types, output_types)

    def _inferNumLevels(self, inputs: List[VariableBuffer],
                        operatorRepresentation: OperatorRepresentation) -> List[int]:
        return [2**(self.input_types[0].referencedType.typeWidth)]

    def _inferSignedness(self, inputs: List[VariableBuffer],
                         operatorRepresentation: OperatorRepresentation) -> List[bool]:
        if inputs[0]._signed:
            return [True]
        else:
            return [False]


class HardswishChecker(SignPropTypeChecker):

    def __init__(self, input_types: Sequence[Type[Pointer]], output_types: Sequence[Type[Pointer]]):
        super().__init__(input_types, output_types)

    def _inferNumLevels(self, inputs: List[VariableBuffer],
                        operatorRepresentation: OperatorRepresentation) -> List[int]:
        input_type = self.input_types[0].referencedType
        if issubclass(input_type, FloatImmediate):
            return [2**(input_type.typeWidth)]
        return [2**(4 * input_type.typeWidth)]

    def _inferSignedness(self, inputs: List[VariableBuffer],
                         operatorRepresentation: OperatorRepresentation) -> List[bool]:
        if inputs[0]._signed:
            return [True]
        else:
            return [False]


class RQHardswishChecker(SignPropTypeChecker):

    def __init__(self, input_types: Sequence[Type[Pointer]], output_types: Sequence[Type[Pointer]]):
        super().__init__(input_types, output_types)

    def _inferNumLevels(self, inputs: List[VariableBuffer],
                        operatorRepresentation: OperatorRepresentation) -> List[int]:
        return [2**(self.input_types[0].referencedType.typeWidth)]

    def _inferSignedness(self, inputs: List[VariableBuffer],
                         operatorRepresentation: OperatorRepresentation) -> List[bool]:
        if inputs[0]._signed:
            return [True]
        else:
            return [False]


class MaxPoolChecker(SignPropTypeChecker):

    def __init__(self, input_types: Sequence[Type[Pointer]], output_types: Sequence[Type[Pointer]]):
        super().__init__(input_types, output_types)

    def _inferNumLevels(self, inputs: List[VariableBuffer],
                        operatorRepresentation: OperatorRepresentation) -> List[int]:
        return [inputs[0].nLevels]

    def _inferSignedness(self, inputs: List[VariableBuffer],
                         operatorRepresentation: OperatorRepresentation) -> List[bool]:
        if inputs[0]._signed:
            return [True]
        else:
            return [False]


class ConvChecker(SignPropTypeChecker):

    def __init__(self, input_types: Sequence[Type[Pointer]], output_types: Sequence[Type[Pointer]]):
        super().__init__(input_types, output_types)

    def _inferNumLevels(self, inputs: List[VariableBuffer],
                        operatorRepresentation: OperatorRepresentation) -> List[int]:
        weight = inputs[1]
        return [
            np.prod(operatorRepresentation['kernel_shape']) * weight.nLevels * weight.shape[1] *
            2**(self.input_types[0].referencedType.typeWidth)
        ]

    def _inferSignedness(self, inputs: List[VariableBuffer],
                         operatorRepresentation: OperatorRepresentation) -> List[bool]:
        if inputs[0]._signed:
            return [True]
        else:
            return [False]


class RequantShiftChecker(SignPropTypeChecker):

    def __init__(self, input_types: Sequence[Type[Pointer]], output_types: Sequence[Type[Pointer]]):
        super().__init__(input_types, output_types)

    def _inferNumLevels(self, inputs: List[VariableBuffer],
                        operatorRepresentation: OperatorRepresentation) -> List[int]:
        return [operatorRepresentation['n_levels']]

    def _inferSignedness(self, inputs: List[VariableBuffer],
                         operatorRepresentation: OperatorRepresentation) -> List[bool]:
        return [operatorRepresentation["signed"]]


class DummyChecker(SignPropTypeChecker):

    def __init__(self, input_types: Sequence[Type[Pointer]], output_types: Sequence[Type[Pointer]]):
        super().__init__(input_types, output_types)

    def _inferNumLevels(self, inputs: List[VariableBuffer],
                        operatorRepresentation: OperatorRepresentation) -> List[int]:
        return [2**(self.input_types[0].referencedType.typeWidth)]


class DebugPrintChecker(SignPropTypeChecker):

    def __init__(self, input_types: Sequence[Type[Pointer]], output_types: Sequence[Type[Pointer]]):
        super().__init__(input_types, output_types)

    def _inferNumLevels(self, inputs: List[VariableBuffer],
                        operatorRepresentation: OperatorRepresentation) -> List[int]:
        return [inputs[0].nLevels]

    def _inferSignedness(self, inputs: List[VariableBuffer],
                         operatorRepresentation: OperatorRepresentation) -> List[bool]:
        if inputs[0]._signed:
            return [True]
        else:
            return [False]


class RQAddChecker(SignPropTypeChecker):

    def __init__(self, input_types: Sequence[Type[Pointer]], output_types: Sequence[Type[Pointer]]):
        super().__init__(input_types, output_types)

    def _inferNumLevels(self, inputs: List[VariableBuffer],
                        operatorRepresentation: OperatorRepresentation) -> List[int]:
        return [operatorRepresentation['rqsOut_n_levels']]

    def _inferSignedness(self, inputs: List[VariableBuffer],
                         operatorRepresentation: OperatorRepresentation) -> List[bool]:
        return [bool(operatorRepresentation["rqsOut_signed"])]

    # Override this. This should compute the signednes of each output node of the Layer
    def checkOutputType(self, inputs: List[VariableBuffer], operatorRepresentation: OperatorRepresentation) -> bool:
        outputTypeSigned = self.output_types[0].referencedType.typeMin < 0
        if operatorRepresentation['rqsOut_signed'] and outputTypeSigned:
            return True
        if (not operatorRepresentation['rqsOut_signed']) and (not outputTypeSigned):
            return True
        return False


class QuantChecker(SignPropTypeChecker):

    def __init__(self, input_types: Sequence[Type[Pointer]], output_types: Sequence[Type[Pointer]]):
        super().__init__(input_types, output_types)

    def _inferNumLevels(self, inputs: List[VariableBuffer],
                        operatorRepresentation: OperatorRepresentation) -> List[int]:
        # Calculate number of levels based on bit_width
        bit_width = operatorRepresentation['bit_width']
        return [2**bit_width]

    def _inferSignedness(self, inputs: List[VariableBuffer],
                         operatorRepresentation: OperatorRepresentation) -> List[bool]:
        # Return signedness from the operation attributes
        return [bool(operatorRepresentation['signed'])]


class DequantChecker(SignPropTypeChecker):

    def __init__(self, input_types: Sequence[Type[Pointer]], output_types: Sequence[Type[Pointer]]):
        super().__init__(input_types, output_types)

    def _inferNumLevels(self, inputs: List[VariableBuffer],
                        operatorRepresentation: OperatorRepresentation) -> List[int]:
        return [2**(self.output_types[0].referencedType.typeWidth)]

    def _inferSignedness(self, inputs: List[VariableBuffer],
                         operatorRepresentation: OperatorRepresentation) -> List[bool]:
        return [True]


class SoftmaxCrossEntropyLossChecker(SignPropTypeChecker):

    def __init__(self, input_types: Sequence[Type[Pointer]], output_types: Sequence[Type[Pointer]]):
        super().__init__(input_types, output_types)

    def _inferNumLevels(self, inputs: List[VariableBuffer],
                        operatorRepresentation: OperatorRepresentation) -> Optional[List[int]]:

        return [2**(self.input_types[0].referencedType.typeWidth)]

    def _inferSignedness(self, inputs: List[VariableBuffer],
                         operatorRepresentation: OperatorRepresentation) -> Optional[List[bool]]:
        return [False]


class SGDChecker(SignPropTypeChecker):

    def __init__(self, input_types: Sequence[Type[Pointer]], output_types: Sequence[Type[Pointer]]):
        super().__init__(input_types, output_types)

    def _inferNumLevels(self, inputs: List[VariableBuffer],
                        operatorRepresentation: OperatorRepresentation) -> Optional[List[int]]:
        return [2**(self.input_types[0].referencedType.typeWidth)]

    def _inferSignedness(self, inputs: List[VariableBuffer],
                         operatorRepresentation: OperatorRepresentation) -> Optional[List[bool]]:
        return [True]


class BatchNormChecker(SignPropTypeChecker):

    def __init__(self, input_types: Sequence[Type[Pointer]], output_types: Sequence[Type[Pointer]]):
        super().__init__(input_types, output_types)

    def _inferNumLevels(self, inputs: List[VariableBuffer],
                        operatorRepresentation: OperatorRepresentation) -> List[int]:
        return [2**(self.input_types[0].referencedType.typeWidth)]

    def _inferSignedness(self, inputs: List[VariableBuffer],
                         operatorRepresentation: OperatorRepresentation) -> List[bool]:
        return [True]


class RMSNormChecker(SignPropTypeChecker):

    def __init__(self, input_types: Sequence[Type[Pointer]], output_types: Sequence[Type[Pointer]]):
        super().__init__(input_types, output_types)

    def _inferNumLevels(self, inputs: List[VariableBuffer],
                        operatorRepresentation: OperatorRepresentation) -> List[int]:
        # RMSNorm: square, mean, sqrt, reciprocal, multiply
        # Output precision similar to input
        return [2**(self.input_types[0].referencedType.typeWidth)]

    def _inferSignedness(self, inputs: List[VariableBuffer],
                         operatorRepresentation: OperatorRepresentation) -> List[bool]:
        # RMSNorm output can be signed (depending on input signedness)
        if inputs[0]._signed:
            return [True]
        else:
            return [False]


class ReduceMaxChecker(SignPropTypeChecker):

    def __init__(self, input_types: Sequence[Type[Pointer]], output_types: Sequence[Type[Pointer]]):
        super().__init__(input_types, output_types)

    def _inferNumLevels(self, inputs: List[VariableBuffer],
                        operatorRepresentation: OperatorRepresentation) -> List[int]:
        return [inputs[0].nLevels]

    def _inferSignedness(self, inputs: List[VariableBuffer],
                         operatorRepresentation: OperatorRepresentation) -> List[bool]:
        return [inputs[0]._signed]


class TanhChecker(SignPropTypeChecker):

    def __init__(self, input_types: Sequence[Type[Pointer]], output_types: Sequence[Type[Pointer]]):
        super().__init__(input_types, output_types)

    def _inferNumLevels(self, inputs: List[VariableBuffer],
                        operatorRepresentation: OperatorRepresentation) -> List[int]:
        return [2**(self.input_types[0].referencedType.typeWidth)]

    def _inferSignedness(self, inputs: List[VariableBuffer],
                         operatorRepresentation: OperatorRepresentation) -> List[bool]:
        return [True]

# SPDX-FileCopyrightText: 2021 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from typing import List, Optional

import numpy as np

from Deeploy.CommonExtensions.TypeCheckers.SignPropTypeChecker import SignPropTypeChecker
from Deeploy.DeeployTypes import ConstantBuffer, OperatorRepresentation, VariableBuffer


class DummyChecker(SignPropTypeChecker):

    def _inferNumLevels(self, inputs: List[VariableBuffer],
                        operatorRepresentation: OperatorRepresentation) -> List[int]:
        return [2**(self.input_types[0].referencedType.typeWidth)]


class PassThroughTypeChecker(SignPropTypeChecker):

    def _inferNumLevels(self, inputs: List[VariableBuffer],
                        operatorRepresentation: OperatorRepresentation) -> Optional[List[int]]:
        return [inputs[0].nLevels]

    def _inferSignedness(self, inputs: List[VariableBuffer],
                         operatorRepresentation: OperatorRepresentation) -> Optional[List[bool]]:
        return [bool(inputs[0]._signed)]


SliceChecker = PassThroughTypeChecker
TransposeChecker = PassThroughTypeChecker
PadChecker = PassThroughTypeChecker
GatherChecker = PassThroughTypeChecker
ReshapeChecker = PassThroughTypeChecker
MaxPoolChecker = PassThroughTypeChecker
DebugPrintChecker = PassThroughTypeChecker


class SignedOutputTypeChecker(SignPropTypeChecker):

    def _inferSignedness(self, inputs: List[VariableBuffer],
                         operatorRepresentation: OperatorRepresentation) -> List[bool]:
        return [True]


class ConcatChecker(SignPropTypeChecker):

    def _inferNumLevels(self, inputs: List[VariableBuffer],
                        operatorRepresentation: OperatorRepresentation) -> Optional[List[int]]:

        maxNLevel = max(i.nLevels for i in inputs)

        return [maxNLevel]

    def _inferSignedness(self, inputs: List[VariableBuffer],
                         operatorRepresentation: OperatorRepresentation) -> Optional[List[bool]]:
        assert (all([inp._signed for inp in inputs])
                or all([[not inp._signed for inp in inputs]])), "Some inputs in concat operation have different signs!"

        if inputs[0]._signed:
            return [True]
        else:
            return [False]


class AddChecker(SignPropTypeChecker):

    def _inferNumLevels(self, inputs: List[VariableBuffer],
                        operatorRepresentation: OperatorRepresentation) -> List[int]:
        return [inputs[0].nLevels + inputs[1].nLevels]

    def _inferSignedness(self, inputs: List[VariableBuffer],
                         operatorRepresentation: OperatorRepresentation) -> List[bool]:
        if inputs[0]._signed or isinstance(inputs[1], ConstantBuffer):
            return [True]
        else:
            return [False]


class MHSAChecker(SignPropTypeChecker):

    def _inferNumLevels(self, inputs: List[VariableBuffer],
                        operatorRepresentation: OperatorRepresentation) -> List[int]:
        return [operatorRepresentation['n_levels']]

    def _inferSignedness(self, inputs: List[VariableBuffer],
                         operatorRepresentation: OperatorRepresentation) -> List[bool]:
        return [True]


class CLCAChecker(DummyChecker):

    def _inferSignedness(self, inputs: List[VariableBuffer],
                         operatorRepresentation: OperatorRepresentation) -> List[bool]:
        return [True]


class LinearAttentionChecker(DummyChecker):

    def _inferSignedness(self, inputs: List[VariableBuffer],
                         operatorRepresentation: OperatorRepresentation) -> List[bool]:
        return [True]


class GEMMChecker(SignPropTypeChecker):

    def _inferNumLevels(self, inputs: List[VariableBuffer],
                        operatorRepresentation: OperatorRepresentation) -> List[int]:
        return [
            2**((self.input_types[0].referencedType.typeWidth) * 2) *
            inputs[0].shape[-1 - operatorRepresentation['transA']]
        ]

    def _inferSignedness(self, inputs: List[VariableBuffer],
                         operatorRepresentation: OperatorRepresentation) -> List[bool]:
        return [True]


class LayerNormChecker(DummyChecker):

    def _inferSignedness(self, inputs: List[VariableBuffer],
                         operatorRepresentation: OperatorRepresentation) -> List[bool]:
        return [True]


class MulChecker(SignPropTypeChecker):

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

    def _inferNumLevels(self, inputs: List[VariableBuffer],
                        operatorRepresentation: OperatorRepresentation) -> List[int]:
        return [np.max(inputs[0].shape) * np.max(inputs[1].shape) * 2**(self.input_types[0].referencedType.typeWidth)]

    def _inferSignedness(self, inputs: List[VariableBuffer],
                         operatorRepresentation: OperatorRepresentation) -> List[bool]:
        # WIESEP: Hack because previous kernel implementation assumed signed to always be true.
        return [True]
        # if inputs[0]._signed or isinstance(inputs[1], ConstantBuffer):
        #   return [True]
        # else:
        # return [False]


class RQMatMulChecker(SignPropTypeChecker):

    def _inferNumLevels(self, inputs: List[VariableBuffer],
                        operatorRepresentation: OperatorRepresentation) -> List[int]:
        return [operatorRepresentation['n_levels']]

    def _inferSignedness(self, inputs: List[VariableBuffer],
                         operatorRepresentation: OperatorRepresentation) -> List[bool]:
        return [bool(operatorRepresentation["signed"])]


class RQGEMMChecker(SignPropTypeChecker):

    def _inferNumLevels(self, inputs: List[VariableBuffer],
                        operatorRepresentation: OperatorRepresentation) -> List[int]:
        return [operatorRepresentation['n_levels']]

    def _inferSignedness(self, inputs: List[VariableBuffer],
                         operatorRepresentation: OperatorRepresentation) -> List[bool]:
        return [bool(operatorRepresentation["signed"])]


class ReduceMeanChecker(DummyChecker):

    def _inferSignedness(self, inputs: List[VariableBuffer],
                         operatorRepresentation: OperatorRepresentation) -> List[bool]:
        if inputs[0]._signed:
            return [True]
        else:
            return [False]


class ReduceSumChecker(SignPropTypeChecker):

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

    def _inferNumLevels(self, inputs, operatorRepresentation):
        return [2**(self.input_types[0].referencedType.typeWidth)]

    def _inferSignedness(self, inputs: List[VariableBuffer],
                         operatorRepresentation: OperatorRepresentation) -> List[bool]:
        return [False]


class SoftmaxChecker(DummyChecker):

    def _inferSignedness(self, inputs: List[VariableBuffer],
                         operatorRepresentation: OperatorRepresentation) -> List[bool]:
        return [False]


class iNoNormChecker(SignPropTypeChecker):

    def _inferNumLevels(self, inputs: List[VariableBuffer],
                        operatorRepresentation: OperatorRepresentation) -> List[int]:
        return [2**(4 * self.input_types[0].referencedType.typeWidth)]

    def _inferSignedness(self, inputs: List[VariableBuffer],
                         operatorRepresentation: OperatorRepresentation) -> List[bool]:
        if inputs[0]._signed:
            return [True]
        else:
            return [False]


class GELUChecker(DummyChecker):

    def _inferSignedness(self, inputs: List[VariableBuffer],
                         operatorRepresentation: OperatorRepresentation) -> List[bool]:
        if inputs[0]._signed:
            return [True]
        else:
            return [False]


class HardswishChecker(SignPropTypeChecker):

    def _inferNumLevels(self, inputs: List[VariableBuffer],
                        operatorRepresentation: OperatorRepresentation) -> List[int]:
        return [2**(4 * self.input_types[0].referencedType.typeWidth)]

    def _inferSignedness(self, inputs: List[VariableBuffer],
                         operatorRepresentation: OperatorRepresentation) -> List[bool]:
        if inputs[0]._signed:
            return [True]
        else:
            return [False]


class RQHardswishChecker(DummyChecker):

    def _inferSignedness(self, inputs: List[VariableBuffer],
                         operatorRepresentation: OperatorRepresentation) -> List[bool]:
        if inputs[0]._signed:
            return [True]
        else:
            return [False]


class ConvChecker(SignPropTypeChecker):

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

    def _inferNumLevels(self, inputs: List[VariableBuffer],
                        operatorRepresentation: OperatorRepresentation) -> List[int]:
        return [operatorRepresentation['n_levels']]

    def _inferSignedness(self, inputs: List[VariableBuffer],
                         operatorRepresentation: OperatorRepresentation) -> List[bool]:
        return [operatorRepresentation["signed"]]


class RQAddChecker(SignPropTypeChecker):

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

    def _inferNumLevels(self, inputs: List[VariableBuffer],
                        operatorRepresentation: OperatorRepresentation) -> List[int]:
        return [2**(self.output_types[0].referencedType.typeWidth)]

    def _inferSignedness(self, inputs: List[VariableBuffer],
                         operatorRepresentation: OperatorRepresentation) -> List[bool]:
        return [True]


class SoftmaxCrossEntropyLossChecker(SignPropTypeChecker):

    def _inferNumLevels(self, inputs: List[VariableBuffer],
                        operatorRepresentation: OperatorRepresentation) -> Optional[List[int]]:

        return [2**(self.input_types[0].referencedType.typeWidth)]

    def _inferSignedness(self, inputs: List[VariableBuffer],
                         operatorRepresentation: OperatorRepresentation) -> Optional[List[bool]]:
        return [False]


class SGDChecker(SignPropTypeChecker):

    def _inferNumLevels(self, inputs: List[VariableBuffer],
                        operatorRepresentation: OperatorRepresentation) -> Optional[List[int]]:
        return [2**(self.input_types[0].referencedType.typeWidth)]

    def _inferSignedness(self, inputs: List[VariableBuffer],
                         operatorRepresentation: OperatorRepresentation) -> Optional[List[bool]]:
        return [True]


class BatchNormChecker(DummyChecker):

    def _inferSignedness(self, inputs: List[VariableBuffer],
                         operatorRepresentation: OperatorRepresentation) -> List[bool]:
        return [True]

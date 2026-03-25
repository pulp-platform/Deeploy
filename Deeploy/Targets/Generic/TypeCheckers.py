# SPDX-FileCopyrightText: 2021 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from typing import List, Optional, Sequence, Type

import numpy as np

from Deeploy.AbstractDataTypes import Pointer
from Deeploy.CommonExtensions.TypeCheckers.SignPropTypeChecker import SignPropTypeChecker
import onnx_graphsurgeon as gs

from Deeploy.DeeployTypes import ConstantBuffer, NetworkContext, OperatorRepresentation, VariableBuffer


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
        return [2**(self.input_types[0].referencedType.typeWidth)] * len(self.output_types)

    def _inferSignedness(self, inputs: List[VariableBuffer],
                         operatorRepresentation: OperatorRepresentation) -> List[bool]:
        return [True] * len(self.output_types)


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
        # if inputs[0]._signed or isinstance(inputs[1], ConstantBuffer):
        #   return [True]
        # else:
        # return [False]


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
        return [2**(4 * self.input_types[0].referencedType.typeWidth)]

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


class MaxPoolGradChecker(SignPropTypeChecker):
    """TypeChecker for MaxPoolGrad: two float inputs (grad_output, original_input), one float output (grad_input)."""

    def __init__(self, input_types: Sequence[Type[Pointer]], output_types: Sequence[Type[Pointer]]):
        super().__init__(input_types, output_types)

    def _inferNumLevels(self, inputs: List[VariableBuffer],
                        operatorRepresentation: OperatorRepresentation) -> List[int]:
        return [inputs[0].nLevels]

    def _inferSignedness(self, inputs: List[VariableBuffer],
                         operatorRepresentation: OperatorRepresentation) -> List[bool]:
        return [True] if inputs[0]._signed else [False]

    def checkOutputType(self, inputs: List[VariableBuffer], operatorRepresentation: OperatorRepresentation) -> bool:
        return True


class AveragePoolChecker(SignPropTypeChecker):

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

    def checkOutputType(self, inputs: List[VariableBuffer],
                        operatorRepresentation: OperatorRepresentation) -> bool:
        # The parser sets 'loss' to a non-empty string for 2-output nodes, '' for 1-output.
        # Use this to determine the actual output count and match it against this binding.
        actual_num_outputs = 2 if operatorRepresentation.get('loss', '') != '' else 1
        return actual_num_outputs == len(self.output_types)

    def _inferNumLevels(self, inputs: List[VariableBuffer],
                        operatorRepresentation: OperatorRepresentation) -> Optional[List[int]]:

        return [2**(self.input_types[0].referencedType.typeWidth)] * len(self.output_types)

    def _inferSignedness(self, inputs: List[VariableBuffer],
                         operatorRepresentation: OperatorRepresentation) -> Optional[List[bool]]:
        return [False] * len(self.output_types)


class MSELossChecker(SignPropTypeChecker):

    def __init__(self, input_types: Sequence[Type[Pointer]], output_types: Sequence[Type[Pointer]]):
        super().__init__(input_types, output_types)

    def _inferNumLevels(self, inputs: List[VariableBuffer],
                        operatorRepresentation: OperatorRepresentation) -> Optional[List[int]]:
        return [2**(self.input_types[0].referencedType.typeWidth)] * len(self.output_types)

    def _inferSignedness(self, inputs: List[VariableBuffer],
                         operatorRepresentation: OperatorRepresentation) -> Optional[List[bool]]:
        return [False] * len(self.output_types)


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


class BatchNormInternalChecker(SignPropTypeChecker):
    """TypeChecker for ORT BatchNormInternal (training-mode BN forward pass).

    Inputs (5):  X, gamma, beta, running_mean, running_var  — all float32
    Outputs (5): Y, updated_running_mean, updated_running_var, saved_mean, saved_inv_std
                 — all float32; outputs[1,2] have no consumers but do have tensor names.

    _inferNumLevels / _inferSignedness return lists of length 5 so that typeInferOutput
    (which zips output_types with node.outputs) correctly types all five outputs.
    """

    def __init__(self, input_types: Sequence[Type[Pointer]], output_types: Sequence[Type[Pointer]]):
        super().__init__(input_types, output_types)

    def _inferNumLevels(self, inputs: List[VariableBuffer],
                        operatorRepresentation: OperatorRepresentation) -> List[int]:
        return [2**(self.input_types[0].referencedType.typeWidth)] * len(self.output_types)

    def _inferSignedness(self, inputs: List[VariableBuffer],
                         operatorRepresentation: OperatorRepresentation) -> List[bool]:
        return [True] * len(self.output_types)


class BatchNormalizationGradChecker(SignPropTypeChecker):
    """TypeChecker for ORT BatchNormalizationGrad (BN backward pass).

    Inputs (5):  dY, X, gamma, saved_mean, saved_inv_std  — all float32
    Outputs (3): dX, dgamma, dbeta                        — all float32
    """

    def __init__(self, input_types: Sequence[Type[Pointer]], output_types: Sequence[Type[Pointer]]):
        super().__init__(input_types, output_types)

    def _inferNumLevels(self, inputs: List[VariableBuffer],
                        operatorRepresentation: OperatorRepresentation) -> List[int]:
        return [2**(self.input_types[0].referencedType.typeWidth)] * len(self.output_types)

    def _inferSignedness(self, inputs: List[VariableBuffer],
                         operatorRepresentation: OperatorRepresentation) -> List[bool]:
        return [True] * len(self.output_types)


class WelfordReduceChecker(SignPropTypeChecker):
    """TypeChecker for WelfordReduce (split BN forward reduction).

    Inputs (1):  X [N,C,H,W]
    Outputs (2): saved_mean [C], saved_inv_std [C]
    """

    def __init__(self, input_types: Sequence[Type[Pointer]], output_types: Sequence[Type[Pointer]]):
        super().__init__(input_types, output_types)

    def _inferNumLevels(self, inputs: List[VariableBuffer],
                        operatorRepresentation: OperatorRepresentation) -> List[int]:
        return [2**(self.input_types[0].referencedType.typeWidth)] * len(self.output_types)

    def _inferSignedness(self, inputs: List[VariableBuffer],
                         operatorRepresentation: OperatorRepresentation) -> List[bool]:
        return [True] * len(self.output_types)


class ChannelNormalizeChecker(SignPropTypeChecker):
    """TypeChecker for ChannelNormalize (split BN forward elementwise).

    Inputs (5):  X, saved_mean, saved_inv_std, gamma, beta
    Outputs (1): Y [N,C,H,W]
    """

    def __init__(self, input_types: Sequence[Type[Pointer]], output_types: Sequence[Type[Pointer]]):
        super().__init__(input_types, output_types)

    def _inferNumLevels(self, inputs: List[VariableBuffer],
                        operatorRepresentation: OperatorRepresentation) -> List[int]:
        return [2**(self.input_types[0].referencedType.typeWidth)]

    def _inferSignedness(self, inputs: List[VariableBuffer],
                         operatorRepresentation: OperatorRepresentation) -> List[bool]:
        return [True]


class BNGradReduceChecker(SignPropTypeChecker):
    """TypeChecker for BNGradReduce (split BN backward reduction).

    Inputs (4):  dY, X, saved_mean, saved_inv_std
    Outputs (2): dgamma [C], dbeta [C]
    """

    def __init__(self, input_types: Sequence[Type[Pointer]], output_types: Sequence[Type[Pointer]]):
        super().__init__(input_types, output_types)

    def _inferNumLevels(self, inputs: List[VariableBuffer],
                        operatorRepresentation: OperatorRepresentation) -> List[int]:
        return [2**(self.input_types[0].referencedType.typeWidth)] * len(self.output_types)

    def _inferSignedness(self, inputs: List[VariableBuffer],
                         operatorRepresentation: OperatorRepresentation) -> List[bool]:
        return [True] * len(self.output_types)


class BNGradNormalizeChecker(SignPropTypeChecker):
    """TypeChecker for BNGradNormalize (split BN backward elementwise).

    Inputs (7):  dY, X, saved_mean, saved_inv_std, gamma, dgamma, dbeta
    Outputs (1): dX [N,C,H,W]
    """

    def __init__(self, input_types: Sequence[Type[Pointer]], output_types: Sequence[Type[Pointer]]):
        super().__init__(input_types, output_types)

    def _inferNumLevels(self, inputs: List[VariableBuffer],
                        operatorRepresentation: OperatorRepresentation) -> List[int]:
        return [2**(self.input_types[0].referencedType.typeWidth)]

    def _inferSignedness(self, inputs: List[VariableBuffer],
                         operatorRepresentation: OperatorRepresentation) -> List[bool]:
        return [True]


class GlobalAveragePoolChecker(SignPropTypeChecker):
    """TypeChecker for GlobalAveragePool.

    Input:  data_in  [N, C, H, W]  float32
    Output: data_out [N, C, 1, 1]  float32
    """

    def __init__(self, input_types: Sequence[Type[Pointer]], output_types: Sequence[Type[Pointer]]):
        super().__init__(input_types, output_types)

    def _inferNumLevels(self, inputs: List[VariableBuffer],
                        operatorRepresentation: OperatorRepresentation) -> List[int]:
        return [inputs[0].nLevels]

    def _inferSignedness(self, inputs: List[VariableBuffer],
                         operatorRepresentation: OperatorRepresentation) -> List[bool]:
        return [inputs[0]._signed]


class GlobalAveragePoolGradChecker(SignPropTypeChecker):
    """TypeChecker for GlobalAveragePoolGrad.

    Input:  dY [N, C, 1, 1]  float32
    Output: dX [N, C, H, W]  float32
    """

    def __init__(self, input_types: Sequence[Type[Pointer]], output_types: Sequence[Type[Pointer]]):
        super().__init__(input_types, output_types)

    def _inferNumLevels(self, inputs: List[VariableBuffer],
                        operatorRepresentation: OperatorRepresentation) -> List[int]:
        return [2**(self.input_types[0].referencedType.typeWidth)]

    def _inferSignedness(self, inputs: List[VariableBuffer],
                         operatorRepresentation: OperatorRepresentation) -> List[bool]:
        return [True]


class InPlaceAccumulatorV2Checker(SignPropTypeChecker):
    """Type checker for ORT InPlaceAccumulatorV2 operator (com.microsoft).

    Inputs:
        0: buffer          (float32*)
        1: gradient        (float32*)
        2: lazy_reset_grad (uint8_t* or bool* - 1 element)

    Output:
        0: output_buffer   (float32*)
    """

    def __init__(self, input_types: Sequence[Type[Pointer]], output_types: Sequence[Type[Pointer]]):
        super().__init__(input_types, output_types)

    def _inferNumLevels(self, inputs: List[VariableBuffer],
                        operatorRepresentation: OperatorRepresentation) -> List[int]:
        # Output has same precision as the buffer input (float32)
        return [2**(self.input_types[0].referencedType.typeWidth)]

    def _inferSignedness(self, inputs: List[VariableBuffer],
                         operatorRepresentation: OperatorRepresentation) -> List[bool]:
        # Float32 output is signed
        return [True]


class PULPConvGradBChecker(SignPropTypeChecker):
    """TypeChecker for ConvGradB which only has one input (output_grad)"""

    def __init__(self, input_types: Sequence[Type[Pointer]], output_types: Sequence[Type[Pointer]]):
        super().__init__(input_types, output_types)

    def _inferNumLevels(self, inputs: List[VariableBuffer],
                        operatorRepresentation: OperatorRepresentation) -> List[int]:
        return [inputs[0].nLevels]

    def _inferSignedness(self, inputs: List[VariableBuffer],
                         operatorRepresentation: OperatorRepresentation) -> List[bool]:
        return [inputs[0]._signed]

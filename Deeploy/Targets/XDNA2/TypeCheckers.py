# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from typing import List, Optional, Sequence, Type

from Deeploy.AbstractDataTypes import Pointer
from Deeploy.CommonExtensions.TypeCheckers.SignPropTypeChecker import SignPropTypeChecker
from Deeploy.DeeployTypes import OperatorRepresentation, VariableBuffer


class XDNA2AddChecker(SignPropTypeChecker):
    """Type checker for BF16 elementwise Add on XDNA2.

    Both inputs and the output are bfloat16_t pointers.
    """

    def __init__(self, input_types: Sequence[Type[Pointer]], output_types: Sequence[Type[Pointer]]):
        super().__init__(input_types, output_types)

    def _inferNumLevels(self, inputs: List[VariableBuffer],
                        operatorRepresentation: OperatorRepresentation) -> Optional[List[int]]:
        # Float types do not have a meaningful nLevels — return 1 as a neutral value.
        return [1]

    def _inferSignedness(self, inputs: List[VariableBuffer],
                         operatorRepresentation: OperatorRepresentation) -> Optional[List[bool]]:
        # BF16 is a signed floating-point type.
        return [True]

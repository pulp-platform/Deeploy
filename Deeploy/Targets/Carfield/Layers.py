# SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from typing import Tuple

from Deeploy.DeeployTypes import OperatorRepresentation, Shape
from Deeploy.Targets.Generic.Layers import SingleOperationPerElementLayer


class CarfieldChannelBroadcastLayer(SingleOperationPerElementLayer):

    def computeShapes(self, inputShapes: Shape, outputShapes: Shape, operatorRepresentation: OperatorRepresentation,
                      channels_first: bool) -> Tuple[Shape, Shape]:
        return (inputShapes, [inputShapes[0]])

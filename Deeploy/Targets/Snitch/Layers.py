# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from typing import List, Tuple

import numpy as np

from Deeploy.DeeployTypes import NodeMapper, ONNXLayer, OperatorRepresentation, Shape


class iNoNormLayer(ONNXLayer):

    def __init__(self, maps: List[NodeMapper]):
        super().__init__(maps)

    def computeOps(self):
        return self.mapper.parser.operatorRepresentation['size'] * 4  # 2 mul, 1 add, 1 right shift

    def computeShapes(self, inputShapes: Shape, outputShapes: Shape, operatorRepresentation: OperatorRepresentation,
                      channels_first: bool) -> Tuple[Shape]:
        # JUNGVI: Broadcast the weights and bias to have as many dimensions as the inputs
        shape = np.broadcast_shapes(*inputShapes)
        return ([shape] * len(inputShapes), outputShapes)

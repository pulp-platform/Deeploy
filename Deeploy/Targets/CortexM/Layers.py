# SPDX-FileCopyrightText: 2021 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from typing import List, Tuple

from Deeploy.DeeployTypes import NodeMapper, Shape
from Deeploy.Targets.Generic.Layers import RQGEMMLayer, RQSConvLayer


class CMSISRQSConvLayer(RQSConvLayer):

    def __init__(self, maps: List[NodeMapper]):
        super().__init__(maps)

    def computeShapes(self, inputShapes: Shape, outputShapes: Shape, operatorRepresentation,
                      channels_first) -> Tuple[Shape, Shape]:
        if channels_first:
            inputShapes[2] = outputShapes[0][1]  # Channels out dimension of Kernel
            inputShapes[3] = outputShapes[0][1]  # Channels out dimension of Kernel
            inputShapes[4] = outputShapes[0][1]  # Channels out dimension of Kernel
        else:
            inputShapes[2] = outputShapes[0][-1]  # Channels out dimension of Kernel
            inputShapes[3] = outputShapes[0][-1]  # Channels out dimension of Kernel
            inputShapes[4] = outputShapes[0][-1]  # Channels out dimension of Kernel
        return (inputShapes, outputShapes)


class CMSISRQSGEMMLayer(RQGEMMLayer):

    def __init__(self, maps: List[NodeMapper]):
        super().__init__(maps)

    def computeShapes(self, inputShapes: Shape, outputShapes: Shape, operatorRepresentation,
                      channels_first) -> Tuple[Shape, Shape]:
        inputShapes[2] = inputShapes[1][-2]  # Channels out dimension of Kernel
        return (inputShapes, outputShapes)

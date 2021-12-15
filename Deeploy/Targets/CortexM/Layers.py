# ----------------------------------------------------------------------
#
# File: CMSISLayers.py
#
# Last edited: 22.12.2021
#
# Copyright (C) 2021, ETH Zurich and University of Bologna.
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

# ----------------------------------------------------------------------
#
# File: BasicLayers.py
#
# Last edited: 17.12.2021
#
# Copyright (C) 2021, ETH Zurich and University of Bologna.
#
# Authors:
# - Moritz Scherer, ETH Zurich
# - Victor Jung, ETH Zurich
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

from typing import Dict, List, Tuple

import numpy as np

from Deeploy.DeeployTypes import NodeMapper, ONNXLayer, Shape, OperatorRepresentation


class ConcatLayer(ONNXLayer):

    def __init__(self, maps: List[NodeMapper]):
        super().__init__(maps)


class iRMSNormLayer(ONNXLayer):

    def __init__(self, maps: List[NodeMapper]):
        super().__init__(maps)


class SliceLayer(ONNXLayer):

    def __init__(self, maps: List[NodeMapper]):
        super().__init__(maps)


class ReshapeLayer(ONNXLayer):

    def __init__(self, maps: List[NodeMapper]):
        super().__init__(maps)


class GatherLayer(ONNXLayer):

    def __init__(self, maps: List[NodeMapper]):
        super().__init__(maps)


class iGELULayer(ONNXLayer):

    def __init__(self, maps: List[NodeMapper]):
        super().__init__(maps)

    def computeOps(self):
        compAbs = self.mapper.parser.operatorRepresentation['size']
        compAdd = self.mapper.parser.operatorRepresentation['size']
        compSqr = self.mapper.parser.operatorRepresentation['size']
        compMul = self.mapper.parser.operatorRepresentation['size']
        compAdd = self.mapper.parser.operatorRepresentation['size']
        compMul2 = self.mapper.parser.operatorRepresentation['size']
        compAdd2 = self.mapper.parser.operatorRepresentation['size']
        compDiv = self.mapper.parser.operatorRepresentation['size']
        return compAbs + compAdd + compSqr + compMul + compAdd + compMul2 + compAdd2 + compDiv


class iHardswishLayer(ONNXLayer):

    def __init__(self, maps: List[NodeMapper]):
        super().__init__(maps)


class iNoNormLayer(ONNXLayer):

    def __init__(self, maps: List[NodeMapper]):
        super().__init__(maps)

    def computeOps(self):
        return self.mapper.parser.operatorRepresentation['size'] * 4  # 2 mul, 1 add, 1 right shift

    def computeShapes(self, inputShapes: Shape, outputShapes: Shape, operatorRepresentation: OperatorRepresentation,
                      channels_first: bool) -> Tuple[Shape]:

        # JUNGVI: Broadcast the weights and bias to have as many dimensions as the inputs
        inputShapes[1] = [1] * (len(inputShapes[0]) - len(inputShapes[1])) + list(inputShapes[1])
        inputShapes[2] = inputShapes[1]
        return (inputShapes, outputShapes)


class RQSiGELULayer(iGELULayer):

    def __init__(self, maps: List[NodeMapper]):
        super().__init__(maps)


class RQSiHardswishLayer(iHardswishLayer):

    def __init__(self, maps: List[NodeMapper]):
        super().__init__(maps)


class iSoftmaxLayer(ONNXLayer):

    def __init__(self, maps: List[NodeMapper]):
        super().__init__(maps)


class ITAMaxLayer(ONNXLayer):

    def __init__(self, maps: List[NodeMapper]):
        super().__init__(maps)


class RequantShiftLayer(ONNXLayer):

    def __init__(self, maps: List[NodeMapper]):
        super().__init__(maps)

    def computeShapes(self, inputShapes: List[Shape], outputShapes: Shape, operatorRepresentation,
                      channels_first) -> Tuple[Shape, Shape]:

        channel_dim = inputShapes[0][1]
        inputShapes[2] = [inputShapes[0][0], channel_dim] + list(inputShapes[2][1:])
        inputShapes[1] = [inputShapes[0][0], channel_dim] + list(inputShapes[1][1:])

        return (inputShapes, outputShapes)

    def computeOps(self):
        return self.mapper.parser.operatorRepresentation['size'] * 3  # One add, one mul, one div


class AddLayer(ONNXLayer):

    def __init__(self, maps: List[NodeMapper]):
        super().__init__(maps)

    def computeShapes(self, inputShapes: Shape, outputShapes: Shape, operatorRepresentation,
                      channels_first) -> Tuple[Shape, Shape]:
        outputShapes = inputShapes.copy()
        if len(inputShapes[0]) > len(inputShapes[1]):
            inputShapes[1] = inputShapes[0]
        else:
            inputShapes[0] = inputShapes[1]

        return (inputShapes, outputShapes)

    def computeOps(self):
        return self.mapper.parser.operatorRepresentation['size']


class MatMulLayer(ONNXLayer):

    def __init__(self, maps: List[NodeMapper]):
        super().__init__(maps)

    def computeOps(self):
        return 2 * self.mapper.parser.operatorRepresentation['M'] * self.mapper.parser.operatorRepresentation[
            'N'] * self.mapper.parser.operatorRepresentation['O'] * self.mapper.parser.operatorRepresentation['batch']


class RQMatMulLayer(MatMulLayer):

    def __init__(self, maps: List[NodeMapper]):
        super().__init__(maps)

    def computeShapes(self, inputShapes: List[Shape], outputShapes: Shape, operatorRepresentation,
                      channels_first) -> Tuple[Shape, Shape]:

        channel_dim = inputShapes[0][1]
        inputShapes[3] = [inputShapes[0][0]] + list(inputShapes[3][1:])
        inputShapes[2] = [inputShapes[0][0]] + list(inputShapes[2][1:])

        return (inputShapes, outputShapes)

    def computeOps(self):
        matmul = super().computeOps()
        rqs = self.mapper.parser.operatorRepresentation['size'] * 3
        return matmul + rqs


class IntegerDivLayer(ONNXLayer):

    def __init__(self, maps: List[NodeMapper]):
        super().__init__(maps)


class RQIntegerDivLayer(IntegerDivLayer):

    def __init__(self, maps: List[NodeMapper]):
        super().__init__(maps)


class GEMMLayer(ONNXLayer):

    def __init__(self, maps: List[NodeMapper]):
        super().__init__(maps)

    def computeShapes(self, inputShapes: Shape, outputShapes: Shape, operatorRepresentation,
                      channels_first) -> Tuple[Shape, Shape]:
        if operatorRepresentation['transA']:
            M = inputShapes[0][-1]
        else:
            M = inputShapes[0][-2]

        if operatorRepresentation['transB']:
            N = inputShapes[1][-2]
        else:
            N = inputShapes[1][-1]

        if len(inputShapes) == 3:
            inputShapes[2] = [M, N]

        return (inputShapes, outputShapes)

    def computeOps(self):
        matmul = 2 * self.mapper.parser.operatorRepresentation['M'] * self.mapper.parser.operatorRepresentation[
            'N'] * self.mapper.parser.operatorRepresentation['O'] * self.mapper.parser.operatorRepresentation['batch']
        gemm = matmul + 3 * self.mapper.parser.operatorRepresentation['M'] * self.mapper.parser.operatorRepresentation[
            'O'] * self.mapper.parser.operatorRepresentation['batch']

        return gemm


class RQGEMMLayer(GEMMLayer):

    def __init__(self, maps: List[NodeMapper]):
        super().__init__(maps)

    def computeShapes(self, inputShapes: List[Shape], outputShapes: Shape, operatorRepresentation,
                      channels_first) -> Tuple[Shape, Shape]:
        if operatorRepresentation['transA']:
            M = inputShapes[0][-1]
        else:
            M = inputShapes[0][-2]

        if operatorRepresentation['transB']:
            N = inputShapes[1][-2]
        else:
            N = inputShapes[1][-1]

        if len(inputShapes) == 5:
            inputShapes[2] = [M, N]
            inputShapes[4] = [inputShapes[0][0]] + list(inputShapes[4][1:])
            inputShapes[3] = [inputShapes[0][0]] + list(inputShapes[3][1:])
        else:
            inputShapes[3] = [inputShapes[0][0]] + list(inputShapes[3][1:])
            inputShapes[2] = [
                inputShapes[0][0],
            ] + list(inputShapes[2][1:])

        return (inputShapes, outputShapes)

    def computeOps(self):
        gemm = super().computeOps()
        rqs = self.mapper.parser.operatorRepresentation['size'] * 3
        return gemm + rqs


class MulLayer(ONNXLayer):

    def __init__(self, maps: List[NodeMapper]):
        super().__init__(maps)

    def computeShapes(self, inputShapes: Shape, outputShapes: Shape, operatorRepresentation,
                      channels_first) -> Tuple[Shape, Shape]:
        if len(inputShapes[0]) > len(inputShapes[1]):
            inputShapes[1] = inputShapes[0]
        else:
            inputShapes[0] = inputShapes[1]
        return (inputShapes, outputShapes)


class ConvLayer(ONNXLayer):

    def __init__(self, maps: List[NodeMapper]):
        super().__init__(maps)

    def computeShapes(self, inputShapes: Shape, outputShapes: Shape, operatorRepresentation,
                      channels_first) -> Tuple[Shape, Shape]:
        if len(inputShapes) == 3:
            inputShapes[2] = inputShapes[1][0]
        return (inputShapes, outputShapes)

    def computeOps(self):
        if "group" in self.mapper.parser.operatorRepresentation:
            groups = self.mapper.parser.operatorRepresentation['group']
        else:
            groups = 1
        opsPerPx = int(
            np.prod(self.mapper.parser.operatorRepresentation['kernel_shape']) *
            self.mapper.parser.operatorRepresentation['ch_im_in'] *
            self.mapper.parser.operatorRepresentation['ch_im_out'] / groups) * 2
        if 'dim_im_out_y' in self.mapper.parser.operatorRepresentation:
            numPx = self.mapper.parser.operatorRepresentation[
                'dim_im_out_x'] * self.mapper.parser.operatorRepresentation['dim_im_out_y']
        else:
            numPx = self.mapper.parser.operatorRepresentation['dim_im_out_x']
        return numPx * opsPerPx


class RQSConvLayer(ConvLayer):

    def __init__(self, maps: List[NodeMapper]):
        super().__init__(maps)

    def computeOps(self):
        conv = super().computeOps()

        if 'dim_im_out_y' in self.mapper.parser.operatorRepresentation:
            rqs = self.mapper.parser.operatorRepresentation['dim_im_out_x'] * self.mapper.parser.operatorRepresentation[
                'dim_im_out_y'] * 3
        else:
            rqs = self.mapper.parser.operatorRepresentation['dim_im_out_x'] * 3

        return conv + rqs


class PadLayer(ONNXLayer):

    def __init__(self, maps: List[NodeMapper]):
        super().__init__(maps)


class MaxPoolLayer(ONNXLayer):

    def __init__(self, maps: List[NodeMapper]):
        super().__init__(maps)


class ReduceMeanLayer(ONNXLayer):

    def __init__(self, maps: List[NodeMapper]):
        super().__init__(maps)


class ReduceSumLayer(ONNXLayer):

    def __init__(self, maps: List[NodeMapper]):
        super().__init__(maps)

    def computeShapes(self, inputShapes: Shape, outputShapes: Shape, operatorRepresentation,
                      channels_first) -> Tuple[Shape, Shape]:
        outputShapes = inputShapes.copy()
        axis = operatorRepresentation['axes'][0]

        if operatorRepresentation['keepdims']:
            outputShapes[0][axis] = 1
        else:
            outputShapes[0] = outputShapes[0][:axis] + outputShapes[0][axis + 1:]
        return (inputShapes, outputShapes)


class iLayerNormLayer(ONNXLayer):

    def __init__(self, maps: List[NodeMapper]):
        super().__init__(maps)

    def computeOps(self):
        compAverage = self.mapper.parser.operatorRepresentation['size']
        compNormalize = self.mapper.parser.operatorRepresentation['size']
        compSqr = self.mapper.parser.operatorRepresentation['size']
        compSum = self.mapper.parser.operatorRepresentation['size']
        compSqrt = self.mapper.parser.operatorRepresentation['size']
        compDiv = self.mapper.parser.operatorRepresentation['size']
        return compAverage + compNormalize + compSqr + compSum + compSqrt + compDiv


class TransposeLayer(ONNXLayer):

    def __init__(self, maps: List[NodeMapper]):
        super().__init__(maps)


class LinearAttentionLayer(ONNXLayer):

    def __init__(self, maps: List[NodeMapper]):
        super().__init__(maps)

    def computeShapes(self, inputShapes: Shape, outputShapes: Shape, operatorRepresentation,
                      channels_first) -> Tuple[Shape, Shape]:
        inputShapes[4] = inputShapes[3][0]
        inputShapes[6] = inputShapes[5][0]
        inputShapes[8] = inputShapes[7][0]
        inputShapes[10] = inputShapes[9][0]

        return (inputShapes, outputShapes)

    def computeOps(self):
        # seqLen = self.mapper.parser.operatorRepresentation['in_C']
        # dim = self.mapper.parser.operatorRepresentation['dim']
        # dim_head = self.mapper.parser.operatorRepresentation['dim_head']
        # heads = self.mapper.parser.operatorRepresentation['heads']
        # QOps = seqLen * dim * dim_head * heads * 2
        # # WQ * Q (H )
        # KOps = seqLen * dim * dim_head * heads * 2
        # # WK * K
        # VOps = seqLen * dim * dim_head * heads * 2
        # # WV * V
        # KVOps = seqLen * dim_head * dim_head * heads * 2
        # # Q * KT
        # QKVOps = seqLen * dim_head * dim_head * heads * 2
        # # N H S S * N H S D -> N H S D
        # OutOps = seqLen * dim_head * heads * dim * 2
        # # WO * O
        # totOps = QOps + KOps + VOps + KVOps + QKVOps + OutOps
        # return totOps

        return 0


class CLCALayer(ONNXLayer):

    def __init__(self, maps: List[NodeMapper]):
        super().__init__(maps)

    def computeShapes(self, inputShapes: Shape, outputShapes: Shape, operatorRepresentation,
                      channels_first) -> Tuple[Shape, Shape]:
        inputShapes[3] = inputShapes[2][0]
        inputShapes[5] = inputShapes[4][0]
        inputShapes[7] = inputShapes[6][0]
        # WQ Requant
        inputShapes[8] = [operatorRepresentation['dim_head'] * operatorRepresentation['heads'], 1]
        inputShapes[9] = [operatorRepresentation['dim_head'] * operatorRepresentation['heads'], 1]
        inputShapes[10] = [operatorRepresentation['dim_head'] * operatorRepresentation['heads'], 1]
        # WK Requant
        inputShapes[11] = [1, 1]
        inputShapes[12] = [1, 1]
        inputShapes[13] = [1, 1]
        # WV Requant
        inputShapes[14] = [operatorRepresentation['dim_head'] * operatorRepresentation['heads'], 1]
        inputShapes[15] = [operatorRepresentation['dim_head'] * operatorRepresentation['heads'], 1]
        inputShapes[16] = [operatorRepresentation['dim_head'] * operatorRepresentation['heads'], 1]
        # Kdiv Requanat
        inputShapes[17] = [1, 1]
        inputShapes[18] = [1, 1]
        inputShapes[19] = [1, 1]
        # Preattn Requant
        inputShapes[20] = [1, 1]
        inputShapes[21] = [1, 1]
        inputShapes[22] = [1, 1]
        # Postattn Requant
        inputShapes[23] = [1, 1]
        inputShapes[24] = [1, 1]
        inputShapes[25] = [1, 1]
        # WO Requant
        inputShapes[26] = [operatorRepresentation['out_dim'], 1]
        inputShapes[27] = [operatorRepresentation['out_dim'], 1]
        inputShapes[28] = [operatorRepresentation['out_dim'], 1]
        return (inputShapes, outputShapes)

    def computeOps(self):

        qLen = self.mapper.parser.operatorRepresentation['q_shape'][-1]
        kLen = self.mapper.parser.operatorRepresentation['kv_shape'][-1]
        inDim = self.mapper.parser.operatorRepresentation['q_shape'][-2]
        heads = self.mapper.parser.operatorRepresentation['heads']
        dim_head = self.mapper.parser.operatorRepresentation['dim_head']
        out_dim = self.mapper.parser.operatorRepresentation['out_dim']

        # q -> Q
        QOps = qLen * 1 * inDim * heads * dim_head * 2
        # v -> V
        VOps = kLen * 1 * inDim * heads * dim_head * 2
        # V -> K
        KOps = kLen * heads * dim_head * 2
        # KOps = 0

        EOps = heads * kLen * heads * dim_head

        MMKTV = heads * dim_head * kLen * dim_head * 2
        MMQA = heads * qLen * dim_head * dim_head * 2
        MMQE = heads * qLen * dim_head * 1 * 2

        # Divs, Adds(eps), muls(delta, eps)
        DivOps = heads * qLen * dim_head + heads * qLen + 2 * heads * qLen * dim_head

        OOps = (heads * dim_head) * qLen * out_dim * 1 * 2

        return QOps + VOps + KOps + EOps + MMKTV + MMQA + MMQE + DivOps + OOps


class MHSALayer(ONNXLayer):

    def __init__(self, maps: List[NodeMapper]):
        super().__init__(maps)

    def computeShapes(self, inputShapes: Shape, outputShapes: Shape, operatorRepresentation,
                      channels_first) -> Tuple[Shape, Shape]:
        outputShapes = [[inputShapes[0][0], operatorRepresentation['heads']] + inputShapes[0][1:]]

        return (inputShapes, outputShapes)

    def computeOps(self):
        seqLen = self.mapper.parser.operatorRepresentation['S']
        dim = self.mapper.parser.operatorRepresentation['dim']
        dim_head = self.mapper.parser.operatorRepresentation['dim_head']
        heads = self.mapper.parser.operatorRepresentation['heads']
        QOps = seqLen * dim * dim_head * heads * 2
        # WQ * Q (H )
        KOps = seqLen * dim * dim_head * heads * 2
        # WK * K
        VOps = seqLen * dim * dim_head * heads * 2
        # WV * V
        QKOps = seqLen * seqLen * dim_head * heads * 2
        # Q * KT
        AVOps = seqLen * seqLen * dim_head * heads * 2
        # N H S S * N H S D -> N H S D
        OutOps = seqLen * dim_head * heads * dim * 2
        # WO * O
        totOps = QOps + KOps + VOps + QKOps + AVOps + OutOps
        return totOps


class DebugPrintLayer(ONNXLayer):

    def __init__(self, maps: List[NodeMapper]):
        super().__init__(maps)

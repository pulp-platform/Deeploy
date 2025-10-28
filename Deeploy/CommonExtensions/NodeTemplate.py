# SPDX-FileCopyrightText: 2021 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from typing import List, Sequence, Tuple

import numpy as np
import onnx_graphsurgeon as gs

from Deeploy.DeeployTypes import NodeTemplate


class ElementwiseTemplate(NodeTemplate):

    def alignShapes(self, node: gs.Node) -> Tuple[List[Sequence[int]], List[Sequence[int]]]:
        assert len(node.outputs) == 1, f"Expected only one output. Received {len(node.outputs)}"
        shape = tuple(np.broadcast_shapes(*[t.shape for t in node.inputs]))
        return [shape] * len(node.inputs), [shape]


class ElementwiseScalarTemplate(NodeTemplate):

    def alignShapes(self, node: gs.Node) -> Tuple[List[Sequence[int]], List[Sequence[int]]]:
        assert len(node.inputs) == 2, f"Expected only two inputs. Received {len(node.inputs)}"
        assert len(node.outputs) == 1, f"Expected only one output. Received {len(node.outputs)}"
        shape = tuple(node.inputs[0].shape)
        return [shape, (1,)], [shape]


class RequantShiftTemplate(NodeTemplate):

    def alignShapes(self, node: gs.Node) -> Tuple[List[Sequence[int]], List[Sequence[int]]]:
        inShapes, outShapes = [t.shape for t in node.inputs], [t.shape for t in node.outputs]
        batch, ch = inShapes[0][:2]
        # TODO: Copied from old computeShape. Should probably be investigated
        inShapes[1] = (batch, ch, *inShapes[1][1:])
        inShapes[2] = (batch, ch, *inShapes[2][1:])
        return inShapes, outShapes


class ConvTemplate(NodeTemplate):

    @staticmethod
    def minPerChannelTensorShape(node: gs.Node, channels: int) -> Tuple[int, ...]:
        spatialDims = len(node.attrs["kernel_shape"])
        if node.attrs["channels_first"]:
            return (channels,) + (1,) * (spatialDims)
        else:
            return (channels,)

    def alignShapes(self, node: gs.Node) -> Tuple[List[Sequence[int]], List[Sequence[int]]]:
        inShapes, outShapes = [t.shape for t in node.inputs], [t.shape for t in node.outputs]
        if len(node.inputs) == 3:
            minBiasShape = self.minPerChannelTensorShape(node, inShapes[1][0])
            inShapes[2] = minBiasShape
        return inShapes, outShapes


class RequantizedConvTemplate(ConvTemplate):

    def alignShapes(self, node: gs.Node) -> Tuple[List[Sequence[int]], List[Sequence[int]]]:
        inShapes, outShapes = [t.shape for t in node.inputs[:2]], [t.shape for t in node.outputs]
        minRqsShape = self.minPerChannelTensorShape(node, inShapes[1][0])
        rqsShapes = [minRqsShape] * len(node.inputs[2:])
        return inShapes + rqsShapes, outShapes


class GemmTemplate(NodeTemplate):

    def alignShapes(self, node: gs.Node) -> Tuple[List[Sequence[int]], List[Sequence[int]]]:
        biasShape = node.outputs[0].shape[-2:]
        return [node.inputs[0].shape, node.inputs[1].shape, biasShape], [node.outputs[0].shape]


class RequantizedGemmTemplate(NodeTemplate):

    def alignShapes(self, node: gs.Node) -> Tuple[List[Sequence[int]], List[Sequence[int]]]:
        inShapes, outShapes = [t.shape for t in node.inputs[:2]], [t.shape for t in node.outputs]
        if node.attrs["transB"]:
            N = inShapes[1][-2]
        else:
            N = inShapes[1][-1]
        rqsShapes = [(N,)] * len(node.inputs[2:])
        return inShapes + rqsShapes, outShapes

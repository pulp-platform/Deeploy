# SPDX-License-Identifier: Apache-2.0

import onnx_graphsurgeon as gs

from Deeploy.CommonExtensions.OptimizationPasses.PassClasses import Pass, contextagnostic


def _decode(value):
    if isinstance(value, bytes):
        return value.decode()
    return value


@contextagnostic
class CarfieldResizeNearestPass(Pass):

    def __init__(self):
        super().__init__()

    def run_pass(self, graph: gs.Graph) -> gs.Graph:
        for node in graph.nodes:
            if node.op != "Resize":
                continue
            if len(node.inputs) != 4 or len(node.outputs) != 1:
                continue
            if _decode(node.attrs.get("mode", "")) != "nearest":
                continue
            if _decode(node.attrs.get("nearest_mode", "")) != "floor":
                continue
            if _decode(node.attrs.get("coordinate_transformation_mode", "")) != "asymmetric":
                continue

            sizes = node.inputs[3]
            if not isinstance(sizes, gs.Constant):
                continue

            inShape = list(node.inputs[0].shape)
            outSizes = [int(v) for v in sizes.values]
            if len(inShape) != 4 or len(outSizes) != 4:
                continue
            if outSizes[0] != inShape[0] or outSizes[1] != inShape[1]:
                continue
            if inShape[2] == 0 or inShape[3] == 0:
                continue
            if outSizes[2] % inShape[2] != 0 or outSizes[3] % inShape[3] != 0:
                continue

            node.attrs["scale_h"] = outSizes[2] // inShape[2]
            node.attrs["scale_w"] = outSizes[3] // inShape[3]
            node.inputs = [node.inputs[0]]

        graph.cleanup().toposort()
        return graph

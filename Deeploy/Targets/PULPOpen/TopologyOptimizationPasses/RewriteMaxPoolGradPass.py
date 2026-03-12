# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import onnx_graphsurgeon as gs

from Deeploy.CommonExtensions.OptimizationPasses.PassClasses import Pass, contextagnostic


@contextagnostic
class RewriteMaxPoolGradInputPass(Pass):
    """Replace MaxPoolGrad's int64 mask input with the float32 forward input from MaxPool.

    ORT training format produces MaxPoolGrad with:
        inputs[0]: dY    (float32, upstream gradient)
        inputs[1]: mask  (int64,   indices from MaxPool output[1])

    Deeploy's MaxPoolGrad kernel recomputes argmax from the original forward input X, so the
    int64 mask is not needed.  This pass rewires input[1] to the float32 X tensor (MaxPool
    input[0]), which:
      1. Makes both MaxPoolGrad inputs float32 → Deeploy typeCheckNodeInputs passes.
      2. Leaves the mask tensor with no consumers → graph.cleanup() removes it, so MaxPool
         ends up with only one typed output and the untyped-buffer issue is also gone.
    """

    def run_pass(self, graph: gs.Graph) -> gs.Graph:
        modified = False

        for node in graph.nodes:
            if node.op != 'MaxPoolGrad':
                continue
            if len(node.inputs) < 2:
                continue

            mask_tensor = node.inputs[1]

            # Identify ORT-format mask by checking whether the tensor is produced by a
            # MaxPool node (i.e. it is MaxPool output[1]).  Using the producer op rather
            # than dtype because ONNX shape inference may leave dtype=None on the mask.
            maxpool_producers = [n for n in mask_tensor.inputs if n.op == 'MaxPool']
            if not maxpool_producers:
                continue  # input[1] is already the float32 forward input — nothing to do

            maxpool_node = maxpool_producers[0]
            x_tensor = maxpool_node.inputs[0]  # float32 forward input X

            # Rewire: replace mask tensor → float32 X
            node.inputs[1] = x_tensor
            modified = True

        if modified:
            graph.cleanup()

        return graph

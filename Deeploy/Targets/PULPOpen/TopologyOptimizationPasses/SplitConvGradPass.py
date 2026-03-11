# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

import onnx_graphsurgeon as gs

from Deeploy.CommonExtensions.OptimizationPasses.PassClasses import Pass, contextagnostic


def _split_single_conv_grad(graph: gs.Graph, node: gs.Node, counter: int):
    """Split one ConvGrad node → ConvGradX + ConvGradW [+ ConvGradB].

    Original ConvGrad:
        inputs : [dY, X, W]           (no bias)
        outputs: [dX, dW]

    or:
        inputs : [dY, X, W, B]        (with bias)
        outputs: [dX, dW, dB]

    After split:
        ConvGradX:  inputs=[dY, W]    → outputs=[dX]
        ConvGradW:  inputs=[dY, X]    → outputs=[dW]
        ConvGradB:  inputs=[dY]       → outputs=[dB]   (only when bias present)
    """
    if len(node.inputs) < 3 or len(node.outputs) < 1:
        return

    dy = node.inputs[0]   # dY: upstream gradient  [N, C_out, H_out, W_out]
    x  = node.inputs[1]   # X:  forward input       [N, C_in,  H_in,  W_in]
    w  = node.inputs[2]   # W:  weight              [C_out, C_in/group, kH, kW]

    dx = node.outputs[0]  # dX: input gradient       [N, C_in,  H_in,  W_in]

    # Copy attrs; add kernel_shape from the weight tensor to avoid
    # Conv2DParser.parseNode computing wrong kernel_shape from inputs[1].
    attrs_x = dict(node.attrs)
    attrs_w = dict(node.attrs)

    if 'kernel_shape' not in attrs_x and w.shape is not None and len(w.shape) >= 4:
        attrs_x['kernel_shape'] = list(w.shape[2:4])

    base_name = node.name if node.name else f'ConvGrad_{counter}'

    # ConvGradX: compute dX from dY and W
    conv_grad_x = gs.Node(
        op      = 'ConvGradX',
        name    = f'{base_name}_ConvGradX',
        inputs  = [dy, w],
        outputs = [dx],
        attrs   = attrs_x,
    )
    graph.nodes.append(conv_grad_x)

    if len(node.outputs) >= 2:
        dw = node.outputs[1]  # dW: weight gradient  [C_out, C_in/group, kH, kW]

        # Propagate shape and dtype from W → dW (same shape; ONNX shape inference misses ConvGrad)
        if dw.shape is None and w.shape is not None:
            dw.shape = list(w.shape)
        if dw.dtype is None and w.dtype is not None:
            dw.dtype = w.dtype

        if 'kernel_shape' not in attrs_w and w.shape is not None and len(w.shape) >= 4:
            attrs_w['kernel_shape'] = list(w.shape[2:4])
        elif 'kernel_shape' not in attrs_w and dw.shape is not None and len(dw.shape) >= 4:
            attrs_w['kernel_shape'] = list(dw.shape[2:4])

        # ConvGradW: compute dW from dY and X
        conv_grad_w = gs.Node(
            op      = 'ConvGradW',
            name    = f'{base_name}_ConvGradW',
            inputs  = [dy, x],
            outputs = [dw],
            attrs   = attrs_w,
        )
        graph.nodes.append(conv_grad_w)

        if len(node.outputs) >= 3:
            db = node.outputs[2]  # dB: bias gradient  [C_out]

            # Propagate bias shape and dtype: dB shape == B shape (or [C_out] from W)
            if db.shape is None:
                if len(node.inputs) >= 4 and node.inputs[3].shape is not None:
                    db.shape = list(node.inputs[3].shape)
                elif w.shape is not None:
                    db.shape = [w.shape[0]]
            if db.dtype is None:
                if len(node.inputs) >= 4 and node.inputs[3].dtype is not None:
                    db.dtype = node.inputs[3].dtype
                elif w.dtype is not None:
                    db.dtype = w.dtype

            # ConvGradB: compute dB = sum(dY, axes=[N, H, W])
            conv_grad_b = gs.Node(
                op      = 'ConvGradB',
                name    = f'{base_name}_ConvGradB',
                inputs  = [dy],
                outputs = [db],
                attrs   = {},
            )
            graph.nodes.append(conv_grad_b)

    # Remove the original ConvGrad node
    node.inputs.clear()
    node.outputs.clear()
    graph.nodes.remove(node)


@contextagnostic
class SplitConvGradPass(Pass):
    """Replace each ConvGrad node with ConvGradX + ConvGradW[B] nodes.

    Handles 1/2/3 outputs:
      1 output (dX only):      ConvGradX
      2 outputs (dX + dW):     ConvGradX + ConvGradW
      3 outputs (dX + dW + dB): ConvGradX + ConvGradW + ConvGradB

    No-op for inference graphs (which have no ConvGrad nodes).
    """

    def run_pass(self, graph: gs.Graph) -> gs.Graph:
        # Collect all ConvGrad nodes first (avoid modifying list while iterating)
        nodes_to_split = [n for n in graph.nodes if n.op == 'ConvGrad']

        for counter, node in enumerate(nodes_to_split):
            _split_single_conv_grad(graph, node, counter)

        if nodes_to_split:
            graph.cleanup()

        return graph

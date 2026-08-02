#!/usr/bin/env python3
# ----------------------------------------------------------------------
# File: generate.py
#
# SoCDAML Part III: TA reference solution.
# Builds a single-node ONNX graph for the integer LeakyReLU operator
# and saves the input/output tensors that DeeployTest's harness will use
# as golden references.
#
# Run from this directory:
#     python generate.py
#
# Resulting artifacts:
#     network.onnx   single-node iLeakyReLU graph
#     inputs.npz     random int8 input tensor named "data_in"
#     outputs.npz    golden int8 output tensor named "data_out"
#
# Quantization-friendly LeakyReLU formula used here:
#     out[i] = x          if x >= 0
#              (mul*x) >> shift   otherwise
# With mul=1, shift=3 this approximates alpha = 0.125.
# ----------------------------------------------------------------------
# SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import onnx
from onnx import TensorProto, helper

SHAPE = (1, 16, 64, 64)  # NCHW; 65 536 elements -> big enough that
# double-buffering's DMA/kernel overlap dominates
# per-tile bookkeeping, so DB visibly beats SB.
MUL = 1
SHIFT = 3
SEED = 0xC0FFEE


def golden(x, mul, shift):
    """Reference int8 LeakyReLU. Arithmetic right shift on negative ints
    matches the C `>>` operator on signed integers on most platforms,
    so we cast to int32, shift, then clip to int8."""
    pos = x.astype(np.int32)
    neg = (mul * pos) >> shift
    out = np.where(pos >= 0, pos, neg)
    return np.clip(out, -128, 127).astype(np.int8)


def build_onnx():
    in_value = helper.make_tensor_value_info('data_in', TensorProto.INT8, SHAPE)
    out_value = helper.make_tensor_value_info('data_out', TensorProto.INT8, SHAPE)

    node = helper.make_node(
        op_type = 'iLeakyReLU',
        inputs = ['data_in'],
        outputs = ['data_out'],
        name = 'iLeakyReLU_0',
        mul = MUL,
        shift = SHIFT,
    )

    graph = helper.make_graph(
        nodes = [node],
        name = 'iLeakyReLU_single_node',
        inputs = [in_value],
        outputs = [out_value],
    )

    model = helper.make_model(graph, producer_name = 'SoCDAML-PartIII')
    model.opset_import[0].version = 13
    model.ir_version = 7
    return model


def main():
    rng = np.random.default_rng(SEED)
    x = rng.integers(low = -128, high = 128, size = SHAPE, dtype = np.int8)
    y = golden(x, MUL, SHIFT)

    model = build_onnx()
    onnx.save(model, 'network.onnx')
    # Deeploy convention: npz tensors saved as int64 (the test harness
    # casts to float64 then to the ONNX dtype). Storing int8 directly
    # confuses the buffer-population path.
    np.savez('inputs.npz', input = x.astype(np.int64))
    np.savez('outputs.npz', output = y.astype(np.int64))

    print(f"Wrote network.onnx (shape={SHAPE}, mul={MUL}, shift={SHIFT})")
    print(f"Wrote inputs.npz  : keys=['input']  shape={x.shape}  int64,  int8-range=[{x.min()}, {x.max()}]")
    print(f"Wrote outputs.npz : keys=['output'] shape={y.shape}  int64,  int8-range=[{y.min()}, {y.max()}]")


if __name__ == '__main__':
    main()

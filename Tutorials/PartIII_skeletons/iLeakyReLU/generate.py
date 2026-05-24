#!/usr/bin/env python3
# ----------------------------------------------------------------------
# File: generate.py  (SoCDAML Part III - Step 1, provided complete)
#
# Builds the single-node ONNX graph + golden tensors that DeeployTest's
# harness will use to validate your iLeakyReLU implementation.
#
# Run from this directory:
#     python generate.py
#
# Outputs:
#     network.onnx, inputs.npz, outputs.npz
#
# Quantization-friendly LeakyReLU formula used here:
#     out[i] = x          if x >= 0
#              (mul*x) >> shift   otherwise
# ----------------------------------------------------------------------
# SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import onnx
from onnx import TensorProto, helper

SHAPE = (1, 16, 64, 64)
MUL = 1
SHIFT = 3
SEED = 0xC0FFEE


def golden(x, mul, shift):
    pos = x.astype(np.int32)
    neg = (mul * pos) >> shift
    out = np.where(pos >= 0, pos, neg)
    return np.clip(out, -128, 127).astype(np.int8)


def build_onnx():
    in_value = helper.make_tensor_value_info('data_in', TensorProto.INT8, SHAPE)
    out_value = helper.make_tensor_value_info('data_out', TensorProto.INT8, SHAPE)
    node = helper.make_node('iLeakyReLU', ['data_in'], ['data_out'], name = 'iLeakyReLU_0', mul = MUL, shift = SHIFT)
    graph = helper.make_graph([node], 'iLeakyReLU_single_node', [in_value], [out_value])
    model = helper.make_model(graph, producer_name = 'SoCDAML-PartIII')
    model.opset_import[0].version = 13
    model.ir_version = 7
    return model


def main():
    rng = np.random.default_rng(SEED)
    x = rng.integers(low = -128, high = 127, size = SHAPE, dtype = np.int8)
    y = golden(x, MUL, SHIFT)
    onnx.save(build_onnx(), 'network.onnx')
    np.savez('inputs.npz', data_in = x)
    np.savez('outputs.npz', data_out = y)
    print(f"OK: network.onnx, inputs.npz, outputs.npz "
          f"(shape={SHAPE}, mul={MUL}, shift={SHIFT})")


if __name__ == '__main__':
    main()

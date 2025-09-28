# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from enum import Enum, IntEnum
from typing import Any, Dict, Tuple, Union

import numpy as np
import onnx_graphsurgeon as gs

from Deeploy.DeeployTypes import AttrDesc, IoDesc, OperatorDescriptor, VariadicIoDesc


def IntUnpack(value: Any) -> int:
    if isinstance(value, (list, tuple)) and len(value) == 1:
        value = value[0]

    if isinstance(value, int):
        return value
    elif isinstance(value, float):
        assert value.is_integer(), f"Received a non-integer value {value}"
        return int(value)
    raise ValueError(f"Unsupported value type {type(value)}")


def BoolUnpack(value: Any) -> bool:
    value = IntUnpack(value)
    assert value in [0, 1], f"Casting to bool only supported from 0, 1. Received {value}"
    return bool(value)


def FloatUnpack(value: Any) -> float:
    if isinstance(value, (list, tuple)) and len(value) == 1:
        value = value[0]

    assert isinstance(value, (int, float)), f"Unsupported value type {type(value)}"
    return float(value)


def IntTupleUnpack(value: Any) -> Tuple[int, ...]:
    try:
        return tuple(IntUnpack(item) for item in value)
    except TypeError:
        return (IntUnpack(value),)


def FloatTupleUnpack(value: Any) -> Tuple[float, ...]:
    try:
        return tuple(FloatUnpack(item) for item in value)
    except TypeError:
        return (FloatUnpack(value),)


def IntTupleIfNotSingleItemUnpack(value: Any) -> Union[int, Tuple[int, ...]]:
    try:
        return IntUnpack(value)
    except:
        return IntTupleUnpack(value)


def attrToTensor(node: gs.Node, attr: str) -> None:
    values = node.attrs[attr]
    if isinstance(values, (int, float)):
        values = np.array([values])
    elif isinstance(values, (list, tuple)):
        values = np.array(values)
    assert isinstance(values, np.ndarray), f"Unsupported values type {type(values)}"
    tensor = gs.Constant(f"{node.name}_{attr}", values)
    node.inputs.append(tensor)
    node.attrs.pop(attr)


concatDesc = OperatorDescriptor(
    inputDescriptor = VariadicIoDesc("data_in", minNumTensors = 2),
    outputDescriptor = IoDesc("data_out"),
    attrDescriptors = [],
)

iRMSNormDesc = OperatorDescriptor(
    inputDescriptor = IoDesc(["data_in", "weight"]),
    outputDescriptor = IoDesc("data_out"),
    attrDescriptors = [
        AttrDesc("D", IntUnpack),
        AttrDesc("n_levels", IntUnpack),
    ],
)


class SliceDescriptor(OperatorDescriptor):

    def canonicalize(self, node: gs.Node, opset: int) -> bool:
        if opset < 10:
            attrToTensor(node, "starts")
            attrToTensor(node, "ends")
            if "axes" in node.attrs:
                attrToTensor(node, "axes")

        return super().canonicalize(node, opset)


# Opset: 13
sliceDesc = SliceDescriptor(
    inputDescriptor = IoDesc(["data_in", "starts", "ends"], ["axes", "steps"]),
    outputDescriptor = IoDesc("data_out"),
    attrDescriptors = [],
)

# Opset: 1
sliceDescOld = OperatorDescriptor(
    inputDescriptor = IoDesc("data_in"),
    outputDescriptor = IoDesc("data_out"),
    attrDescriptors = [
        AttrDesc("axes", IntTupleUnpack, lambda n: range(len(n.attrs["starts"]))),
        AttrDesc("ends", IntTupleUnpack),
        AttrDesc("starts", IntTupleUnpack),
    ],
)

transposeDesc = OperatorDescriptor(
    inputDescriptor = IoDesc("data_in"),
    outputDescriptor = IoDesc("data_out"),
    attrDescriptors = [AttrDesc("perm", IntTupleUnpack)],
)


class CeilMode(IntEnum):
    floor = 0
    ceil = 1


maxPoolDesc = OperatorDescriptor(inputDescriptor = IoDesc("data_in"),
                                 outputDescriptor = IoDesc("data_out"),
                                 attrDescriptors = [
                                     AttrDesc("ceil_mode", unpacker = CeilMode, default = CeilMode.floor),
                                     AttrDesc("kernel_shape", IntTupleUnpack),
                                     AttrDesc("pads", IntTupleUnpack),
                                     AttrDesc("strides", IntTupleUnpack),
                                 ])


class PadMode(str, Enum):
    constant = "constant"
    reflect = "reflect"
    edge = "edge"
    wrap = "wrap"


# Opset 24
padDesc = OperatorDescriptor(
    inputDescriptor = IoDesc(["data_in", "pads"], ["constant_value", "axes"]),
    outputDescriptor = IoDesc("data_out"),
    attrDescriptors = [
        AttrDesc('mode', unpacker = PadMode, default = PadMode.constant),
    ],
)


class PadModeOld(str, Enum):
    constant = "constant"
    reflect = "reflect"
    edge = "edge"


padDescOld = OperatorDescriptor(
    inputDescriptor = IoDesc("data_in"),
    outputDescriptor = IoDesc("data_out"),
    attrDescriptors = [
        AttrDesc("mode", unpacker = PadModeOld, default = PadModeOld.constant),
        AttrDesc("pads", IntTupleUnpack),
        AttrDesc("value", FloatUnpack),
    ],
)

addDesc = OperatorDescriptor(
    inputDescriptor = VariadicIoDesc("data_in", minNumTensors = 2),
    outputDescriptor = IoDesc("data_out"),
    attrDescriptors = [],
)


class ReduceMeanDescriptor(OperatorDescriptor):

    def canonicalize(self, node: gs.Node, opset: int) -> bool:
        if opset < 18:
            if "axes" in node.attrs:
                attrToTensor(node, "axes")
        return super().canonicalize(node, opset)


# Opset 18
reduceMeanDesc = ReduceMeanDescriptor(
    inputDescriptor = IoDesc("data_in", optional = "axes"),
    outputDescriptor = IoDesc("data_out"),
    attrDescriptors = [
        AttrDesc("keepdims", unpacker = BoolUnpack, default = True),
        AttrDesc("noop_with_empty_axes", unpacker = BoolUnpack, default = False),
    ],
)

reduceSumDesc = OperatorDescriptor(
    inputDescriptor = IoDesc("data_in", optional = "axes"),
    outputDescriptor = IoDesc("data_out"),
    attrDescriptors = [
        AttrDesc("keepdims", unpacker = BoolUnpack, default = True),
        AttrDesc("noop_with_empty_axes", unpacker = BoolUnpack, default = False),
    ],
)

softmaxDesc = OperatorDescriptor(
    inputDescriptor = IoDesc("data_in"),
    outputDescriptor = IoDesc("data_out"),
    attrDescriptors = [AttrDesc("axis", IntUnpack, default = -1)],
)

softmaxGradDesc = OperatorDescriptor(
    inputDescriptor = IoDesc(["upstream_grad", "softmax_output"]),
    outputDescriptor = IoDesc("softmax_grad"),
    attrDescriptors = [AttrDesc("axis", IntUnpack, default = -1)],
)

iSoftmaxDesc = OperatorDescriptor(
    inputDescriptor = IoDesc("data_in"),
    outputDescriptor = IoDesc("data_out"),
    attrDescriptors = [
        AttrDesc("axis", IntUnpack, default = -1),
        AttrDesc("coeffA", IntUnpack),
        AttrDesc("coeffB", IntUnpack),
        AttrDesc("coeffC", IntUnpack),
        AttrDesc("log2", IntUnpack),
        AttrDesc("n_levels", IntUnpack),
    ],
)

itaMaxDesc = OperatorDescriptor(
    inputDescriptor = IoDesc("data_in"),
    outputDescriptor = IoDesc("data_out"),
    attrDescriptors = [
        AttrDesc("axis", IntUnpack, default = -1),
        AttrDesc("n_levels", IntUnpack),
    ],
)

itaPartialMaxDesc = OperatorDescriptor(
    inputDescriptor = IoDesc("data_in"),
    outputDescriptor = IoDesc("data_out"),
    attrDescriptors = [
        AttrDesc("axis", IntUnpack, default = -1),
        AttrDesc("n_levels", IntUnpack),
        AttrDesc("group_width", IntUnpack),
    ],
)


class GeluApprox(str, Enum):
    tanh = "tanh"
    none = "none"


geluDesc = OperatorDescriptor(
    inputDescriptor = IoDesc("data_in"),
    outputDescriptor = IoDesc("data_out"),
    attrDescriptors = [
        AttrDesc("approximate", GeluApprox, default = GeluApprox.none),
    ],
)

iGeluDesc = OperatorDescriptor(
    inputDescriptor = IoDesc("data_in"),
    outputDescriptor = IoDesc("data_out"),
    attrDescriptors = [
        AttrDesc("b", IntUnpack),
        AttrDesc("one", IntUnpack),
    ],
)

requantizedIGeluDesc = OperatorDescriptor(inputDescriptor = IoDesc(["data_in", "mul", "add", "shift"]),
                                          outputDescriptor = IoDesc("data_out"),
                                          attrDescriptors = [
                                              AttrDesc("b", IntUnpack),
                                              AttrDesc("one", IntUnpack),
                                          ])

iHardswishDesc = OperatorDescriptor(inputDescriptor = IoDesc("data_in"),
                                    outputDescriptor = IoDesc("data_out"),
                                    attrDescriptors = [
                                        AttrDesc("one_over_six", IntUnpack),
                                        AttrDesc("six", IntUnpack),
                                        AttrDesc("three", IntUnpack),
                                    ])

requantizedIHardswishDesc = OperatorDescriptor(inputDescriptor = IoDesc("data_in"),
                                               outputDescriptor = IoDesc("data_out"),
                                               attrDescriptors = [
                                                   AttrDesc("one_over_six", IntUnpack),
                                                   AttrDesc("six", IntUnpack),
                                                   AttrDesc("three", IntUnpack),
                                                   AttrDesc("mul", IntUnpack),
                                                   AttrDesc("add", IntUnpack),
                                                   AttrDesc("shift", IntUnpack),
                                               ])

iNoNormDesc = OperatorDescriptor(inputDescriptor = IoDesc(["data_in", "weights", "bias"]),
                                 outputDescriptor = IoDesc("data_out"),
                                 attrDescriptors = [
                                     AttrDesc("D", IntUnpack),
                                     AttrDesc("mul", IntUnpack),
                                     AttrDesc("n_levels", IntUnpack),
                                 ])

quantDesc = OperatorDescriptor(
    inputDescriptor = IoDesc("data_in"),
    outputDescriptor = IoDesc("data_out"),
    attrDescriptors = [
        AttrDesc("scale", FloatUnpack),
        AttrDesc("zero_point", FloatUnpack),
        AttrDesc("bit_width", IntUnpack),
        AttrDesc("signed", BoolUnpack, default = True),
        AttrDesc("min_val",
                 IntUnpack,
                 default = lambda node: -(2**(node.attrs["bit_width"] - 1)) if node.attrs["signed"] else 0),
        AttrDesc("max_val",
                 IntUnpack,
                 default = lambda node: 2**(node.attrs["bit_width"] - 1) - 1
                 if node.attrs["signed"] else 2**node.attrs["bit_width"] - 1),
    ],
)


class AutoPad(str, Enum):
    NOTSET = "NOTSET"
    SAME_UPPER = "SAME_UPPER"
    SAME_LOWER = "SAME_LOWER"
    VALID = "VALID"


def _dilationsDefault(node: gs.Node) -> Tuple[int, ...]:
    # Remove 2 dims for input and output channels
    nSpatialDims = len(node.inputs[1].shape) - 2
    return tuple([1] * nSpatialDims)


def _kernelShapeDefault(node: gs.Node) -> Tuple[int, ...]:
    # Remove 2 dims for input and output channels
    nSpatialDims = len(node.inputs[1].shape) - 2
    return node.inputs[1].shape[-nSpatialDims:]


def _stridesDefault(node: gs.Node) -> Tuple[int, ...]:
    # Remove 2 dims for input and output channels
    nSpatialDims = len(node.inputs[1].shape) - 2
    return tuple([1] * nSpatialDims)


def _padsDefault(node: gs.Node) -> Tuple[int, ...]:
    # Remove 2 dims for input and output channels
    nSpatialDims = len(node.inputs[1].shape) - 2
    # Two 0's per dimension for begin and end
    return tuple([0] * (2 * nSpatialDims))


convDesc = OperatorDescriptor(
    inputDescriptor = IoDesc(["data_in", "weight"], optional = "bias"),
    outputDescriptor = IoDesc("data_out"),
    attrDescriptors = [
        AttrDesc("auto_pad", AutoPad, default = AutoPad.NOTSET),
        AttrDesc("dilations", IntTupleUnpack, default = _dilationsDefault),
        AttrDesc("group", IntUnpack, default = 1),
        AttrDesc("kernel_shape", IntTupleUnpack, default = _kernelShapeDefault),
        AttrDesc("pads", IntTupleUnpack, default = _padsDefault),
        AttrDesc("strides", IntTupleUnpack, default = _stridesDefault),
    ],
)


class RequantizedOperatorDescriptor(OperatorDescriptor):

    def canonicalize(self, node: gs.Node, opset: int) -> bool:
        if "n_levels_out" in node.attrs and "n_levels" in node.attrs:
            # TODO: Change to log
            print("[WARNING] Requantized operator cannot have n_levels_out and n_levels in its attributes")
            return False

        if "n_levels_out" in node.attrs:
            node.attrs["n_levels"] = node.attrs["n_levels_out"]
            node.attrs.pop("n_levels_out")

        return super().canonicalize(node, opset)


requantizedConvDesc = RequantizedOperatorDescriptor(
    inputDescriptor = IoDesc(["data_in", "weight", "mul", "add"], optional = ["shift"]),
    outputDescriptor = IoDesc("data_out"),
    attrDescriptors = [
        # Conv attrs
        AttrDesc("auto_pad", AutoPad, default = AutoPad.NOTSET),
        AttrDesc("dilations", IntTupleUnpack, default = _dilationsDefault),
        AttrDesc("group", IntUnpack, default = 1),
        AttrDesc("kernel_shape", IntTupleUnpack, default = _kernelShapeDefault),
        AttrDesc("pads", IntTupleUnpack, default = _padsDefault),
        AttrDesc("strides", IntTupleUnpack, default = _stridesDefault),
        # RequantizedShift attrs
        AttrDesc("n_levels", IntUnpack),
        AttrDesc("signed", BoolUnpack),
        AttrDesc("div", IntUnpack),
    ],
)

dequantDesc = OperatorDescriptor(
    inputDescriptor = IoDesc("data_in"),
    outputDescriptor = IoDesc("data_out"),
    attrDescriptors = [
        AttrDesc("scale", FloatUnpack),
        AttrDesc("zero_point", FloatUnpack),
        AttrDesc("bit_width", IntUnpack),
        AttrDesc("signed", BoolUnpack),
    ],
)

divDesc = OperatorDescriptor(
    inputDescriptor = IoDesc(["input1", "input2"]),
    outputDescriptor = IoDesc("output"),
    attrDescriptors = [],
)

integerDivDescriptor = OperatorDescriptor(
    inputDescriptor = IoDesc(["A", "B"]),
    outputDescriptor = IoDesc("C"),
    attrDescriptors = [
        AttrDesc("Delta", IntUnpack),
        AttrDesc("eps", IntUnpack),
        AttrDesc("eta", IntUnpack),
    ],
)

requantizedIntegerDivDescriptor = RequantizedOperatorDescriptor(
    inputDescriptor = IoDesc(["A", "B", "requant_mul", "requant_add", "requant_div"]),
    outputDescriptor = IoDesc("C"),
    attrDescriptors = [
        # IntegerDiv attrs
        AttrDesc("Delta", IntUnpack),
        AttrDesc("eps", IntUnpack),
        AttrDesc("eta", IntUnpack),
        # RequantizedShift attrs
        AttrDesc("n_levels", IntUnpack),
        AttrDesc("signed", BoolUnpack),
        AttrDesc("div", IntUnpack),
    ])

debugPrintDesc = OperatorDescriptor(
    inputDescriptor = IoDesc("data_in"),
    outputDescriptor = IoDesc("data_out"),
    attrDescriptors = [],
)

layerNormalizationDesc = OperatorDescriptor(
    inputDescriptor = IoDesc(["data_in", "weight", "bias"]),
    outputDescriptor = IoDesc("data_out"),
    attrDescriptors = [AttrDesc("epsilon", FloatUnpack)],
)

iLayerNormDesc = OperatorDescriptor(
    inputDescriptor = IoDesc(["data_in", "weight", "bias"]),
    outputDescriptor = IoDesc("data_out"),
    attrDescriptors = [AttrDesc("D", IntUnpack), AttrDesc("n_levels", IntUnpack)],
)

flattenDesc = OperatorDescriptor(
    inputDescriptor = IoDesc("data_in"),
    outputDescriptor = IoDesc("data_out"),
    attrDescriptors = [AttrDesc("axis", IntUnpack, default = 1)],
)

gatherDesc = OperatorDescriptor(
    inputDescriptor = IoDesc(["data_in", "indices"]),
    outputDescriptor = IoDesc("data_out"),
    attrDescriptors = [AttrDesc("axis", IntUnpack, default = 0)],
)

# Opset <= 11
unsqueezeDesc = OperatorDescriptor(
    inputDescriptor = IoDesc("data_in"),
    outputDescriptor = IoDesc("data_out"),
    attrDescriptors = [AttrDesc("axes", IntTupleUnpack)],
)

# Opset <= 11
squeezeDesc = OperatorDescriptor(
    inputDescriptor = IoDesc("data_in"),
    outputDescriptor = IoDesc("data_out"),
    attrDescriptors = [AttrDesc("axes", IntTupleUnpack)],
)

mulDesc = OperatorDescriptor(
    inputDescriptor = IoDesc(["A", "B"]),
    outputDescriptor = IoDesc("C"),
    attrDescriptors = [],
)

matMulDesc = OperatorDescriptor(
    inputDescriptor = IoDesc(["A", "B"]),
    outputDescriptor = IoDesc("data_out"),
    attrDescriptors = [],
)

rqMatMulDesc = RequantizedOperatorDescriptor(
    inputDescriptor = IoDesc(["A", "B", "add", "mul"]),
    outputDescriptor = IoDesc("data_out"),
    attrDescriptors = [
        # RequantizedShift attrs
        AttrDesc("n_levels", IntUnpack),
        AttrDesc("signed", BoolUnpack),
        AttrDesc("div", IntUnpack),
    ],
)

gemmDesc = OperatorDescriptor(
    inputDescriptor = IoDesc(["A", "B"], optional = ["C"]),
    outputDescriptor = IoDesc("data_out"),
    attrDescriptors = [
        AttrDesc("alpha", FloatUnpack, default = 1.0),
        AttrDesc("beta", FloatUnpack, default = 1.0),
        AttrDesc("transA", BoolUnpack, default = False),
        AttrDesc("transB", BoolUnpack, default = False),
    ],
)

rqGemmDesc = RequantizedOperatorDescriptor(
    inputDescriptor = IoDesc(["A", "B", "C", "add", "mul"]),
    outputDescriptor = IoDesc("data_out"),
    attrDescriptors = [
        AttrDesc("alpha", FloatUnpack, default = 1.0),
        AttrDesc("beta", FloatUnpack, default = 1.0),
        AttrDesc("transA", BoolUnpack, default = False),
        AttrDesc("transB", BoolUnpack, default = False),
        # RequantizedShift attrs
        AttrDesc("n_levels", IntUnpack),
        AttrDesc("signed", BoolUnpack),
        AttrDesc("div", IntUnpack),
    ])

requantizedGemmDesc = RequantizedOperatorDescriptor(
    inputDescriptor = IoDesc(["A", "B", "add", "mul"]),  # Important diff to RQGemm
    outputDescriptor = IoDesc("data_out"),
    attrDescriptors = [
        AttrDesc("alpha", FloatUnpack, default = 1.0),
        AttrDesc("beta", FloatUnpack, default = 1.0),
        AttrDesc("transA", BoolUnpack, default = False),
        AttrDesc("transB", BoolUnpack, default = False),
        # RequantizedShift attrs
        AttrDesc("n_levels", IntUnpack),
        AttrDesc("signed", BoolUnpack),
        AttrDesc("div", IntUnpack),
    ])

linearAttentionDesc = OperatorDescriptor(
    inputDescriptor = IoDesc(
        ["q", "k", "v", "wq_weight", "wq_bias", "wk_weight", "wk_bias", "wv_weight", "wv_bias", "wo_weight",
         "wo_bias"]),
    outputDescriptor = IoDesc("data_out"),
    attrDescriptors = [
        AttrDesc("preattn_requant_mul", IntTupleUnpack),
        AttrDesc("preattn_requant_div", IntTupleUnpack),
        AttrDesc("normalizer_requant_mul", IntTupleUnpack),
        AttrDesc("normalizer_requant_shift", IntTupleUnpack),
        AttrDesc("normalizer_requant_div", IntTupleUnpack),
        AttrDesc("postattn_requant_mul", IntTupleUnpack),
        AttrDesc("postattn_requant_shift", IntTupleUnpack),
        AttrDesc("postattn_requant_div", IntTupleUnpack),
        AttrDesc("wo_requant_mul", IntTupleUnpack),
        AttrDesc("wo_requant_shift", IntTupleUnpack),
        AttrDesc("wo_requant_div", IntTupleUnpack),
        AttrDesc("wq_requant_mul", IntTupleUnpack),
        AttrDesc("wq_requant_shift", IntTupleUnpack),
        AttrDesc("wq_requant_div", IntTupleUnpack),
        AttrDesc("wk_requant_mul", IntTupleUnpack),
        AttrDesc("wk_requant_shift", IntTupleUnpack),
        AttrDesc("wk_requant_div", IntTupleUnpack),
        AttrDesc("wv_requant_mul", IntTupleUnpack),
        AttrDesc("wv_requant_shift", IntTupleUnpack),
        AttrDesc("wv_requant_div", IntTupleUnpack),
        AttrDesc("Delta", IntUnpack),
        AttrDesc("eps", IntUnpack),
        AttrDesc("act_type", IntUnpack),
        AttrDesc("n_levels", IntUnpack),
        AttrDesc("dim", IntUnpack),
        AttrDesc("dim_head", IntUnpack),
        AttrDesc("heads", IntUnpack),
    ],
)

clcaDesc = OperatorDescriptor(
    inputDescriptor = IoDesc([
        "q", "k", "wq_weight", "wq_bias", "wk_weight", "wk_bias", "wo_weight", "wo_bias", "wq_requant_mul",
        "wq_requant_add", "wq_requant_div", "wk_requant_mul", "wk_requant_add", "wk_requant_div", "wv_requant_mul",
        "wv_requant_add", "wv_requant_div", "kdiv_requant_mul", "kdiv_requant_add", "kdiv_requant_div",
        "preattn_requant_mul", "preattn_requant_add", "preattn_requant_div", "postattn_requant_mul",
        "postattn_requant_add", "postattn_requant_div", "wo_requant_mul", "wo_requant_add", "wo_requant_div"
    ]),
    outputDescriptor = IoDesc("data_out"),
    attrDescriptors = [
        AttrDesc("Delta", IntUnpack),
        AttrDesc("eps", IntUnpack),
        AttrDesc("eta", IntUnpack),
        AttrDesc("act_type", IntUnpack),
        AttrDesc("n_levels", IntUnpack),
        AttrDesc("dim", IntUnpack),
        AttrDesc("dim_head", IntUnpack),
        AttrDesc("out_dim", IntUnpack),
        AttrDesc("heads", IntUnpack),
    ],
)

mhsaDesc = OperatorDescriptor(
    inputDescriptor = IoDesc(
        ["q", "k", "v", "wq_weight", "wq_bias", "wk_weight", "wk_bias", "wv_weight", "wv_bias", "wo_weight",
         "wo_bias"]),
    outputDescriptor = IoDesc("data_out"),
    attrDescriptors = [
        AttrDesc("preattn_requant_mul", IntTupleIfNotSingleItemUnpack),
        AttrDesc("preattn_requant_div", IntTupleIfNotSingleItemUnpack),
        AttrDesc("postattn_requant_mul", IntTupleIfNotSingleItemUnpack),
        AttrDesc("postattn_requant_div", IntTupleIfNotSingleItemUnpack),
        AttrDesc("wo_requant_mul", IntTupleIfNotSingleItemUnpack),
        AttrDesc("wo_requant_div", IntTupleIfNotSingleItemUnpack),
        AttrDesc("wq_requant_mul", IntTupleIfNotSingleItemUnpack),
        AttrDesc("wq_requant_div", IntTupleIfNotSingleItemUnpack),
        AttrDesc("wk_requant_mul", IntTupleIfNotSingleItemUnpack),
        AttrDesc("wk_requant_div", IntTupleIfNotSingleItemUnpack),
        AttrDesc("wv_requant_mul", IntTupleIfNotSingleItemUnpack),
        AttrDesc("wv_requant_div", IntTupleIfNotSingleItemUnpack),
        AttrDesc("n_levels", IntUnpack),
        AttrDesc("dim", IntUnpack),
        AttrDesc("dim_head", IntUnpack),
        AttrDesc("heads", IntUnpack),
        AttrDesc("signed", BoolUnpack),
    ],
)

reluDesc = OperatorDescriptor(
    inputDescriptor = IoDesc("data_in"),
    outputDescriptor = IoDesc("data_out"),
    attrDescriptors = [],
)

reshapeDesc = OperatorDescriptor(
    inputDescriptor = IoDesc(["data_in", "shape"]),
    outputDescriptor = IoDesc("data_out"),
    attrDescriptors = [],
)

requantShiftDesc = RequantizedOperatorDescriptor(
    inputDescriptor = IoDesc(["data_in", "mul", "add"]),
    outputDescriptor = IoDesc("data_out"),
    attrDescriptors = [
        AttrDesc("n_levels", IntUnpack),
        AttrDesc("signed", BoolUnpack),
        AttrDesc("div", IntUnpack),
    ],
)


class RequantizedAddDescriptor(OperatorDescriptor):

    def canonicalize(self, node: gs.Node, opset: int) -> bool:
        for tensor in ["rqs1", "rqs2", "rqsOut"]:
            n_levels = f"{tensor}_n_levels"
            n_levels_out = f"{tensor}_n_levels_out"
            if n_levels_out in node.attrs and n_levels in node.attrs:
                # TODO: Change to log
                print(
                    f"[WARNING] RequantizedAdd tensor {tensor} cannot have {n_levels_out} and {n_levels} in its attributes"
                )
                return False

            if n_levels_out in node.attrs:
                node.attrs[n_levels] = node.attrs[n_levels_out]
                node.attrs.pop(n_levels_out)

        return super().canonicalize(node, opset)


requantizedAddDesc = RequantizedAddDescriptor(
    inputDescriptor = IoDesc(["data_in_0", "data_in_1"]),
    outputDescriptor = IoDesc("data_out"),
    attrDescriptors = [
        AttrDesc("rqs1_n_levels", IntUnpack),
        AttrDesc("rqs1_mul", IntUnpack),
        AttrDesc("rqs1_add", IntUnpack),
        AttrDesc("rqs1_div", IntUnpack),
        AttrDesc("rqs1_signed", BoolUnpack),
        AttrDesc("rqs1_n_levels", IntUnpack),
        AttrDesc("rqs2_mul", IntUnpack),
        AttrDesc("rqs2_add", IntUnpack),
        AttrDesc("rqs2_div", IntUnpack),
        AttrDesc("rqs2_signed", BoolUnpack),
        AttrDesc("rqs2_n_levels", IntUnpack),
        AttrDesc("rqsOut_mul", IntUnpack),
        AttrDesc("rqsOut_add", IntUnpack),
        AttrDesc("rqsOut_div", IntUnpack),
        AttrDesc("rqsOut_signed", BoolUnpack),
        AttrDesc("rqsOut_n_levels", IntUnpack),
    ],
)

sgdDesc = OperatorDescriptor(
    inputDescriptor = IoDesc(["weight", "grad"]),
    outputDescriptor = IoDesc("weight_updated"),
    attrDescriptors = [AttrDesc("lr", FloatUnpack)],
)

defaultOperatorDescriptors: Dict[str, OperatorDescriptor] = {
    "Add": addDesc,
    "CLCA": clcaDesc,
    "Concat": concatDesc,
    "Conv": convDesc,
    "DebugPrint": debugPrintDesc,
    "Dequant": dequantDesc,
    "Div": divDesc,
    "Flatten": flattenDesc,
    "Gather": gatherDesc,
    "Gelu": geluDesc,
    "Gemm": gemmDesc,
    "ITAMax": itaMaxDesc,
    "ITAPartialMax": itaPartialMaxDesc,
    "IntegerDiv": integerDivDescriptor,
    "IntegerMean": reduceMeanDesc,
    "LayerNormalization": layerNormalizationDesc,
    "LinearAttention": linearAttentionDesc,
    "MHSA": mhsaDesc,
    "MatMul": matMulDesc,
    "MatMulInteger": matMulDesc,
    "MaxPool": maxPoolDesc,
    "Mul": mulDesc,
    "Pad": padDescOld,
    "Quant": quantDesc,
    "RQGemm": rqGemmDesc,
    "RQIntegerDiv": requantizedIntegerDivDescriptor,
    "RQMatMul": rqMatMulDesc,
    "ReduceMean": reduceMeanDesc,
    "ReduceSum": reduceSumDesc,
    "Relu": reluDesc,
    "RequantizedAdd": requantizedAddDesc,
    "RequantizedConv": requantizedConvDesc,
    "RequantizedGemm": requantizedGemmDesc,
    "RequantizediGELU": requantizedIGeluDesc,
    "RequantizediHardswish": requantizedIHardswishDesc,
    "RequantShift": requantShiftDesc,
    "Reshape": reshapeDesc,
    "SGD": sgdDesc,
    "Slice": sliceDesc,
    "Softmax": softmaxDesc,
    "SoftmaxGrad": softmaxGradDesc,
    "Squeeze": squeezeDesc,
    "Transpose": transposeDesc,
    "Unsqueeze": unsqueezeDesc,
    "iGELU": iGeluDesc,
    "iHardswish": iHardswishDesc,
    "iLayerNorm": iLayerNormDesc,
    "iNoNorm": iNoNormDesc,
    "iRMSNorm": iRMSNormDesc,
    "iSoftmax": iSoftmaxDesc,
}

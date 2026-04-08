# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

import aie.ir as ir

from Deeploy.AbstractDataTypes import PointerClass
from Deeploy.CommonExtensions.DataTypes import bfloat16_t
from Deeploy.DeeployTypes import NodeBinding
from Deeploy.MLIRDataTypes import MLIRCodeTransformation
from Deeploy.Targets.XDNA2.CodeTransformationPasses.MLIRComputeCorePass import MLIRComputeCorePass
from Deeploy.Targets.XDNA2.CodeTransformationPasses.MLIRObjectFifoPass import MLIRObjectFifoPass
from Deeploy.Targets.XDNA2.CodeTransformationPasses.MLIRRuntimeSequencePass import MLIRRuntimeSequencePass
from Deeploy.Targets.XDNA2.Templates import AddTemplate, SiLUTemplate
from Deeploy.Targets.XDNA2.TypeCheckers import XDNA2AddChecker, XDNA2SiLUChecker

_ADD_INPUT_KEYS = ['data_in_1', 'data_in_2']
_ADD_OUTPUT_KEYS = ['data_out']

_SILU_INPUT_KEYS = ['data_in']
_SILU_OUTPUT_KEYS = ['data_out']

# JUNGVI: TODO: This logic should not be boiled down for 1 operator but should be applied on every nodes of the network
# Likewise the kernelName and object file name should be specified in the node template of each operator.
XDNA2AddTransformer = MLIRCodeTransformation(
    devicePasses = [
        MLIRObjectFifoPass(
            inputTensorKeys = _ADD_INPUT_KEYS,
            outputTensorKeys = _ADD_OUTPUT_KEYS,
            kernelFuncName = "eltwise_add_bf16_vector",
            kernelObjFile = "add.o",
        ),
        MLIRComputeCorePass(
            inputTensorKeys = _ADD_INPUT_KEYS,
            outputTensorKeys = _ADD_OUTPUT_KEYS,
        ),
    ],
    runtimeSequencePasses = [
        MLIRRuntimeSequencePass(
            inputTensorKeys = _ADD_INPUT_KEYS,
            outputTensorKeys = _ADD_OUTPUT_KEYS,
        ),
    ],
)


def _unary_kernel_arg_types(tile_ty):
    """Kernel arg types for unary ops: (input, output, size)."""
    i32 = ir.IntegerType.get_signless(32)
    return [tile_ty, tile_ty, i32]


XDNA2SiLUTransformer = MLIRCodeTransformation(
    devicePasses = [
        MLIRObjectFifoPass(
            inputTensorKeys = _SILU_INPUT_KEYS,
            outputTensorKeys = _SILU_OUTPUT_KEYS,
            kernelFuncName = "silu_bf16",
            kernelObjFile = "silu.o",
            kernelArgTypes = _unary_kernel_arg_types,
        ),
        MLIRComputeCorePass(
            inputTensorKeys = _SILU_INPUT_KEYS,
            outputTensorKeys = _SILU_OUTPUT_KEYS,
        ),
    ],
    runtimeSequencePasses = [
        MLIRRuntimeSequencePass(
            inputTensorKeys = _SILU_INPUT_KEYS,
            outputTensorKeys = _SILU_OUTPUT_KEYS,
        ),
    ],
)

XDNA2AddBindings = [
    NodeBinding(
        XDNA2AddChecker([PointerClass(bfloat16_t), PointerClass(bfloat16_t)], [PointerClass(bfloat16_t)]),
        AddTemplate.referenceTemplate,
        XDNA2AddTransformer,
    )
]

XDNA2SiLUBindings = [
    NodeBinding(
        XDNA2SiLUChecker([PointerClass(bfloat16_t)], [PointerClass(bfloat16_t)]),
        SiLUTemplate.referenceTemplate,
        XDNA2SiLUTransformer,
    )
]

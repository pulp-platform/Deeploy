# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from Deeploy.AbstractDataTypes import PointerClass
from Deeploy.CommonExtensions.DataTypes import bfloat16_t
from Deeploy.DeeployTypes import NodeBinding
from Deeploy.MLIRDataTypes import MLIRCodeTransformation
from Deeploy.Targets.XDNA2.CodeTransformationPasses.MLIRComputeCorePass import MLIRComputeCorePass
from Deeploy.Targets.XDNA2.CodeTransformationPasses.MLIRObjectFifoPass import MLIRObjectFifoPass
from Deeploy.Targets.XDNA2.CodeTransformationPasses.MLIRRuntimeSequencePass import MLIRRuntimeSequencePass
from Deeploy.Targets.XDNA2.Templates import AddTemplate
from Deeploy.Targets.XDNA2.TypeCheckers import XDNA2AddChecker

_ADD_INPUT_KEYS = ['data_in_1', 'data_in_2']
_ADD_OUTPUT_KEYS = ['data_out']

# JUNGVI: TODO: This logic should not be boiled down for 1 operator but should be applied on every nodes of the network
# Likewise the kernelName and object file name should be specified in the node template of each operator.
XDNA2Transformer = MLIRCodeTransformation(
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

XDNA2AddBindings = [
    NodeBinding(
        XDNA2AddChecker([PointerClass(bfloat16_t), PointerClass(bfloat16_t)], [PointerClass(bfloat16_t)]),
        AddTemplate.referenceTemplate,
        XDNA2Transformer,
    )
]

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
from Deeploy.Targets.XDNA2.Templates import AddTemplate, LayerNormTemplate, SiLUTemplate
from Deeploy.Targets.XDNA2.TypeCheckers import XDNA2AddChecker, XDNA2LayerNormChecker, XDNA2SiLUChecker

XDNA2Transformer = MLIRCodeTransformation(
    devicePasses = [
        MLIRObjectFifoPass(),
        MLIRComputeCorePass(),
    ],
    runtimeSequencePasses = [
        MLIRRuntimeSequencePass(),
    ],
)

XDNA2AddBindings = [
    NodeBinding(
        XDNA2AddChecker([PointerClass(bfloat16_t), PointerClass(bfloat16_t)], [PointerClass(bfloat16_t)]),
        AddTemplate.referenceTemplate,
        XDNA2Transformer,
    )
]

XDNA2SiLUBindings = [
    NodeBinding(
        XDNA2SiLUChecker([PointerClass(bfloat16_t)], [PointerClass(bfloat16_t)]),
        SiLUTemplate.referenceTemplate,
        XDNA2Transformer,
    )
]

XDNA2LayerNormBindings = [
    NodeBinding(
        XDNA2LayerNormChecker(
            [PointerClass(bfloat16_t), PointerClass(bfloat16_t),
             PointerClass(bfloat16_t)], [PointerClass(bfloat16_t)]),
        LayerNormTemplate.referenceTemplate,
        XDNA2Transformer,
    )
]

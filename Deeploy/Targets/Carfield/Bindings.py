# SPDX-License-Identifier: Apache-2.0

# Imports from Deeploy
from Deeploy.AbstractDataTypes import PointerClass
from Deeploy.CommonExtensions.DataTypes import float32_t, int32_t, int64_t
from Deeploy.Targets.Carfield.CodeTransformationPasses.CarfieldClusterSynch import CarfieldSynchCoresPass
from Deeploy.DeeployTypes import CodeTransformation, NodeBinding
from Deeploy.CommonExtensions.CodeTransformationPasses.MemoryAllocation import MemoryManagementGeneration
from Deeploy.Targets.Generic.TypeCheckers import AddChecker, ConcatChecker, ConvChecker, DivChecker, MaxPoolChecker, PadChecker, \
    ReluChecker
from .TypeCheckers import CarfieldArgMaxChecker, CarfieldCastChecker
from . import Templates  # imports Carfield/Templates/

# Transformer: runs memory management (malloc/free) on intermediate buffers
BasicTransformer = CodeTransformation([CarfieldSynchCoresPass(), MemoryManagementGeneration()])

# CarfieldAddBindings: tells Deeploy how to handle an ONNX Add node on Carfield
# only for float32 inputs and float32 output
# generates the C call using our AddTemplate
# manages buffer memory with BasicTransformer.
# ([every input tensor],[every output tensor])
CarfieldAddBindings = [
    NodeBinding(
        AddChecker([PointerClass(float32_t), PointerClass(float32_t)], [PointerClass(float32_t)]),
        Templates.AddTemplate.referenceTemplate,
        BasicTransformer)
]

CarfieldPad2DBindings = [
    NodeBinding(
        PadChecker([PointerClass(float32_t)], [PointerClass(float32_t)]),
        Templates.PadTemplate.reference2DTemplate,
        BasicTransformer
    )
]

CarfieldConv2DBindings = [
    NodeBinding(
        ConvChecker([PointerClass(float32_t),
                     PointerClass(float32_t),
                     PointerClass(float32_t)], [PointerClass(float32_t)]),
        Templates.ConvTemplate.referenceTemplate,
        BasicTransformer)
]

CarfieldReluBindings = [
    NodeBinding(
        ReluChecker([PointerClass(float32_t)], [PointerClass(float32_t)]),
        Templates.ReluTemplate.referenceTemplate,
        BasicTransformer)
]

CarfieldSubBindings = [
    NodeBinding(
        AddChecker([PointerClass(float32_t), PointerClass(float32_t)], [PointerClass(float32_t)]),
        Templates.SubTemplate.referenceTemplate,
        BasicTransformer)
]

CarfieldDivBindings = [
    NodeBinding(
        DivChecker([PointerClass(float32_t), PointerClass(float32_t)], [PointerClass(float32_t)]),
        Templates.DivTemplate.referenceTemplate,
        BasicTransformer)
]

CarfieldMaxPool2DBindings = [
    NodeBinding(
        MaxPoolChecker([PointerClass(float32_t)], [PointerClass(float32_t)]),
        Templates.MaxPoolTemplate.reference2DTemplate,
        BasicTransformer)
]

# MaxPoolChecker reused
CarfieldResizeNearest2DBindings = [
    NodeBinding(
        MaxPoolChecker([PointerClass(float32_t)], [PointerClass(float32_t)]),
        Templates.ResizeTemplate.referenceNearest2DTemplate,
        BasicTransformer)
]

CarfieldConcat2Bindings = [
    NodeBinding(
        ConcatChecker([PointerClass(float32_t), PointerClass(float32_t)], [PointerClass(float32_t)]),
        Templates.ConcatTemplate.reference2InputTemplate,
        BasicTransformer)
]

CarfieldArgMaxChannelBindings = [
    NodeBinding(
        CarfieldArgMaxChecker([PointerClass(float32_t)], [PointerClass(int64_t)]),
        Templates.ArgMaxTemplate.referenceChannelTemplate,
        BasicTransformer)
]

CarfieldCastInt32Bindings = [
    NodeBinding(
        CarfieldCastChecker([PointerClass(int64_t)], [PointerClass(int32_t)]),
        Templates.CastTemplate.referenceInt64ToInt32Template,
        BasicTransformer)
]

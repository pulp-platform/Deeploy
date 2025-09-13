# Copyright (C) 2025, ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

from Deeploy.AbstractDataTypes import PointerClass
from Deeploy.CommonExtensions.CodeTransformationPasses.MemoryAllocation import ArgumentStructGeneration, \
    MemoryManagementGeneration
from Deeploy.CommonExtensions.DataTypes import int8_t, int16_t, int32_t, int64_t
from Deeploy.DeeployTypes import CodeTransformation, NodeBinding
from Deeploy.FutureExtension.CodeTransformationPasses.FutureCodeTransformation import FutureGeneration
from Deeploy.Targets.CortexM.Templates import CLCATemplate, ConvTemplate, DWConvTemplate, GEMMTemplate, \
    LinearAttentionTemplate, MaxPool2DTemplate
from Deeploy.Targets.CortexM.TypeCheckers import CMSISConvChecker, CMSISLinearChecker, CMSISMaxPoolChecker
from Deeploy.Targets.Generic.TypeCheckers import CLCAChecker, LinearAttentionChecker

BasicTransformer = CodeTransformation([ArgumentStructGeneration(), MemoryManagementGeneration(), FutureGeneration()])

CMSISCLCABinding = NodeBinding(
    CLCAChecker([PointerClass(int8_t), PointerClass(int8_t)] +
                [PointerClass(int8_t), PointerClass(int32_t)] * 3 +
                [PointerClass(int32_t), PointerClass(int32_t),
                 PointerClass(int32_t)] * 7, [PointerClass(int8_t)]), CLCATemplate.referenceTemplate, BasicTransformer)

CMSISConv1DBinding_16 = NodeBinding(
    CMSISConvChecker([
        PointerClass(int16_t),
        PointerClass(int8_t),
        PointerClass(int32_t),
        PointerClass(int64_t),
        PointerClass(int32_t)
    ], [PointerClass(int16_t)]), ConvTemplate.cmsis1D_16_Template, BasicTransformer)
CMSISConv1DBinding_8 = NodeBinding(
    CMSISConvChecker([
        PointerClass(int8_t),
        PointerClass(int8_t),
        PointerClass(int32_t),
        PointerClass(int32_t),
        PointerClass(int32_t)
    ], [PointerClass(int8_t)]), ConvTemplate.cmsis1D_8_Template, BasicTransformer)
CMSISConv1DBindings = [CMSISConv1DBinding_8, CMSISConv1DBinding_16]

CMSISConv2DBinding = NodeBinding(
    CMSISConvChecker([
        PointerClass(int8_t),
        PointerClass(int8_t),
        PointerClass(int32_t),
        PointerClass(int32_t),
        PointerClass(int32_t)
    ], [PointerClass(int8_t)]), ConvTemplate.cmsis2D_8_Template, BasicTransformer)

CMSISDWConv1DBinding_16 = NodeBinding(
    CMSISConvChecker([
        PointerClass(int16_t),
        PointerClass(int8_t),
        PointerClass(int32_t),
        PointerClass(int64_t),
        PointerClass(int32_t)
    ], [PointerClass(int16_t)]), DWConvTemplate.conv1D_16_Template, BasicTransformer)
CMSISDWConv1DBinding_8 = NodeBinding(
    CMSISConvChecker([
        PointerClass(int8_t),
        PointerClass(int8_t),
        PointerClass(int32_t),
        PointerClass(int32_t),
        PointerClass(int32_t)
    ], [PointerClass(int8_t)]), DWConvTemplate.conv1D_8_Template, BasicTransformer)
CMSISDWConv1DBindings = [CMSISDWConv1DBinding_8, CMSISDWConv1DBinding_16]

CMSISDWConv2DBinding = NodeBinding(
    CMSISConvChecker([
        PointerClass(int8_t),
        PointerClass(int8_t),
        PointerClass(int32_t),
        PointerClass(int32_t),
        PointerClass(int32_t)
    ], [PointerClass(int8_t)]), DWConvTemplate.conv2D_8_Template, BasicTransformer)

CMSISGEMM_16_Binding = NodeBinding(
    CMSISLinearChecker([PointerClass(int16_t),
                        PointerClass(int16_t),
                        PointerClass(int64_t),
                        PointerClass(int64_t)], [PointerClass(int16_t)]), GEMMTemplate.Linear_16_Template,
    BasicTransformer)
CMSISGEMM_8_Binding = NodeBinding(
    CMSISLinearChecker(
        [PointerClass(int8_t), PointerClass(int8_t),
         PointerClass(int32_t),
         PointerClass(int32_t)], [PointerClass(int8_t)]), GEMMTemplate.Linear_8_Template, BasicTransformer)
CMSISGEMMBindings = [CMSISGEMM_8_Binding, CMSISGEMM_16_Binding]

CMSISLinearAttentionBinding = NodeBinding(
    LinearAttentionChecker([PointerClass(int16_t), PointerClass(int16_t),
                            PointerClass(int16_t)] + [PointerClass(int8_t), PointerClass(int64_t)] * 4,
                           [PointerClass(int16_t)]), LinearAttentionTemplate.referenceTemplate, BasicTransformer)

CMSISMaxPool2DBinding = NodeBinding(CMSISMaxPoolChecker([PointerClass(int8_t)], [PointerClass(int8_t)]),
                                    MaxPool2DTemplate.cmsisTemplate, BasicTransformer)

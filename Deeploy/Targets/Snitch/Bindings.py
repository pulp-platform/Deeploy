# ----------------------------------------------------------------------
#
# File: SnitchBindings.py
#
# Last edited: 30.05.2024
#
# Copyright (C) 2024, ETH Zurich and University of Bologna.
#
# Author:
# - Victor Jung, jungvi@iis.ee.ethz.ch, ETH Zurich
#
# ----------------------------------------------------------------------
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the License); you may
# not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an AS IS BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from functools import partial

from Deeploy.AbstractDataTypes import PointerClass
from Deeploy.CommonExtensions.CodeTransformationPasses.Closure import ClosureGeneration, MemoryAwareClosureGeneration
from Deeploy.CommonExtensions.CodeTransformationPasses.MemoryAllocation import ArgumentStructGeneration, \
    MemoryManagementGeneration
from Deeploy.CommonExtensions.DataTypes import int8_t, int32_t, uint8_t
from Deeploy.DeeployTypes import CodeTransformation, NodeBinding
from Deeploy.FutureExtension.CodeTransformationPasses.FutureCodeTransformation import FutureGeneration
from Deeploy.Targets.Generic.Templates import iNoNormTemplate
from Deeploy.Targets.Generic.TypeCheckers import AddChecker, GEMMChecker, SoftmaxChecker, iNoNormChecker
from Deeploy.Targets.Snitch.CodeTransformationPasses import SnitchClusterTiling, SnitchCoreFilterPass, \
    SnitchProfileExecutionBlockPass, SnitchSynchCoresPass
from Deeploy.Targets.Snitch.Templates import AddTemplate, RQAddTemplate, iSoftmaxTemplate
from Deeploy.Targets.Snitch.Templates.GemmTemplate import SnitchGemm_Template
from Deeploy.Targets.Snitch.Templates.RqGemmTemplate import SnitchRqGemm_Template
from Deeploy.Targets.Snitch.TypeCheckers import SnitchRQAddChecker
from Deeploy.TilingExtension.CodeTransformationPasses.TilingVariableReplacement import TilingVariableReplacement

TilingCallClosure = partial(ClosureGeneration, closureSuffix = "_tiling_closure")
MemoryAwareFunctionCallClosure = partial(MemoryAwareClosureGeneration,
                                         closureSuffix = "_closure",
                                         startRegion = "L2",
                                         endRegion = "L1")

BasicTransformer = CodeTransformation(
    [SnitchSynchCoresPass(),
     ArgumentStructGeneration(),
     MemoryManagementGeneration(),
     FutureGeneration()])

TiledTransformer = CodeTransformation([
    SnitchCoreFilterPass("compute"),
    SnitchProfileExecutionBlockPass(),
    TilingVariableReplacement("L1"),
    TilingCallClosure(writeback = False),
    SnitchSynchCoresPass(),
    SnitchClusterTiling("L1"),
    ArgumentStructGeneration(),
    MemoryManagementGeneration("L1"),
    MemoryAwareFunctionCallClosure(writeback = False, generateStruct = True),
    MemoryManagementGeneration()
])

SnitchiSoftmaxBindings = [
    NodeBinding(SoftmaxChecker([PointerClass(_type)], [PointerClass(uint8_t)]), iSoftmaxTemplate.referenceTemplate,
                TiledTransformer) for _type in [int8_t, uint8_t]
]
SnitchiNoNormBindings = [
    NodeBinding(
        iNoNormChecker([PointerClass(_type), PointerClass(int8_t),
                        PointerClass(int32_t)], [PointerClass(int8_t)]), iNoNormTemplate.referenceTemplate,
        TiledTransformer) for _type in [int8_t]
]
SnitchRQAddBindings = [
    NodeBinding(SnitchRQAddChecker([PointerClass(_type), PointerClass(_type)], [PointerClass(_type)]),
                RQAddTemplate.referenceTemplate, TiledTransformer) for _type in [int8_t]
]
SnitchAddBindings = [
    NodeBinding(AddChecker([PointerClass(_type), PointerClass(_type)], [PointerClass(int32_t)]),
                AddTemplate.referenceTemplate, TiledTransformer) for _type in [int8_t]
]
SnitchGemmBindings = [
    NodeBinding(
        GEMMChecker([PointerClass(int8_t), PointerClass(int8_t),
                     PointerClass(int32_t)], [PointerClass(int32_t)]), SnitchGemm_Template, TiledTransformer)
]
SnitchRqGemmBindings = [
    NodeBinding(
        GEMMChecker([
            PointerClass(int8_t),
            PointerClass(int8_t),
            PointerClass(int32_t),
            PointerClass(int32_t),
            PointerClass(int32_t)
        ], [PointerClass(int8_t)]), SnitchRqGemm_Template, TiledTransformer)
]

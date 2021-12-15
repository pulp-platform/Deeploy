# ----------------------------------------------------------------------
#
# File: NeurekaBindings.py
#
# Last edited: 10.07.2024
#
# Copyright (C) 2024, ETH Zurich and University of Bologna.
#
# Author:
# Luka Macan, University of Bologna
# Moritz Scherer, ETH Zurich
#
# ----------------------------------------------------------------------
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the License); you may
# not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an AS IS BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from Deeploy.AbstractDataTypes import PointerClass
from Deeploy.CommonExtensions.DataTypes import int8_t, int32_t, uint8_t
from Deeploy.DeeployTypes import NodeBinding
from Deeploy.MemoryLevelExtension.MemoryLevels import NodeMemoryLevelChecker, memoryAwareNodeBindingExtension
from Deeploy.Targets.Generic.TypeCheckers import ConvChecker
from Deeploy.Targets.Neureka.Templates.ConvTemplate import NeurekaDenseConv2D_Template, NeurekaDWConv2D_Template, \
    NeurekaPWConv2D_Template, NeurekaRqntDenseConv2D_Template, NeurekaRqntDWConv2D_Template, \
    NeurekaRqntPWConv2D_Template
from Deeploy.Targets.PULPOpen.Bindings import ClusterTransformer
from Deeploy.Targets.PULPOpen.TypeCheckers import PULPConvChecker

NeurekaRQSPWConv2DBindings = [
    NodeBinding(
        PULPConvChecker(
            [PointerClass(data_in_type),
             PointerClass(weight_type),
             PointerClass(int32_t),
             PointerClass(int32_t)], [PointerClass(data_out_type)]), NeurekaRqntPWConv2D_Template, ClusterTransformer)
    for data_in_type in [uint8_t, int8_t]
    for data_out_type in [uint8_t, int8_t]
    for weight_type in [uint8_t, int8_t]
]
NeurekaPWConv2DBindings = [
    NodeBinding(
        ConvChecker(
            [PointerClass(data_in_type), PointerClass(weight_type),
             PointerClass(int32_t)], [PointerClass(int32_t)]), NeurekaPWConv2D_Template, ClusterTransformer)
    for data_in_type in [uint8_t, int8_t]
    for weight_type in [uint8_t, int8_t]
]

NeurekaWmemRQSPWConv2DBindings = [
    memoryAwareNodeBindingExtension(binding, NodeMemoryLevelChecker([None, "WeightMemory_SRAM", None, None], [None]))
    for binding in NeurekaRQSPWConv2DBindings
]
NeurekaWmemPWConv2DBindings = [
    memoryAwareNodeBindingExtension(binding, NodeMemoryLevelChecker([None, "WeightMemory_SRAM"], [None]))
    for binding in NeurekaPWConv2DBindings
]

NeurekaRQSDWConv2DBindings = [
    NodeBinding(
        PULPConvChecker(
            [PointerClass(data_in_type),
             PointerClass(weight_type),
             PointerClass(int32_t),
             PointerClass(int32_t)], [PointerClass(data_out_type)]), NeurekaRqntDWConv2D_Template, ClusterTransformer)
    for data_in_type in [uint8_t, int8_t]
    for data_out_type in [uint8_t, int8_t]
    for weight_type in [uint8_t, int8_t]
]
NeurekaDWConv2DBindings = [
    NodeBinding(
        ConvChecker(
            [PointerClass(data_in_type), PointerClass(weight_type),
             PointerClass(int32_t)], [PointerClass(int32_t)]), NeurekaDWConv2D_Template, ClusterTransformer)
    for data_in_type in [uint8_t, int8_t]
    for weight_type in [uint8_t, int8_t]
]

NeurekaWmemRQSDWConv2DBindings = [
    memoryAwareNodeBindingExtension(binding, NodeMemoryLevelChecker([None, "WeightMemory_SRAM", None, None], [None]))
    for binding in NeurekaRQSDWConv2DBindings
]
NeurekaWmemDWConv2DBindings = [
    memoryAwareNodeBindingExtension(binding, NodeMemoryLevelChecker([None, "WeightMemory_SRAM"], [None]))
    for binding in NeurekaDWConv2DBindings
]

NeurekaRQSDenseConv2DBindings = [
    NodeBinding(
        PULPConvChecker(
            [PointerClass(data_in_type),
             PointerClass(weight_type),
             PointerClass(int32_t),
             PointerClass(int32_t)], [PointerClass(data_out_type)]), NeurekaRqntDenseConv2D_Template,
        ClusterTransformer)
    for data_in_type in [uint8_t, int8_t]
    for data_out_type in [uint8_t, int8_t]
    for weight_type in [uint8_t, int8_t]
]
NeurekaDenseConv2DBindings = [
    NodeBinding(
        ConvChecker(
            [PointerClass(data_in_type), PointerClass(weight_type),
             PointerClass(int32_t)], [PointerClass(int32_t)]), NeurekaDenseConv2D_Template, ClusterTransformer)
    for data_in_type in [uint8_t, int8_t]
    for weight_type in [uint8_t, int8_t]
]

NeurekaWmemRQSDenseConv2DBindings = [
    memoryAwareNodeBindingExtension(binding, NodeMemoryLevelChecker([None, "WeightMemory_SRAM", None, None], [None]))
    for binding in NeurekaRQSDenseConv2DBindings
]
NeurekaWmemDenseConv2DBindings = [
    memoryAwareNodeBindingExtension(binding, NodeMemoryLevelChecker([None, "WeightMemory_SRAM"], [None]))
    for binding in NeurekaDenseConv2DBindings
]

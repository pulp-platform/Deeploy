# ----------------------------------------------------------------------
#
# File: Tiler.py
#
# Last edited: 26.07.2024
#
# Copyright (C) 2024, ETH Zurich and University of Bologna.
#
# Author: Moritz Scherer, ETH Zurich
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


from Deeploy.Targets.Neureka.Bindings import NeurekaDenseConv2DBindings, NeurekaDWConv2DBindings, \
    NeurekaPWConv2DBindings, NeurekaRQSDenseConv2DBindings, NeurekaRQSDWConv2DBindings, NeurekaRQSPWConv2DBindings, \
    NeurekaWmemDenseConv2DBindings, NeurekaWmemDWConv2DBindings, NeurekaWmemPWConv2DBindings, \
    NeurekaWmemRQSDenseConv2DBindings, NeurekaWmemRQSDWConv2DBindings, NeurekaWmemRQSPWConv2DBindings
from Deeploy.Targets.Neureka.TileConstraints.NeurekaDenseConstraint import NeurekaDenseConv2DTileConstraint, \
    NeurekaRQSDenseConv2DTileConstraint, NeurekaWmemDenseConv2DTileConstraint, \
    NeurekaWmemRQSDenseConv2DTileConstraint
from Deeploy.Targets.Neureka.TileConstraints.NeurekaDepthwiseConstraint import NeurekaDWConv2DTileConstraint, \
    NeurekaRQSDWConv2DTileConstraint, NeurekaWmemDWConv2DTileConstraint, NeurekaWmemRQSDWConv2DTileConstraint
from Deeploy.Targets.Neureka.TileConstraints.NeurekaPointwiseConstraint import NeurekaPWConv2DTileConstraint, \
    NeurekaRQSPWConv2DTileConstraint, NeurekaWmemPWConv2DTileConstraint, NeurekaWmemRQSPWConv2DTileConstraint
from Deeploy.TilingExtension.TilerExtension import TilingReadyNodeBindings

NeurekaRQSPWConv2DTilingReadyBindings = TilingReadyNodeBindings(nodeBindings = NeurekaRQSPWConv2DBindings,
                                                                tileConstraint = NeurekaRQSPWConv2DTileConstraint())
NeurekaPWConv2DTilingReadyBindings = TilingReadyNodeBindings(nodeBindings = NeurekaPWConv2DBindings,
                                                             tileConstraint = NeurekaPWConv2DTileConstraint())

NeurekaWmemRQSPWConv2DTilingReadyBindings = TilingReadyNodeBindings(
    nodeBindings = NeurekaWmemRQSPWConv2DBindings, tileConstraint = NeurekaWmemRQSPWConv2DTileConstraint())
NeurekaWmemPWConv2DTilingReadyBindings = TilingReadyNodeBindings(nodeBindings = NeurekaWmemPWConv2DBindings,
                                                                 tileConstraint = NeurekaWmemPWConv2DTileConstraint())

NeurekaRQSDWConv2DTilingReadyBindings = TilingReadyNodeBindings(nodeBindings = NeurekaRQSDWConv2DBindings,
                                                                tileConstraint = NeurekaRQSDWConv2DTileConstraint())
NeurekaDWConv2DTilingReadyBindings = TilingReadyNodeBindings(nodeBindings = NeurekaDWConv2DBindings,
                                                             tileConstraint = NeurekaDWConv2DTileConstraint())

NeurekaWmemRQSDWConv2DTilingReadyBindings = TilingReadyNodeBindings(
    nodeBindings = NeurekaWmemRQSDWConv2DBindings, tileConstraint = NeurekaWmemRQSDWConv2DTileConstraint())
NeurekaWmemDWConv2DTilingReadyBindings = TilingReadyNodeBindings(nodeBindings = NeurekaWmemDWConv2DBindings,
                                                                 tileConstraint = NeurekaWmemDWConv2DTileConstraint())

NeurekaRQSDenseConv2DTilingReadyBindings = TilingReadyNodeBindings(
    nodeBindings = NeurekaRQSDenseConv2DBindings, tileConstraint = NeurekaRQSDenseConv2DTileConstraint())
NeurekaDenseConv2DTilingReadyBindings = TilingReadyNodeBindings(nodeBindings = NeurekaDenseConv2DBindings,
                                                                tileConstraint = NeurekaDenseConv2DTileConstraint())

NeurekaWmemRQSDenseConv2DTilingReadyBindings = TilingReadyNodeBindings(
    nodeBindings = NeurekaWmemRQSDenseConv2DBindings, tileConstraint = NeurekaWmemRQSDenseConv2DTileConstraint())
NeurekaWmemDenseConv2DTilingReadyBindings = TilingReadyNodeBindings(
    nodeBindings = NeurekaWmemDenseConv2DBindings, tileConstraint = NeurekaWmemDenseConv2DTileConstraint())

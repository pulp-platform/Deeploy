# SPDX-FileCopyrightText: 2024 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0


from Deeploy.Targets.Neureka.Bindings import NeurekaDenseConv2DBindings, NeurekaDWConv2DBindings, \
    NeurekaPWConv2DBindings, NeurekaRQSDenseConv2DBindings, NeurekaRQSDWConv2DBindings, NeurekaRQSPWConv2DBindings
from Deeploy.Targets.Neureka.TileConstraints.NeurekaDenseConstraint import NeurekaDenseConv2DTileConstraint, \
    NeurekaRQSDenseConv2DTileConstraint
from Deeploy.Targets.Neureka.TileConstraints.NeurekaDepthwiseConstraint import NeurekaDWConv2DTileConstraint, \
    NeurekaRQSDWConv2DTileConstraint
from Deeploy.Targets.Neureka.TileConstraints.NeurekaPointwiseConstraint import NeurekaPWConv2DTileConstraint, \
    NeurekaRQSPWConv2DTileConstraint
from Deeploy.TilingExtension.TilerExtension import TilingReadyNodeBindings

NeurekaRQSPWConv2DTilingReadyBindings = TilingReadyNodeBindings(nodeBindings = NeurekaRQSPWConv2DBindings,
                                                                tileConstraint = NeurekaRQSPWConv2DTileConstraint())
NeurekaPWConv2DTilingReadyBindings = TilingReadyNodeBindings(nodeBindings = NeurekaPWConv2DBindings,
                                                             tileConstraint = NeurekaPWConv2DTileConstraint())

NeurekaRQSDWConv2DTilingReadyBindings = TilingReadyNodeBindings(nodeBindings = NeurekaRQSDWConv2DBindings,
                                                                tileConstraint = NeurekaRQSDWConv2DTileConstraint())
NeurekaDWConv2DTilingReadyBindings = TilingReadyNodeBindings(nodeBindings = NeurekaDWConv2DBindings,
                                                             tileConstraint = NeurekaDWConv2DTileConstraint())

NeurekaRQSDenseConv2DTilingReadyBindings = TilingReadyNodeBindings(
    nodeBindings = NeurekaRQSDenseConv2DBindings, tileConstraint = NeurekaRQSDenseConv2DTileConstraint())
NeurekaDenseConv2DTilingReadyBindings = TilingReadyNodeBindings(nodeBindings = NeurekaDenseConv2DBindings,
                                                                tileConstraint = NeurekaDenseConv2DTileConstraint())

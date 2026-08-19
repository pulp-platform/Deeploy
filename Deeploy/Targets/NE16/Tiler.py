# SPDX-FileCopyrightText: 2024 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0


from Deeploy.Targets.NE16.Bindings import NE16DenseConv2DBindings, NE16DWConv2DBindings, NE16PWConv2DBindings, \
    NE16RQSDenseConv2DBindings, NE16RQSDWConv2DBindings, NE16RQSPWConv2DBindings
from Deeploy.Targets.NE16.TileConstraints.NE16DenseConstraint import NE16DenseConv2DTileConstraint, \
    NE16RQSDenseConv2DTileConstraint
from Deeploy.Targets.NE16.TileConstraints.NE16DepthwiseConstraint import NE16DWConv2DTileConstraint, \
    NE16RQSDWConv2DTileConstraint
from Deeploy.Targets.NE16.TileConstraints.NE16PointwiseConstraint import NE16PWConv2DTileConstraint, \
    NE16RQSPWConv2DTileConstraint
from Deeploy.TilingExtension.TilerExtension import TilingReadyNodeBindings

NE16RQSPWConv2DTilingReadyBindings = TilingReadyNodeBindings(nodeBindings = NE16RQSPWConv2DBindings,
                                                             tileConstraint = NE16RQSPWConv2DTileConstraint())
NE16PWConv2DTilingReadyBindings = TilingReadyNodeBindings(nodeBindings = NE16PWConv2DBindings,
                                                          tileConstraint = NE16PWConv2DTileConstraint())

NE16RQSDWConv2DTilingReadyBindings = TilingReadyNodeBindings(nodeBindings = NE16RQSDWConv2DBindings,
                                                             tileConstraint = NE16RQSDWConv2DTileConstraint())
NE16DWConv2DTilingReadyBindings = TilingReadyNodeBindings(nodeBindings = NE16DWConv2DBindings,
                                                          tileConstraint = NE16DWConv2DTileConstraint())

NE16RQSDenseConv2DTilingReadyBindings = TilingReadyNodeBindings(nodeBindings = NE16RQSDenseConv2DBindings,
                                                                tileConstraint = NE16RQSDenseConv2DTileConstraint())
NE16DenseConv2DTilingReadyBindings = TilingReadyNodeBindings(nodeBindings = NE16DenseConv2DBindings,
                                                             tileConstraint = NE16DenseConv2DTileConstraint())

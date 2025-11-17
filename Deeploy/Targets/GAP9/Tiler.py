# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

"""
GAP9-specific tiler bindings using ClDma instead of MchanDma.

This module creates GAP9-specific tiling ready bindings that use ClDma
instead of the low-level MCHAN API.
"""

import copy

from Deeploy.Targets.Generic.TileConstraints.AddTileConstraint import AddTileConstraint
from Deeploy.Targets.Generic.TileConstraints.ConcatTileConstraint import ConcatTileConstraint
from Deeploy.Targets.Generic.TileConstraints.iHardswishTileConstraint import iHardswishTileConstraint
from Deeploy.Targets.Generic.TileConstraints.iRMSNormTileConstraint import iRMSNormTileConstraint
from Deeploy.Targets.Generic.TileConstraints.MulTileConstraint import MulTileConstraint
from Deeploy.Targets.Generic.TileConstraints.NOPTileConstraint import NOPTileConstraint
from Deeploy.Targets.Generic.TileConstraints.RQSiGELUTileConstraint import RQSiGELUTileConstraint
from Deeploy.Targets.Generic.TileConstraints.RQSiHardswishTileConstraint import RQSiHardswishTileConstraint
from Deeploy.Targets.Generic.TileConstraints.TransposeTileConstraint import TransposeTileConstraint
from Deeploy.Targets.Generic.TileConstraints.UnaryTileConstraint import UnaryTileConstraint
from Deeploy.Targets.Generic.TileConstraints.UntiledTileConstraint import UntiledTileConstraint
from Deeploy.Targets.GAP9.Bindings import (
    GAP9AddBindings,
    GAP9ConcatBindings,
    GAP9FloatConv2DBindings,
    GAP9FloatGELUBinding,
    GAP9FloatGEMMBindings,
    GAP9GatherBindings,
    GAP9iHardswishBindings,
    GAP9iRMSNormBindings,
    GAP9iRQSGELUBindings,
    GAP9LayernormBinding,
    GAP9MatMulBindings,
    GAP9MaxPool2DBindings,
    GAP9MulBindings,
    GAP9ReduceSumBindings,
    GAP9ReluBinding,
    GAP9ReshapeBindings,
    GAP9RQAddBindings,
    GAP9RQSBindings,
    GAP9RQSConv2DBindings,
    GAP9RQSDWConv2DBindings,
    GAP9RQSGEMMBindings,
    GAP9RQSiHardswishBindings,
    GAP9RQSMatrixVecBindings,
    GAP9RQSTallGEMMBindings,
    GAP9SGDBindings,
    GAP9SoftmaxBindings,
    GAP9SoftmaxCrossEntropyLossBindings,
    GAP9SoftmaxCrossEntropyLossGradBindings,
    GAP9SoftmaxGradBindings,
    GAP9TransposeBindings,
    GAP9UniformRQSBindings,
)
from Deeploy.Targets.PULPOpen.TileConstraints.ConvTileConstraint import Conv2DTileConstraint, RQConv2DTileConstraint
from Deeploy.Targets.PULPOpen.TileConstraints.DWConvTileConstraint import DWConv2DTileConstraint
from Deeploy.Targets.PULPOpen.TileConstraints.GatherTileConstraint import GatherTileConstraint
from Deeploy.Targets.PULPOpen.TileConstraints.GEMMTileConstraint import FloatGEMMTileConstraint, GEMMTileConstraint
from Deeploy.Targets.PULPOpen.TileConstraints.iSoftmaxTileConstraint import iSoftmaxTileConstraint
from Deeploy.Targets.PULPOpen.TileConstraints.LayernormTileConstraint import LayernormTileConstraint
from Deeploy.Targets.PULPOpen.TileConstraints.MatMulTileConstraint import MatMulTileConstraint
from Deeploy.Targets.PULPOpen.TileConstraints.MaxPoolTileConstraint import MaxPoolCTileConstraint
from Deeploy.Targets.PULPOpen.TileConstraints.RequantShiftTileConstraint import RequantShiftTileConstraint
from Deeploy.Targets.PULPOpen.TileConstraints.SGDTileConstraint import SGDTileConstraint
from Deeploy.Targets.PULPOpen.TileConstraints.SoftmaxCrossEntropyTileConstraint import \
    SoftmaxCrossEntropyGradTileConstraint, SoftmaxCrossEntropyTileConstraint
from Deeploy.TilingExtension.TilerExtension import TilingReadyNodeBindings

# GAP9-specific tiling ready bindings using ClDma
GAP9RQSConv2DTilingReadyBindings = TilingReadyNodeBindings(nodeBindings = GAP9RQSConv2DBindings,
                                                           tileConstraint = RQConv2DTileConstraint())

GAP9RQSDWConv2DTilingReadyBindings = TilingReadyNodeBindings(nodeBindings = GAP9RQSDWConv2DBindings,
                                                             tileConstraint = DWConv2DTileConstraint())

GAP9Conv2DTilingReadyBindings = TilingReadyNodeBindings(nodeBindings = GAP9FloatConv2DBindings,
                                                        tileConstraint = Conv2DTileConstraint())

GAP9RQSGEMMTilingReadyBindings = TilingReadyNodeBindings(nodeBindings = GAP9RQSGEMMBindings,
                                                         tileConstraint = GEMMTileConstraint())

GAP9FPGEMMTilingReadyBindings = TilingReadyNodeBindings(nodeBindings = GAP9FloatGEMMBindings,
                                                        tileConstraint = FloatGEMMTileConstraint())

GAP9RQSMatrixVecTilingReadyBindings = TilingReadyNodeBindings(nodeBindings = GAP9RQSMatrixVecBindings,
                                                              tileConstraint = GEMMTileConstraint())

GAP9RQSTallGEMMTilingReadyBindings = TilingReadyNodeBindings(nodeBindings = GAP9RQSTallGEMMBindings,
                                                             tileConstraint = GEMMTileConstraint())

GAP9MatMulTilingReadyBindings = TilingReadyNodeBindings(nodeBindings = GAP9MatMulBindings,
                                                        tileConstraint = MatMulTileConstraint())

GAP9RQAddTilingReadyBindings = TilingReadyNodeBindings(nodeBindings = GAP9RQAddBindings,
                                                       tileConstraint = AddTileConstraint())

GAP9iHardswishTilingReadyBindings = TilingReadyNodeBindings(nodeBindings = GAP9iHardswishBindings,
                                                            tileConstraint = iHardswishTileConstraint())

GAP9RQSiHardswishTilingReadyBindings = TilingReadyNodeBindings(nodeBindings = GAP9RQSiHardswishBindings,
                                                               tileConstraint = RQSiHardswishTileConstraint())

_GAP9FlattenBindings = copy.deepcopy(GAP9ReshapeBindings)

GAP9FlattenTilingReadyBindings = TilingReadyNodeBindings(nodeBindings = _GAP9FlattenBindings,
                                                         tileConstraint = NOPTileConstraint())

GAP9MaxPool2DTilingReadyBindings = TilingReadyNodeBindings(nodeBindings = GAP9MaxPool2DBindings,
                                                           tileConstraint = MaxPoolCTileConstraint())

GAP9RQSTilingReadyBindings = TilingReadyNodeBindings(nodeBindings = GAP9RQSBindings,
                                                     tileConstraint = RequantShiftTileConstraint())

GAP9UniformRQSTilingReadyBindings = TilingReadyNodeBindings(nodeBindings = GAP9UniformRQSBindings,
                                                            tileConstraint = UnaryTileConstraint())

GAP9TransposeTilingReadyBindings = TilingReadyNodeBindings(nodeBindings = GAP9TransposeBindings,
                                                           tileConstraint = TransposeTileConstraint())

GAP9AddTilingReadyBindings = TilingReadyNodeBindings(nodeBindings = GAP9AddBindings,
                                                     tileConstraint = AddTileConstraint())

GAP9SoftmaxTilingReadyBindings = TilingReadyNodeBindings(nodeBindings = GAP9SoftmaxBindings,
                                                         tileConstraint = iSoftmaxTileConstraint())

GAP9ConcatTilingReadyBindings = TilingReadyNodeBindings(nodeBindings = GAP9ConcatBindings,
                                                        tileConstraint = ConcatTileConstraint())

GAP9iRMSNormTilingReadyBindings = TilingReadyNodeBindings(nodeBindings = GAP9iRMSNormBindings,
                                                          tileConstraint = iRMSNormTileConstraint())

GAP9iRQSGELUTilingReadyBindings = TilingReadyNodeBindings(nodeBindings = GAP9iRQSGELUBindings,
                                                          tileConstraint = RQSiGELUTileConstraint())

GAP9MulTilingReadyBindings = TilingReadyNodeBindings(nodeBindings = GAP9MulBindings,
                                                     tileConstraint = MulTileConstraint())

GAP9ReluTilingReadyBindings = TilingReadyNodeBindings(nodeBindings = [GAP9ReluBinding],
                                                      tileConstraint = UnaryTileConstraint())

GAP9LayernormTilingReadyBindings = TilingReadyNodeBindings(nodeBindings = [GAP9LayernormBinding],
                                                           tileConstraint = LayernormTileConstraint())

GAP9FPGELUTilingReadyBindings = TilingReadyNodeBindings(nodeBindings = [GAP9FloatGELUBinding],
                                                        tileConstraint = UnaryTileConstraint())

GAP9GatherTilingReadyBindings = TilingReadyNodeBindings(nodeBindings = GAP9GatherBindings,
                                                        tileConstraint = GatherTileConstraint())

GAP9SoftmaxCrossEntropyTilingReadyBindings = TilingReadyNodeBindings(
    nodeBindings = GAP9SoftmaxCrossEntropyLossBindings, tileConstraint = SoftmaxCrossEntropyTileConstraint())

GAP9SoftmaxCrossEntropyGradTilingReadyBindings = TilingReadyNodeBindings(
    nodeBindings = GAP9SoftmaxCrossEntropyLossGradBindings, tileConstraint = SoftmaxCrossEntropyGradTileConstraint())

GAP9SoftmaxGradTilingReadyBindings = TilingReadyNodeBindings(nodeBindings = GAP9SoftmaxGradBindings,
                                                             tileConstraint = UntiledTileConstraint())

GAP9ReduceSumTilingReadyBindings = TilingReadyNodeBindings(nodeBindings = GAP9ReduceSumBindings,
                                                           tileConstraint = UntiledTileConstraint())

GAP9SGDTilingReadyBindings = TilingReadyNodeBindings(nodeBindings = GAP9SGDBindings,
                                                     tileConstraint = SGDTileConstraint())

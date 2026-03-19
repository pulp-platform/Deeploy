# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0
"""
GAP9-specific tiler bindings using ClDma instead of MchanDma.

This module creates GAP9-specific tiling ready bindings that use ClDma
instead of the low-level MCHAN API.
"""

import copy

from Deeploy.Targets.GAP9.Bindings import GAP9AddBindings, GAP9AveragePool2DBindings, \
    GAP9AveragePoolGrad2DBindings, GAP9BatchNormInternalBindings, GAP9BatchNormalizationGradBindings, \
    GAP9ConcatBindings, GAP9FloatConv2DBindings, GAP9FloatConvGradBBindings, GAP9FloatConvGradW2DBindings, \
    GAP9FloatConvGradX2DBindings, GAP9FloatDWConv2DBindings, GAP9FloatDWConvGradW2DBindings, \
    GAP9FloatDWConvGradX2DBindings, GAP9FloatGELUBinding, GAP9FloatGELUGradBinding, GAP9FloatGEMMBindings, \
    GAP9FloatPWConvGradW2DBindings, GAP9FloatPWConvGradX2DBindings, GAP9GatherBindings, \
    GAP9GlobalAveragePool2DBindings, GAP9GlobalAveragePoolGrad2DBindings, \
    GAP9iHardswishBindings, GAP9InPlaceAccumulatorV2TiledBindings, GAP9iRMSNormBindings, GAP9iRQSGELUBindings, \
    GAP9LayernormBinding, GAP9LayernormGradBinding, GAP9LayernormInferenceBinding, \
    GAP9MatMulBindings, GAP9MaxPool2DBindings, \
    GAP9MaxPoolGrad2DBindings, GAP9MSELossBindings, GAP9MSELossGradBindings, \
    GAP9MulBindings, GAP9ReduceSumBindings, GAP9ReluBinding, GAP9ReluGradBinding, GAP9ReshapeBindings, \
    GAP9RQAddBindings, GAP9RQSBindings, GAP9RQSConv2DBindings, GAP9RQSDWConv2DBindings, GAP9RQSGEMMBindings, \
    GAP9RQSiHardswishBindings, GAP9RQSMatrixVecBindings, GAP9RQSTallGEMMBindings, GAP9SGDBindings, \
    GAP9SoftmaxBindings, GAP9SoftmaxCrossEntropyLossBindings, GAP9SoftmaxCrossEntropyLossDualOutputBindings, \
    GAP9SliceBindings, GAP9SoftmaxCrossEntropyLossGradBindings, \
    GAP9SoftmaxGradBindings, GAP9TransposeBindings, GAP9UniformRQSBindings
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
from Deeploy.Targets.PULPOpen.TileConstraints.ConvTileConstraint import Conv2DTileConstraint, RQConv2DTileConstraint
from Deeploy.Targets.PULPOpen.TileConstraints.DWConvTileConstraint import DWConv2DTileConstraint, \
    RQDWConv2DTileConstraint
from Deeploy.Targets.PULPOpen.TileConstraints.AveragePoolTileConstraint import AveragePoolCTileConstraint
from Deeploy.Targets.PULPOpen.TileConstraints.BatchNormTileConstraint import BatchNormInternalTileConstraint, \
    BatchNormalizationGradTileConstraint
from Deeploy.Targets.PULPOpen.TileConstraints.ConvGradConstraint import ConvGradBTileConstraint, \
    ConvGradX2DIm2ColHWTileConstraint, ConvGradW2DTileConstraint, \
    DWConvGradX2DTileConstraint, DWConvGradW2DTileConstraint, PWConvGradXTileConstraint, PWConvGradWTileConstraint
from Deeploy.Targets.PULPOpen.TileConstraints.GatherTileConstraint import GatherTileConstraint
from Deeploy.Targets.PULPOpen.TileConstraints.GEMMTileConstraint import FloatGEMMTileConstraint, GEMMTileConstraint
from Deeploy.Targets.PULPOpen.TileConstraints.GeluTileConstraint import GeluGradTileConstraint
from Deeploy.Targets.PULPOpen.TileConstraints.GlobalAveragePoolTileConstraint import GlobalAveragePoolTileConstraint, \
    GlobalAveragePoolGradTileConstraint
from Deeploy.Targets.PULPOpen.TileConstraints.InPlaceAccumulatorV2TileConstraint import \
    InPlaceAccumulatorV2TileConstraint
from Deeploy.Targets.PULPOpen.TileConstraints.iSoftmaxTileConstraint import SoftmaxGradTileConstraint, \
    iSoftmaxTileConstraint
from Deeploy.Targets.PULPOpen.TileConstraints.ReduceSumTileConstraint import ReduceSumTileConstraint
from Deeploy.Targets.PULPOpen.TileConstraints.LayernormTileConstraint import LayernormGradTileConstraint, \
    LayernormTileConstraint
from Deeploy.Targets.PULPOpen.TileConstraints.MatMulTileConstraint import MatMulTileConstraint
from Deeploy.Targets.PULPOpen.TileConstraints.MaxPoolTileConstraint import MaxPoolCTileConstraint
from Deeploy.Targets.PULPOpen.TileConstraints.MaxPoolGradTileConstraint import MaxPoolGradCTileConstraint
from Deeploy.Targets.PULPOpen.TileConstraints.MSELossTileConstraint import MSELossTileConstraint, \
    MSELossGradTileConstraint
from Deeploy.Targets.PULPOpen.TileConstraints.RequantShiftTileConstraint import RequantShiftTileConstraint
from Deeploy.Targets.PULPOpen.TileConstraints.SGDTileConstraint import ReluGradTileConstraint, SGDTileConstraint
from Deeploy.Targets.PULPOpen.TileConstraints.SliceConstraint import SliceTileConstraint
from Deeploy.Targets.PULPOpen.TileConstraints.SoftmaxCrossEntropyTileConstraint import \
    SoftmaxCrossEntropyGradTileConstraint, SoftmaxCrossEntropyTileConstraint
from Deeploy.Targets.PULPOpen.TileConstraints.SoftmaxCrossEntropyLossDualOutputTileConstraint import \
    SoftmaxCrossEntropyLossDualOutputTileConstraint
from Deeploy.TilingExtension.TilerExtension import TilingReadyNodeBindings

# GAP9-specific tiling ready bindings using ClDma
GAP9RQSConv2DTilingReadyBindings = TilingReadyNodeBindings(nodeBindings = GAP9RQSConv2DBindings,
                                                           tileConstraint = RQConv2DTileConstraint())

GAP9RQSDWConv2DTilingReadyBindings = TilingReadyNodeBindings(nodeBindings = GAP9RQSDWConv2DBindings,
                                                             tileConstraint = RQDWConv2DTileConstraint())

GAP9Conv2DTilingReadyBindings = TilingReadyNodeBindings(nodeBindings = GAP9FloatConv2DBindings,
                                                        tileConstraint = Conv2DTileConstraint())

GAP9DWConv2DTilingReadyBindings = TilingReadyNodeBindings(nodeBindings = GAP9FloatDWConv2DBindings,
                                                          tileConstraint = DWConv2DTileConstraint())

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

# Training LayerNorm (3 outputs: Y, mean, inv_std) tried first, then inference (1 output)
GAP9LayernormTilingReadyBindings = TilingReadyNodeBindings(
    nodeBindings = [GAP9LayernormBinding, GAP9LayernormInferenceBinding],
    tileConstraint = LayernormTileConstraint())

GAP9FPGELUTilingReadyBindings = TilingReadyNodeBindings(nodeBindings = [GAP9FloatGELUBinding],
                                                        tileConstraint = UnaryTileConstraint())

GAP9GatherTilingReadyBindings = TilingReadyNodeBindings(nodeBindings = GAP9GatherBindings,
                                                        tileConstraint = GatherTileConstraint())

GAP9SoftmaxCrossEntropyTilingReadyBindings = TilingReadyNodeBindings(
    nodeBindings = GAP9SoftmaxCrossEntropyLossBindings, tileConstraint = SoftmaxCrossEntropyTileConstraint())

GAP9SoftmaxCrossEntropyLossDualOutputTilingReadyBindings = TilingReadyNodeBindings(
    nodeBindings = GAP9SoftmaxCrossEntropyLossDualOutputBindings, tileConstraint = SoftmaxCrossEntropyLossDualOutputTileConstraint())

GAP9SoftmaxCrossEntropyGradTilingReadyBindings = TilingReadyNodeBindings(
    nodeBindings = GAP9SoftmaxCrossEntropyLossGradBindings, tileConstraint = SoftmaxCrossEntropyGradTileConstraint())

GAP9SoftmaxGradTilingReadyBindings = TilingReadyNodeBindings(nodeBindings = GAP9SoftmaxGradBindings,
                                                             tileConstraint = SoftmaxGradTileConstraint())

GAP9ReduceSumTilingReadyBindings = TilingReadyNodeBindings(nodeBindings = GAP9ReduceSumBindings,
                                                           tileConstraint = ReduceSumTileConstraint())

GAP9SGDTilingReadyBindings = TilingReadyNodeBindings(nodeBindings = GAP9SGDBindings,
                                                     tileConstraint = SGDTileConstraint())

# ── Training / Gradient tiling ready bindings ────────────────────────────

GAP9ReluGradTilingReadyBindings = TilingReadyNodeBindings(nodeBindings = [GAP9ReluGradBinding],
                                                           tileConstraint = ReluGradTileConstraint())

GAP9FPGELUGradTilingReadyBindings = TilingReadyNodeBindings(nodeBindings = [GAP9FloatGELUGradBinding],
                                                            tileConstraint = GeluGradTileConstraint())

GAP9LayernormGradTilingReadyBindings = TilingReadyNodeBindings(nodeBindings = [GAP9LayernormGradBinding],
                                                               tileConstraint = LayernormGradTileConstraint())

GAP9ConvGradX2DTilingReadyBindings = TilingReadyNodeBindings(nodeBindings = GAP9FloatConvGradX2DBindings,
                                                            tileConstraint = ConvGradX2DIm2ColHWTileConstraint())

GAP9ConvGradW2DTilingReadyBindings = TilingReadyNodeBindings(nodeBindings = GAP9FloatConvGradW2DBindings,
                                                             tileConstraint = ConvGradW2DTileConstraint())

GAP9DWConvGradX2DTilingReadyBindings = TilingReadyNodeBindings(nodeBindings = GAP9FloatDWConvGradX2DBindings,
                                                              tileConstraint = DWConvGradX2DTileConstraint())

GAP9DWConvGradW2DTilingReadyBindings = TilingReadyNodeBindings(nodeBindings = GAP9FloatDWConvGradW2DBindings,
                                                               tileConstraint = DWConvGradW2DTileConstraint())

GAP9PWConvGradW2DTilingReadyBindings = TilingReadyNodeBindings(nodeBindings = GAP9FloatPWConvGradW2DBindings,
                                                                tileConstraint = PWConvGradWTileConstraint())

GAP9PWConvGradX2DTilingReadyBindings = TilingReadyNodeBindings(nodeBindings = GAP9FloatPWConvGradX2DBindings,
                                                                tileConstraint = PWConvGradXTileConstraint())

GAP9ConvGradBTilingReadyBindings = TilingReadyNodeBindings(
    nodeBindings = GAP9FloatConvGradBBindings, tileConstraint = ConvGradBTileConstraint())

GAP9MaxPoolGrad2DTilingReadyBindings = TilingReadyNodeBindings(nodeBindings = GAP9MaxPoolGrad2DBindings,
                                                               tileConstraint = MaxPoolGradCTileConstraint())

GAP9AveragePool2DTilingReadyBindings = TilingReadyNodeBindings(nodeBindings = GAP9AveragePool2DBindings,
                                                                tileConstraint = AveragePoolCTileConstraint())

GAP9AveragePoolGrad2DTilingReadyBindings = TilingReadyNodeBindings(nodeBindings = GAP9AveragePoolGrad2DBindings,
                                                                     tileConstraint = AveragePoolCTileConstraint())

GAP9GlobalAveragePool2DTilingReadyBindings = TilingReadyNodeBindings(
    nodeBindings = GAP9GlobalAveragePool2DBindings, tileConstraint = GlobalAveragePoolTileConstraint())

GAP9GlobalAveragePoolGrad2DTilingReadyBindings = TilingReadyNodeBindings(
    nodeBindings = GAP9GlobalAveragePoolGrad2DBindings, tileConstraint = GlobalAveragePoolGradTileConstraint())

GAP9MSELossTilingReadyBindings = TilingReadyNodeBindings(
    nodeBindings = GAP9MSELossBindings, tileConstraint = MSELossTileConstraint())

GAP9MSELossGradTilingReadyBindings = TilingReadyNodeBindings(
    nodeBindings = GAP9MSELossGradBindings, tileConstraint = MSELossGradTileConstraint())

GAP9InPlaceAccumulatorV2TilingReadyBindings = TilingReadyNodeBindings(
    nodeBindings = GAP9InPlaceAccumulatorV2TiledBindings, tileConstraint = InPlaceAccumulatorV2TileConstraint())

GAP9BatchNormInternalTilingReadyBindings = TilingReadyNodeBindings(
    nodeBindings = GAP9BatchNormInternalBindings, tileConstraint = BatchNormInternalTileConstraint())

GAP9BatchNormalizationGradTilingReadyBindings = TilingReadyNodeBindings(
    nodeBindings = GAP9BatchNormalizationGradBindings, tileConstraint = BatchNormalizationGradTileConstraint())


GAP9SliceTilingReadyBindings = TilingReadyNodeBindings(
    nodeBindings = GAP9SliceBindings, tileConstraint = SliceTileConstraint())

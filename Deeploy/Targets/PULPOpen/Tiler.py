# SPDX-FileCopyrightText: 2023 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

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
from Deeploy.Targets.PULPOpen.Bindings import PULPAddBindings, PULPConcatBindings, PULPFloatConv2DBindings, \
    PULPFloatDWConv2DBindings, PULPFloatGELUBinding, PULPFloatGELUGradBinding, PULPFloatGEMMBindings, \
    PULPGatherBindings, PULPiHardswishBindings, PULPiRMSNormBindings, PULPiRQSGELUBindings, PULPLayernormBinding, \
    PULPLayernormGradBinding, PULPMatMulBindings, PULPMaxPool1DBindings, PULPMaxPool2DBindings, PULPMulBindings, \
    PULPReduceMeanBindings, PULPReduceSumBindings, PULPReluBinding, PULPReshapeBindings, PULPRQAddBindings, \
    PULPRQSBindings, PULPRQSConv2DBindings, PULPRQSDWConv2DBindings, PULPRQSGEMMBindings, PULPRQSiHardswishBindings, \
    PULPRQSMatrixVecBindings, PULPRQSTallGEMMBindings, PULPSGDBindings, PULPSliceBindings, PULPSoftmaxBindings, \
    PULPSoftmaxCrossEntropyLossBindings, PULPSoftmaxCrossEntropyLossGradBindings, PULPSoftmaxGradBindings, \
    PULPTransposeBindings, PULPUniformRQSBindings
from Deeploy.Targets.PULPOpen.TileConstraints.ConvTileConstraint import Conv2DTileConstraint, RQConv2DTileConstraint
from Deeploy.Targets.PULPOpen.TileConstraints.DWConvTileConstraint import DWConv2DTileConstraint, \
    RQDWConv2DTileConstraint
from Deeploy.Targets.PULPOpen.TileConstraints.GatherTileConstraint import GatherTileConstraint
from Deeploy.Targets.PULPOpen.TileConstraints.GeluTileConstraint import GeluGradTileConstraint
from Deeploy.Targets.PULPOpen.TileConstraints.GEMMTileConstraint import FloatGEMMTileConstraint, GEMMTileConstraint
from Deeploy.Targets.PULPOpen.TileConstraints.iSoftmaxTileConstraint import SoftmaxGradTileConstraint, \
    iSoftmaxTileConstraint
from Deeploy.Targets.PULPOpen.TileConstraints.LayernormTileConstraint import LayernormGradTileConstraint, \
    LayernormTileConstraint
from Deeploy.Targets.PULPOpen.TileConstraints.MatMulTileConstraint import MatMulTileConstraint
from Deeploy.Targets.PULPOpen.TileConstraints.MaxPoolTileConstraint import MaxPoolCTileConstraint
from Deeploy.Targets.PULPOpen.TileConstraints.ReduceMeanConstraint import ReduceMeanTileConstraint
from Deeploy.Targets.PULPOpen.TileConstraints.ReduceSumTileConstraint import ReduceSumTileConstraint
from Deeploy.Targets.PULPOpen.TileConstraints.RequantShiftTileConstraint import RequantShiftTileConstraint
from Deeploy.Targets.PULPOpen.TileConstraints.SGDTileConstraint import SGDTileConstraint
from Deeploy.Targets.PULPOpen.TileConstraints.SliceConstraint import SliceTileConstraint
from Deeploy.Targets.PULPOpen.TileConstraints.SoftmaxCrossEntropyTileConstraint import \
    SoftmaxCrossEntropyGradTileConstraint, SoftmaxCrossEntropyTileConstraint
from Deeploy.TilingExtension.TilerExtension import TilingReadyNodeBindings

PULPRQSConv2DTilingReadyBindings = TilingReadyNodeBindings(nodeBindings = PULPRQSConv2DBindings,
                                                           tileConstraint = RQConv2DTileConstraint())

PULPRQSDWConv2DTilingReadyBindings = TilingReadyNodeBindings(nodeBindings = PULPRQSDWConv2DBindings,
                                                             tileConstraint = RQDWConv2DTileConstraint())

PULPConv2DTilingReadyBindings = TilingReadyNodeBindings(nodeBindings = PULPFloatConv2DBindings,
                                                        tileConstraint = Conv2DTileConstraint())

PULPDWConv2DTilingReadyBindings = TilingReadyNodeBindings(nodeBindings = PULPFloatDWConv2DBindings,
                                                          tileConstraint = DWConv2DTileConstraint())

PULPRQSGEMMTilingReadyBindings = TilingReadyNodeBindings(nodeBindings = PULPRQSGEMMBindings,
                                                         tileConstraint = GEMMTileConstraint())

PULPFPGEMMTilingReadyBindings = TilingReadyNodeBindings(nodeBindings = PULPFloatGEMMBindings,
                                                        tileConstraint = FloatGEMMTileConstraint())

PULPRQSMatrixVecTilingReadyBindings = TilingReadyNodeBindings(nodeBindings = PULPRQSMatrixVecBindings,
                                                              tileConstraint = GEMMTileConstraint())

PULPRQSTallGEMMTilingReadyBindings = TilingReadyNodeBindings(nodeBindings = PULPRQSTallGEMMBindings,
                                                             tileConstraint = GEMMTileConstraint())

PULPMatMulTilingReadyBindings = TilingReadyNodeBindings(nodeBindings = PULPMatMulBindings,
                                                        tileConstraint = MatMulTileConstraint())

PULPRQAddTilingReadyBindings = TilingReadyNodeBindings(nodeBindings = PULPRQAddBindings,
                                                       tileConstraint = AddTileConstraint())

PULPiHardswishTilingReadyBindings = TilingReadyNodeBindings(nodeBindings = PULPiHardswishBindings,
                                                            tileConstraint = iHardswishTileConstraint())

PULPRQSiHardswishTilingReadyBindings = TilingReadyNodeBindings(nodeBindings = PULPRQSiHardswishBindings,
                                                               tileConstraint = RQSiHardswishTileConstraint())

_BasicFlattenBindings = copy.deepcopy(PULPReshapeBindings)

PULPFlattenTilingReadyBindings = TilingReadyNodeBindings(nodeBindings = _BasicFlattenBindings,
                                                         tileConstraint = NOPTileConstraint())

PULPMaxPool1DTilingReadyBindings = TilingReadyNodeBindings(nodeBindings = PULPMaxPool1DBindings,
                                                           tileConstraint = MaxPoolCTileConstraint())

PULPMaxPool2DTilingReadyBindings = TilingReadyNodeBindings(nodeBindings = PULPMaxPool2DBindings,
                                                           tileConstraint = MaxPoolCTileConstraint())

PULPRQSTilingReadyBindings = TilingReadyNodeBindings(nodeBindings = PULPRQSBindings,
                                                     tileConstraint = RequantShiftTileConstraint())

PULPUniformRQSTilingReadyBindings = TilingReadyNodeBindings(nodeBindings = PULPUniformRQSBindings,
                                                            tileConstraint = UnaryTileConstraint())

PULPTransposeTilingReadyBindings = TilingReadyNodeBindings(nodeBindings = PULPTransposeBindings,
                                                           tileConstraint = TransposeTileConstraint())

PULPAddTilingReadyBindings = TilingReadyNodeBindings(nodeBindings = PULPAddBindings,
                                                     tileConstraint = AddTileConstraint())

PULPSoftmaxTilingReadyBindings = TilingReadyNodeBindings(nodeBindings = PULPSoftmaxBindings,
                                                         tileConstraint = iSoftmaxTileConstraint())

PULPConcatTilingReadyBindings = TilingReadyNodeBindings(nodeBindings = PULPConcatBindings,
                                                        tileConstraint = ConcatTileConstraint())

PULPiRMSNormTilingReadyBindings = TilingReadyNodeBindings(nodeBindings = PULPiRMSNormBindings,
                                                          tileConstraint = iRMSNormTileConstraint())

PULPiRQSGELUTilingReadyBindings = TilingReadyNodeBindings(nodeBindings = PULPiRQSGELUBindings,
                                                          tileConstraint = RQSiGELUTileConstraint())

PULPMulTilingReadyBindings = TilingReadyNodeBindings(nodeBindings = PULPMulBindings,
                                                     tileConstraint = MulTileConstraint())

PULPReluTilingReadyBindings = TilingReadyNodeBindings(nodeBindings = [PULPReluBinding],
                                                      tileConstraint = UnaryTileConstraint())

PULPLayernormTilingReadyBindings = TilingReadyNodeBindings(nodeBindings = [PULPLayernormBinding],
                                                           tileConstraint = LayernormTileConstraint())

PULPLayernormGradTilingReadyBindings = TilingReadyNodeBindings(nodeBindings = [PULPLayernormGradBinding],
                                                               tileConstraint = LayernormGradTileConstraint())

PULPFPGELUTilingReadyBindings = TilingReadyNodeBindings(nodeBindings = [PULPFloatGELUBinding],
                                                        tileConstraint = UnaryTileConstraint())

PULPFPGELUGradTilingReadyBindings = TilingReadyNodeBindings(nodeBindings = [PULPFloatGELUGradBinding],
                                                            tileConstraint = GeluGradTileConstraint())

PULPGatherTilingReadyBindings = TilingReadyNodeBindings(nodeBindings = PULPGatherBindings,
                                                        tileConstraint = GatherTileConstraint())

PULPSoftmaxCrossEntropyTilingReadyBindings = TilingReadyNodeBindings(
    nodeBindings = PULPSoftmaxCrossEntropyLossBindings, tileConstraint = SoftmaxCrossEntropyTileConstraint())

PULPSoftmaxCrossEntropyGradTilingReadyBindings = TilingReadyNodeBindings(
    nodeBindings = PULPSoftmaxCrossEntropyLossGradBindings, tileConstraint = SoftmaxCrossEntropyGradTileConstraint())

PULPSoftmaxGradTilingReadyBindings = TilingReadyNodeBindings(nodeBindings = PULPSoftmaxGradBindings,
                                                             tileConstraint = SoftmaxGradTileConstraint())

PULPReduceSumTilingReadyBindings = TilingReadyNodeBindings(nodeBindings = PULPReduceSumBindings,
                                                           tileConstraint = ReduceSumTileConstraint())

PULPSGDTilingReadyBindings = TilingReadyNodeBindings(nodeBindings = PULPSGDBindings,
                                                     tileConstraint = SGDTileConstraint())

PULPSliceTilingReadyBindings = TilingReadyNodeBindings(nodeBindings = PULPSliceBindings,
                                                       tileConstraint = SliceTileConstraint())

PULPReduceMeanTilingReadyBindings = TilingReadyNodeBindings(nodeBindings = PULPReduceMeanBindings,
                                                            tileConstraint = ReduceMeanTileConstraint())

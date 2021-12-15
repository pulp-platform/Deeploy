# ----------------------------------------------------------------------
#
# File: PULPTiler.py
#
# Last edited: 09.05.2023
#
# Copyright (C) 2023, ETH Zurich and University of Bologna.
#
# Author:
# - Moritz Scherer, scheremo@iis.ee.ethz.ch, ETH Zurich
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

import copy

from Deeploy.CommonExtensions.CodeTransformationPasses.MemoryAllocation import MemoryPassthroughGeneration
from Deeploy.DeeployTypes import CodeTransformation
from Deeploy.Targets.Generic.Bindings import BasicAddBindings, BasicReshapeBindings
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
from Deeploy.Targets.PULPOpen.Bindings import PULPConcatBindings, PULPiHardswishBindings, PULPiRMSNormBindings, \
    PULPiRQSGELUBindings, PULPMatMulBinding, PULPMaxPool2DBindings, PULPMulBindings, PULPRQAddBindings, \
    PULPRQSBindings, PULPRQSConv2DBindings, PULPRQSDWConv2DBindings, PULPRQSGEMMBindings, PULPRQSiHardswishBindings, \
    PULPRQSMatrixVecBindings, PULPRQSTallGEMMBindings, PULPSoftmaxBindings, PULPTransposeBindings, \
    PULPUniformRQSBindings, SimpleTransformer
from Deeploy.Targets.PULPOpen.TileConstraints.ConvTileConstraint import Conv2DTileConstraint
from Deeploy.Targets.PULPOpen.TileConstraints.DWConvTileConstraint import DWConv2DTileConstraint
from Deeploy.Targets.PULPOpen.TileConstraints.GEMMTileConstraint import GEMMTileConstraint, MatrixVecTileConstraint, \
    TallGEMMTileConstraint
from Deeploy.Targets.PULPOpen.TileConstraints.iSoftmaxTileConstraint import iSoftmaxTileConstraint
from Deeploy.Targets.PULPOpen.TileConstraints.MatMulTileConstraint import MatMulTileConstraint
from Deeploy.Targets.PULPOpen.TileConstraints.MaxPoolTileConstraint import MaxPoolTileConstraint
from Deeploy.Targets.PULPOpen.TileConstraints.RequantShiftTileConstraint import RequantShiftTileConstraint
from Deeploy.TilingExtension.TilerExtension import TilingReadyNodeBindings

PULPRQSConv2DTilingReadyBindings = TilingReadyNodeBindings(nodeBindings = PULPRQSConv2DBindings,
                                                           tileConstraint = Conv2DTileConstraint())

PULPRQSDWConv2DTilingReadyBindings = TilingReadyNodeBindings(nodeBindings = PULPRQSDWConv2DBindings,
                                                             tileConstraint = DWConv2DTileConstraint())

PULPRQSGEMMTilingReadyBindings = TilingReadyNodeBindings(nodeBindings = PULPRQSGEMMBindings,
                                                         tileConstraint = GEMMTileConstraint())

PULPRQSMatrixVecTilingReadyBindings = TilingReadyNodeBindings(nodeBindings = PULPRQSMatrixVecBindings,
                                                              tileConstraint = MatrixVecTileConstraint())

PULPRQSTallGEMMTilingReadyBindings = TilingReadyNodeBindings(nodeBindings = PULPRQSTallGEMMBindings,
                                                             tileConstraint = TallGEMMTileConstraint())

PULPMatMulTilingReadyBindings = TilingReadyNodeBindings(nodeBindings = [PULPMatMulBinding],
                                                        tileConstraint = MatMulTileConstraint())

PULPRQAddTilingReadyBindings = TilingReadyNodeBindings(nodeBindings = PULPRQAddBindings,
                                                       tileConstraint = AddTileConstraint())

PULPiHardswishTilingReadyBindings = TilingReadyNodeBindings(nodeBindings = PULPiHardswishBindings,
                                                            tileConstraint = iHardswishTileConstraint())

PULPRQSiHardswishTilingReadyBindings = TilingReadyNodeBindings(nodeBindings = PULPRQSiHardswishBindings,
                                                               tileConstraint = RQSiHardswishTileConstraint())

_BasicFlattenBindings = copy.deepcopy(BasicReshapeBindings)
for binding in _BasicFlattenBindings:
    binding.codeTransformer = CodeTransformation([MemoryPassthroughGeneration("L.*"), MemoryPassthroughGeneration()])

PULPFlattenTilingReadyBindings = TilingReadyNodeBindings(nodeBindings = _BasicFlattenBindings,
                                                         tileConstraint = NOPTileConstraint())

PULPMaxPool2DTilingReadyBindings = TilingReadyNodeBindings(nodeBindings = PULPMaxPool2DBindings,
                                                           tileConstraint = MaxPoolTileConstraint())

PULPRQSTilingReadyBindings = TilingReadyNodeBindings(nodeBindings = PULPRQSBindings,
                                                     tileConstraint = RequantShiftTileConstraint())

PULPUniformRQSTilingReadyBindings = TilingReadyNodeBindings(nodeBindings = PULPUniformRQSBindings,
                                                            tileConstraint = UnaryTileConstraint())

PULPTransposeTilingReadyBindings = TilingReadyNodeBindings(nodeBindings = PULPTransposeBindings,
                                                           tileConstraint = TransposeTileConstraint())

_PULPAddBindings = copy.deepcopy(BasicAddBindings)
for binding in _PULPAddBindings:
    binding.codeTransformer = SimpleTransformer

PULPAddTilingReadyBindings = TilingReadyNodeBindings(nodeBindings = _PULPAddBindings,
                                                     tileConstraint = UntiledTileConstraint())

PULPiSoftmaxTilingReadyBindings = TilingReadyNodeBindings(nodeBindings = PULPSoftmaxBindings,
                                                          tileConstraint = iSoftmaxTileConstraint())

PULPConcatTilingReadyBindings = TilingReadyNodeBindings(nodeBindings = PULPConcatBindings,
                                                        tileConstraint = ConcatTileConstraint())

PULPiRMSNormTilingReadyBindings = TilingReadyNodeBindings(nodeBindings = PULPiRMSNormBindings,
                                                          tileConstraint = iRMSNormTileConstraint())

PULPiRQSGELUTilingReadyBindings = TilingReadyNodeBindings(nodeBindings = PULPiRQSGELUBindings,
                                                          tileConstraint = RQSiGELUTileConstraint())

PULPMulTilingReadyBindings = TilingReadyNodeBindings(nodeBindings = PULPMulBindings,
                                                     tileConstraint = MulTileConstraint())

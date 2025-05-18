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

from Deeploy.Targets.Redmule.Bindings import RedmuleMatmulBindings, RedmuleConv2DBindings, RedmuleGEMMBindings
from Deeploy.Targets.Redmule.TileConstraints.MatmulTileConstraint import RedmuleMatmulTileConstraint
from Deeploy.Targets.Redmule.TileConstraints.ConvTileConstraint import RedmuleConv2DTileConstraint
from Deeploy.TilingExtension.TilerExtension import TilingReadyNodeBindings
from Deeploy.Targets.Redmule.TileConstraints.GEMMTileConstraint import RedmuleGEMMTileConstraint

RedmuleMatMulTilingReadyBindings = TilingReadyNodeBindings(nodeBindings = RedmuleMatmulBindings,
                                                                tileConstraint = RedmuleMatmulTileConstraint())
RedmuleConvTilingReadyBindings = TilingReadyNodeBindings(nodeBindings = RedmuleConv2DBindings,
                                                                tileConstraint = RedmuleConv2DTileConstraint())
RedmuleGEMMTilingReadyBindings = TilingReadyNodeBindings(nodeBindings = RedmuleGEMMBindings,
                                                                tileConstraint = RedmuleGEMMTileConstraint())
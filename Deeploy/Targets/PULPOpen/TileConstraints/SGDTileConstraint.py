# SPDX-FileCopyrightText: 2023 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from Deeploy.Targets.Generic.TileConstraints.BOPTileConstraint import BOPTileConstraint


class SGDTileConstraint(BOPTileConstraint):

    dataIn1Name = 'weight'
    dataIn2Name = 'grad'
    dataOutName = 'weight_updated'


class ReluGradTileConstraint(BOPTileConstraint):

    dataIn1Name = 'grad_out'
    dataIn2Name = 'data_in'
    dataOutName = 'grad_in'

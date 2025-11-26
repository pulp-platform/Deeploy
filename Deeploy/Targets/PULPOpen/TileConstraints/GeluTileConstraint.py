# SPDX-FileCopyrightText: 2023 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from Deeploy.Targets.Generic.TileConstraints.BOPTileConstraint import BOPTileConstraint


class GeluGradTileConstraint(BOPTileConstraint):

    dataIn1Name = 'grad_in'
    dataIn2Name = 'data_in'
    dataOutName = 'grad_out'
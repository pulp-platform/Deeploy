# Copyright (C) 2025, ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

from Deeploy.Targets.Generic.TileConstraints.BOPTileConstraint import BOPTileConstraint


class SGDTileConstraint(BOPTileConstraint):

    dataIn1Name = 'weight'
    dataIn2Name = 'grad'
    dataOutName = 'weight_updated'

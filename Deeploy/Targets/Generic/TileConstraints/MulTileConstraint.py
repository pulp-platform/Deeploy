# SPDX-FileCopyrightText: 2024 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from .BOPTileConstraint import BOPScalarTileConstraint, BOPTileConstraint


class MulTileConstraint(BOPTileConstraint):
    dataIn1Name = "A"
    dataIn2Name = "B"
    dataOutName = "C"


class MulScalarTileConstraint(BOPScalarTileConstraint):
    dataIn1Name = "A"
    dataIn2Name = "B"
    dataOutName = "C"

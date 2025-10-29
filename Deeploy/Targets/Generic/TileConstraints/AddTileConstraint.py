# SPDX-FileCopyrightText: 2023 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from .BOPTileConstraint import BOPTileConstraint


class AddTileConstraint(BOPTileConstraint):
    dataIn1Name = "data_in_1"
    dataIn2Name = "data_in_2"
    dataOutName = "data_out"

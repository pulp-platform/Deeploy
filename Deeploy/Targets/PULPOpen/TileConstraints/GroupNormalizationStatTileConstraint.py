# SPDX-FileCopyrightText: 2024 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, List, Tuple

import numpy as np

from Deeploy.AbstractDataTypes import PointerClass
from Deeploy.CommonExtensions.DataTypes import uint16_t
from Deeploy.DeeployTypes import NetworkContext, OperatorRepresentation
from Deeploy.TilingExtension.MemoryConstraints import NodeMemoryConstraint
from Deeploy.TilingExtension.TileConstraint import TileConstraint
from Deeploy.TilingExtension.TilerModel import TilerModel
from Deeploy.TilingExtension.TilingCodegen import AbsoluteHyperRectangle, HyperRectangle, TilingSchedule, \
    VariableReplacementScheme


class GroupNormalizationStatTileConstraint(TileConstraint):

    @staticmethod
    def addGeometricalConstraint(tilerModel: TilerModel, parseDict: Dict, ctxt: NetworkContext) -> TilerModel:
        X_buffer_name = parseDict['X']
        stat_buffer_name = parseDict['stat']

        for buffer_name in [X_buffer_name, stat_buffer_name]:
            tilerModel.addTensorDimToModel(ctxt, buffer_name)

        input_shape = ctxt.lookup(X_buffer_name).shape
        N = input_shape[0]
        num_groups = parseDict['num_groups']

        # stat has shape [N, num_groups, 2]
        tilerModel.addConstraint(tilerModel.getTensorDimVar(tensorName = stat_buffer_name, dimIdx = 0) == N)
        tilerModel.addConstraint(tilerModel.getTensorDimVar(tensorName = stat_buffer_name, dimIdx = 1) == num_groups)
        tilerModel.addConstraint(tilerModel.getTensorDimVar(tensorName = stat_buffer_name, dimIdx = 2) == 2)

        return tilerModel

    @classmethod
    def serializeTilingSolution(
            cls, tilingSolution: NodeMemoryConstraint, absoluteOutputCubes: List[AbsoluteHyperRectangle],
            targetMemLevel: str, ctxt: NetworkContext,
            operatorRepresentation: OperatorRepresentation) -> Tuple[VariableReplacementScheme, TilingSchedule]:

        output_cubes = [cube.rectangle for cube in absoluteOutputCubes]
        addr_names = ['X', 'stat']
        input_base_offsets, output_base_offsets = cls.extractBaseAddr(tilingSolution, targetMemLevel,
                                                                      operatorRepresentation, addr_names)

        replacements = {"size": []}
        replacement_types = {"size": PointerClass(uint16_t)}

        input_load_schedule = []
        output_load_schedule = []

        num_groups = operatorRepresentation['num_groups']
        N = operatorRepresentation['N']
        C = operatorRepresentation['C']
        H = operatorRepresentation['H']
        W = operatorRepresentation['W']

        for cube in output_cubes:
            # stat cube is [N, num_groups, 2]
            # X cube should be [N, C, H, W]
            X_cube = HyperRectangle((0, 0, 0, 0), (N, C, H, W))
            new_size = N * C * H * W
            replacements["size"].append(new_size)

            input_load_schedule.append({"X": X_cube})
            output_load_schedule.append({"stat": cube})

        tiling_schedule = TilingSchedule(input_base_offsets, output_base_offsets, input_load_schedule,
                                         output_load_schedule)
        variable_replacement_schedule = VariableReplacementScheme(replacements, replacement_types)

        return variable_replacement_schedule, tiling_schedule

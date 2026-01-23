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


class GroupNormGradXTileConstraint(TileConstraint):

    @staticmethod
    def addGeometricalConstraint(tilerModel: TilerModel, parseDict: Dict, ctxt: NetworkContext) -> TilerModel:
        dY_buffer_name = parseDict['dY']
        X_buffer_name = parseDict['X']
        gamma_buffer_name = parseDict['gamma']
        stat_buffer_name = parseDict['stat']
        grad_stat_buffer_name = parseDict['grad_stat']
        dX_buffer_name = parseDict['dX']

        for buffer_name in [
                dY_buffer_name, X_buffer_name, gamma_buffer_name, stat_buffer_name, grad_stat_buffer_name, dX_buffer_name
        ]:
            tilerModel.addTensorDimToModel(ctxt, buffer_name)

        input_shape = ctxt.lookup(dY_buffer_name).shape

        # dY, X, and dX must have the same shape
        for idx, dim in enumerate(input_shape):
            tilerModel.addConstraint(
                tilerModel.getTensorDimVar(tensorName = dY_buffer_name, dimIdx = idx) ==
                tilerModel.getTensorDimVar(tensorName = X_buffer_name, dimIdx = idx))
            tilerModel.addConstraint(
                tilerModel.getTensorDimVar(tensorName = dY_buffer_name, dimIdx = idx) ==
                tilerModel.getTensorDimVar(tensorName = dX_buffer_name, dimIdx = idx))

        # gamma has shape [C]
        C = input_shape[1]
        tilerModel.addConstraint(tilerModel.getTensorDimVar(tensorName = gamma_buffer_name, dimIdx = 0) == C)

        # stat has shape [N, num_groups, 2] where [:,:,0]=mean, [:,:,1]=inv_std
        N = input_shape[0]
        num_groups = parseDict['num_groups']
        tilerModel.addConstraint(tilerModel.getTensorDimVar(tensorName = stat_buffer_name, dimIdx = 0) == N)
        tilerModel.addConstraint(tilerModel.getTensorDimVar(tensorName = stat_buffer_name, dimIdx = 1) == num_groups)
        tilerModel.addConstraint(tilerModel.getTensorDimVar(tensorName = stat_buffer_name, dimIdx = 2) == 2)
        
        grad_stat_shape = ctxt.lookup(grad_stat_buffer_name).shape
        tilerModel.addConstraint(
            tilerModel.getTensorDimVar(tensorName = grad_stat_buffer_name, dimIdx = 0) == N
        )
        tilerModel.addConstraint(
            tilerModel.getTensorDimVar(tensorName = grad_stat_buffer_name, dimIdx = 1) == num_groups
        )
        tilerModel.addConstraint(
            tilerModel.getTensorDimVar(tensorName = grad_stat_buffer_name, dimIdx = 2) == 2
        )   
        return tilerModel

    @classmethod
    def serializeTilingSolution(
            cls, tilingSolution: NodeMemoryConstraint, absoluteOutputCubes: List[AbsoluteHyperRectangle],
            targetMemLevel: str, ctxt: NetworkContext,
            operatorRepresentation: OperatorRepresentation) -> Tuple[VariableReplacementScheme, TilingSchedule]:

        output_cubes = [cube.rectangle for cube in absoluteOutputCubes]
        addr_names = ['dY', 'X', 'gamma', 'stat', 'grad_stat', 'dX']
        input_base_offsets, output_base_offsets = cls.extractBaseAddr(tilingSolution, targetMemLevel,
                                                                      operatorRepresentation, addr_names)

        replacements = {"size": []}
        replacement_types = {"size": PointerClass(uint16_t)}

        input_load_schedule = []
        output_load_schedule = []

        for cube in output_cubes:
            new_size = np.prod(cube.dims)
            replacements["size"].append(new_size)

            # gamma has shape [C], load full gamma
            C = cube.dims[1] if len(cube.dims) > 1 else cube.dims[0]
            gamma_cube = HyperRectangle((0,), (C,))

            # stat has shape [N, num_groups, 2] where [:,:,0]=mean, [:,:,1]=inv_std
            N = cube.dims[0]
            num_groups = operatorRepresentation['num_groups']
            stat_cube = HyperRectangle((0, 0, 0), (N, num_groups, 2))

            input_load_schedule.append({
                "dY": cube,
                "X": cube,
                "gamma": gamma_cube,
                "stat": HyperRectangle((0, 0, 0), (N, num_groups, 2)),
                "grad_stat": HyperRectangle((0, 0, 0), (N, num_groups, 2))
            })

            output_load_schedule.append({"dX": cube})

        tiling_schedule = TilingSchedule(input_base_offsets, output_base_offsets, input_load_schedule,
                                         output_load_schedule)
        variable_replacement_schedule = VariableReplacementScheme(replacements, replacement_types)

        return variable_replacement_schedule, tiling_schedule

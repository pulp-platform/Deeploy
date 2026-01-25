

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
    
class GroupNormGradXStatTileConstraint(TileConstraint):

    @staticmethod
    def addGeometricalConstraint(tilerModel: TilerModel, parseDict: Dict, ctxt: NetworkContext) -> TilerModel:
        dY_buffer_name = parseDict["dY"]
        X_buffer_name = parseDict["X"]
        gamma_buffer_name = parseDict["gamma"]
        stat_buffer_name = parseDict["stat"]
        grad_stat_buffer_name = parseDict["grad_stat"]

        # register tensor dims in tiler model
        for buffer_name in [
            dY_buffer_name,
            X_buffer_name,
            gamma_buffer_name,
            stat_buffer_name,
            grad_stat_buffer_name,
        ]:
            tilerModel.addTensorDimToModel(ctxt, buffer_name)

        input_shape = ctxt.lookup(X_buffer_name).shape  # expect [N, C, H, W]
        N = input_shape[0]
        num_groups = parseDict["num_groups"]

        # stat has shape [N, num_groups, 2]
        tilerModel.addConstraint(tilerModel.getTensorDimVar(tensorName=stat_buffer_name, dimIdx=0) == N)
        tilerModel.addConstraint(tilerModel.getTensorDimVar(tensorName=stat_buffer_name, dimIdx=1) == num_groups)
        tilerModel.addConstraint(tilerModel.getTensorDimVar(tensorName=stat_buffer_name, dimIdx=2) == 2)

        tilerModel.addConstraint(
            tilerModel.getTensorDimVar(tensorName=grad_stat_buffer_name, dimIdx=0) == N
        )
        tilerModel.addConstraint(
            tilerModel.getTensorDimVar(tensorName=grad_stat_buffer_name, dimIdx=1) == num_groups
        )
        tilerModel.addConstraint(
            tilerModel.getTensorDimVar(tensorName=grad_stat_buffer_name, dimIdx=2) == 2
        )

        # not tile dy and X
        # tilerModel.addConstraint(
        #     tilerModel.getTensorDimVar(tensorName=dY_buffer_name, dimIdx=0) == N
        # )
        # tilerModel.addConstraint(
        #     tilerModel.getTensorDimVar(tensorName=dY_buffer_name, dimIdx=1) == input_shape[1]
        # )       
        # tilerModel.addConstraint(
        #     tilerModel.getTensorDimVar(tensorName=dY_buffer_name, dimIdx=2) == input_shape[2]
        # )
        # tilerModel.addConstraint(
        #     tilerModel.getTensorDimVar(tensorName=dY_buffer_name, dimIdx=3) == input_shape[3]
        # )   
        # tilerModel.addConstraint(
        #     tilerModel.getTensorDimVar(tensorName=X_buffer_name, dimIdx=0) == N
        # )
        # tilerModel.addConstraint(
        #     tilerModel.getTensorDimVar(tensorName=X_buffer_name, dimIdx=1) == input_shape[1]
        # )       
        # tilerModel.addConstraint(
        #     tilerModel.getTensorDimVar(tensorName=X_buffer_name, dimIdx=2) == input_shape[2]
        # )
        # tilerModel.addConstraint(
        #     tilerModel.getTensorDimVar(tensorName=X_buffer_name, dimIdx=3) == input_shape[3]
        # )   

        return tilerModel

    @classmethod
    def serializeTilingSolution(
        cls,
        tilingSolution: NodeMemoryConstraint,
        absoluteOutputCubes: List[AbsoluteHyperRectangle],
        targetMemLevel: str,
        ctxt: NetworkContext,
        operatorRepresentation: OperatorRepresentation,
    ) -> Tuple[VariableReplacementScheme, TilingSchedule]:

        output_cubes = [cube.rectangle for cube in absoluteOutputCubes]

        # IMPORTANT: names must match operatorRepresentation keys used by codegen
        addr_names = ["dY", "X", "gamma", "stat", "grad_stat"]

        input_base_offsets, output_base_offsets = cls.extractBaseAddr(
            tilingSolution, targetMemLevel, operatorRepresentation, addr_names
        )

        # replacement example: a "size" parameter (often used by C kernel for DMA size etc.)
        replacements = {"size": []}
        replacement_types = {"size": PointerClass(uint16_t)}

        input_load_schedule = []
        output_load_schedule = []

        # Pull shapes from op-repr (as in your reference)
        num_groups = operatorRepresentation["num_groups"]
        N = operatorRepresentation["N"]
        C = operatorRepresentation["C"]
        H = operatorRepresentation["H"]
        W = operatorRepresentation["W"]

        # Full input cubes (no tiling on inputs)
        # X and dY are [N, C, H, W]
        X_cube = HyperRectangle((0, 0, 0, 0), (N, C, H, W))
        dY_cube = HyperRectangle((0, 0, 0, 0), (N, C, H, W))

        # gamma is [C]
        gamma_cube = HyperRectangle((0,), (C,))

        # mean and inv_std are [N, num_groups]
        grad_stat_cube = HyperRectangle((0, 0, 0), (N, num_groups, 2))
        stat_cube = HyperRectangle((0, 0, 0), (N, num_groups, 2))

        for cube in output_cubes:
            # output stat cube is [N, num_groups, 2]
            # conservatively assume kernel reads full inputs for any output tile
            # size here: number of X (or dY) elements read (example)
            new_size = N * C * H * W
            replacements["size"].append(new_size)

            input_load_schedule.append(
                {
                    "dY": dY_cube,
                    "X": X_cube,
                    "gamma": gamma_cube,
                    "stat": stat_cube,
                }
            )
            output_load_schedule.append({ "grad_stat": grad_stat_cube,})

        tiling_schedule = TilingSchedule(
            input_base_offsets,
            output_base_offsets,
            input_load_schedule,
            output_load_schedule,
        )
        variable_replacement_schedule = VariableReplacementScheme(replacements, replacement_types)

        return variable_replacement_schedule, tiling_schedule
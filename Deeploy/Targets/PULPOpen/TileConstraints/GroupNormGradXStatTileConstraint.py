

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

        return tilerModel

    @staticmethod
    def addPolicyConstraint(tilerModel: TilerModel, parseDict: Dict, ctxt: NetworkContext) -> TilerModel:
        dY_buffer_name = parseDict["dY"]
        X_buffer_name = parseDict["X"]

        input_shape = ctxt.lookup(X_buffer_name).shape  # [N, C, H, W]
        N = input_shape[0]
        C = input_shape[1]
        H = input_shape[2]
        W = input_shape[3]

        # Force dY and X to not be tiled (must read full inputs)
        # This ensures the tiler correctly accounts for memory usage
        tilerModel.addConstraint(tilerModel.getTensorDimVar(tensorName=dY_buffer_name, dimIdx=0) == N)
        tilerModel.addConstraint(tilerModel.getTensorDimVar(tensorName=dY_buffer_name, dimIdx=1) == C)
        tilerModel.addConstraint(tilerModel.getTensorDimVar(tensorName=dY_buffer_name, dimIdx=2) == H)
        # tilerModel.addConstraint(tilerModel.getTensorDimVar(tensorName=dY_buffer_name, dimIdx=3) == W)

        tilerModel.addConstraint(tilerModel.getTensorDimVar(tensorName=X_buffer_name, dimIdx=0) == N)
        tilerModel.addConstraint(tilerModel.getTensorDimVar(tensorName=X_buffer_name, dimIdx=1) == C)
        tilerModel.addConstraint(tilerModel.getTensorDimVar(tensorName=X_buffer_name, dimIdx=2) == H)
        # tilerModel.addConstraint(tilerModel.getTensorDimVar(tensorName=X_buffer_name, dimIdx=3) == W)

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

        # Pull shapes from op-repr
        num_groups = operatorRepresentation["num_groups"]
        N = operatorRepresentation["N"]
        C = operatorRepresentation["C"]
        H = operatorRepresentation["H"]
        W = operatorRepresentation["W"]

        # Get buffers
        dY_buffer = ctxt.lookup(operatorRepresentation["dY"])
        X_buffer = ctxt.lookup(operatorRepresentation["X"])

        # Extract tiled dimensions from the solution for dY
        try:
            dY_tile_shape = tilingSolution.tensorMemoryConstraints[operatorRepresentation["dY"]].memoryConstraints[targetMemLevel].shape
        except Exception:
            # Fallback to full shape if tiling info not available
            dY_tile_shape = tuple(dY_buffer.shape)

        # Get the tiled dimensions (N, C, H are constrained, W can be tiled)
        N_tile = dY_tile_shape[0]
        C_tile = dY_tile_shape[1]
        H_tile = dY_tile_shape[2]
        W_tile = dY_tile_shape[3]

        # gamma is always [C] (no tiling)
        gamma_cube = HyperRectangle((0,), (C,))

        # stat is always [N, num_groups, 2] (no tiling on stat input)
        stat_cube = HyperRectangle((0, 0, 0), (N, num_groups, 2))

        for cube in output_cubes:
            # X and dY cubes use tiled dimensions
            # Since N, C, H are constrained to not tile, only W can vary
            X_cube = HyperRectangle((0, 0, 0, 0), (N_tile, C_tile, H_tile, W_tile))
            dY_cube = HyperRectangle((0, 0, 0, 0), (N_tile, C_tile, H_tile, W_tile))

            # grad_stat output matches the output cube
            grad_stat_cube = cube

            # size: number of X (or dY) elements read per tile
            new_size = N_tile * C_tile * H_tile * W_tile
            replacements["size"].append(new_size)

            input_load_schedule.append(
                {
                    "dY": dY_cube,
                    "X": X_cube,
                    "gamma": gamma_cube,
                    "stat": stat_cube,
                }
            )
            output_load_schedule.append({"grad_stat": grad_stat_cube})

        tiling_schedule = TilingSchedule(
            input_base_offsets,
            output_base_offsets,
            input_load_schedule,
            output_load_schedule,
        )
        variable_replacement_schedule = VariableReplacementScheme(replacements, replacement_types)

        return variable_replacement_schedule, tiling_schedule
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
from Deeploy.TilingExtension.TilingCodegen import (
    AbsoluteHyperRectangle,
    HyperRectangle,
    TilingSchedule,
    VariableReplacementScheme,
)


class GroupNormGradWTileConstraint(TileConstraint):

    @staticmethod
    def addGeometricalConstraint(tilerModel: TilerModel, parseDict: Dict, ctxt: NetworkContext) -> TilerModel:
        dY_name = parseDict["dY"]
        X_name = parseDict["X"]
        stat_name = parseDict["stat"]
        dGamma_name = parseDict["dGamma"]

        for name in [dY_name, X_name, stat_name, dGamma_name]:
            tilerModel.addTensorDimToModel(ctxt, name)

        dY_shape = ctxt.lookup(dY_name).shape  # expect [N,C,H,W]
        # N, C, H, W = dY_shape[0], dY_shape[1], dY_shape[2], dY_shape[3]
        N, C, H, W = dY_shape[0], dY_shape[1], dY_shape[2], dY_shape[3]
        num_groups = parseDict["num_groups"]

        # dY and X must have same shape
        for idx in range(len(dY_shape)):
            tilerModel.addConstraint(
                tilerModel.getTensorDimVar(tensorName=dY_name, dimIdx=idx) ==
                tilerModel.getTensorDimVar(tensorName=X_name, dimIdx=idx)
            )

        # dGamma shape: [C]
        tilerModel.addConstraint(tilerModel.getTensorDimVar(tensorName=dGamma_name, dimIdx=0) == C)

        # stat shape: [N, num_groups, 2]
        tilerModel.addConstraint(tilerModel.getTensorDimVar(tensorName=stat_name, dimIdx=0) == N)
        tilerModel.addConstraint(tilerModel.getTensorDimVar(tensorName=stat_name, dimIdx=1) == num_groups)
        tilerModel.addConstraint(tilerModel.getTensorDimVar(tensorName=stat_name, dimIdx=2) == 2)

        # Don't tile on N and C dimensions for dY/X (must compute full gradient for all channels)
        # But H and W can be tiled
        tilerModel.addConstraint(tilerModel.getTensorDimVar(tensorName=dY_name, dimIdx=0) == N)
        tilerModel.addConstraint(tilerModel.getTensorDimVar(tensorName=dY_name, dimIdx=1) == C)
        # tilerModel.addConstraint(tilerModel.getTensorDimVar(tensorName=dY_name, dimIdx=2) == H)
        # tilerModel.addConstraint(tilerModel.getTensorDimVar(tensorName=dY_name, dimIdx=3) == W)
        tilerModel.addConstraint(tilerModel.getTensorDimVar(tensorName=X_name, dimIdx=0) == N)
        tilerModel.addConstraint(tilerModel.getTensorDimVar(tensorName=X_name, dimIdx=1) == C)
        # tilerModel.addConstraint(tilerModel.getTensorDimVar(tensorName=X_name, dimIdx=2) == H)
        # tilerModel.addConstraint(tilerModel.getTensorDimVar(tensorName=X_name, dimIdx=3) == W)

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

        dY_name = operatorRepresentation["dY"]
        addr_names = ["dY", "X", "stat", "dGamma"]
        input_base_offsets, output_base_offsets = cls.extractBaseAddr(
            tilingSolution, targetMemLevel, operatorRepresentation, addr_names
        )
        
        X_name = operatorRepresentation["X"]
        
        replacements: Dict[str, List[int]] = {"size": [], "H": [], "W": []}
        replacement_types: Dict[str, PointerClass] = {
            "size": PointerClass(uint16_t),
            "H": PointerClass(uint16_t),
            "W": PointerClass(uint16_t)
        }

        input_load_schedule = []
        output_load_schedule = []

        # Get full shape from context
        dY_full_shape = ctxt.lookup(dY_name).shape
        N = dY_full_shape[0]
        C = dY_full_shape[1]
        H_full = dY_full_shape[2]
        W_full = dY_full_shape[3]
        num_groups = operatorRepresentation["num_groups"]

        # Get tiled dY shape to construct input cubes
        dYTileShape = tilingSolution.tensorMemoryConstraints[dY_name].memoryConstraints[targetMemLevel].shape
        XTileShape = tilingSolution.tensorMemoryConstraints[X_name].memoryConstraints[targetMemLevel].shape

        # For GroupNormGradW, output dGamma is [C] (not tiled)
        # But inputs dY and X are tiled on HW dimensions
        # We need to iterate over HW tiles, not output tiles
        
        # Generate HW tiles based on dYTileShape
        H_tile_size = dYTileShape[2]
        W_tile_size = dYTileShape[3]
        
        # Generate tile coordinates
        h_tiles = []
        h_offset = 0
        while h_offset < H_full:
            h_size = min(H_tile_size, H_full - h_offset)
            h_tiles.append((h_offset, h_size))
            h_offset += h_size
        
        w_tiles = []
        w_offset = 0
        while w_offset < W_full:
            w_size = min(W_tile_size, W_full - w_offset)
            w_tiles.append((w_offset, w_size))
            w_offset += w_size
        
        # Create cubes for each HW tile
        for h_off, h_sz in h_tiles:
            for w_off, w_sz in w_tiles:
                # dGamma is [C], always the same for all tiles
                dGamma_cube = output_cubes[0] if output_cubes else HyperRectangle((0,), (C,))
                
                new_size = np.prod(dGamma_cube.dims)
                replacements["size"].append(new_size)
                
                # Add tiled H and W for this specific tile
                replacements["H"].append(h_sz)
                replacements["W"].append(w_sz)

                # stat has shape [N, num_groups, 2] - always load full stat
                stat_cube = HyperRectangle((0, 0, 0), (N, num_groups, 2))

                # dY and X cubes for this HW tile
                dY_cube = HyperRectangle((0, 0, h_off, w_off), (N, C, h_sz, w_sz))
                X_cube = HyperRectangle((0, 0, h_off, w_off), (N, C, h_sz, w_sz))

                input_load_schedule.append({"dY": dY_cube, "X": X_cube, "stat": stat_cube})
                output_load_schedule.append({"dGamma": dGamma_cube})

        tiling_schedule = TilingSchedule(
            input_base_offsets, output_base_offsets, input_load_schedule, output_load_schedule
        )
        variable_replacement_schedule = VariableReplacementScheme(replacements, replacement_types)

        return variable_replacement_schedule, tiling_schedule

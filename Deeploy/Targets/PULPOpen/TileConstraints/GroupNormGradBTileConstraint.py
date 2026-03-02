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


class GroupNormGradBTileConstraint(TileConstraint):

    @staticmethod
    def addGeometricalConstraint(tilerModel: TilerModel, parseDict: Dict, ctxt: NetworkContext) -> TilerModel:
        dY_name = parseDict["dY"]
        dBeta_name = parseDict["dBeta"]

        for name in [dY_name, dBeta_name]:
            tilerModel.addTensorDimToModel(ctxt, name)

        dY_shape = ctxt.lookup(dY_name).shape  # expect [N,C,H,W]
        N, C = dY_shape[0], dY_shape[1]

        # dBeta shape: [C]
        tilerModel.addConstraint(tilerModel.getTensorDimVar(tensorName=dBeta_name, dimIdx=0) == C)

        # Don't tile on N and C dimensions for dY (must compute full gradient for all channels)
        # But H and W can be tiled
        tilerModel.addConstraint(tilerModel.getTensorDimVar(tensorName=dY_name, dimIdx=0) == N)
        tilerModel.addConstraint(tilerModel.getTensorDimVar(tensorName=dY_name, dimIdx=1) == C)

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
        addr_names = ["dY", "dBeta"]
        input_base_offsets, output_base_offsets = cls.extractBaseAddr(
            tilingSolution, targetMemLevel, operatorRepresentation, addr_names
        )

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

        # Get tiled dY shape to construct input cubes
        dYTileShape = tilingSolution.tensorMemoryConstraints[dY_name].memoryConstraints[targetMemLevel].shape

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
                # dBeta is [C], always the same for all tiles
                dBeta_cube = output_cubes[0] if output_cubes else HyperRectangle((0,), (C,))

                new_size = np.prod(dBeta_cube.dims)
                replacements["size"].append(new_size)

                # Add tiled H and W for this specific tile
                replacements["H"].append(h_sz)
                replacements["W"].append(w_sz)

                # dY cube for this HW tile
                dY_cube = HyperRectangle((0, 0, h_off, w_off), (N, C, h_sz, w_sz))

                input_load_schedule.append({"dY": dY_cube})
                output_load_schedule.append({"dBeta": dBeta_cube})

        tiling_schedule = TilingSchedule(
            input_base_offsets, output_base_offsets, input_load_schedule, output_load_schedule
        )
        variable_replacement_schedule = VariableReplacementScheme(replacements, replacement_types)

        return variable_replacement_schedule, tiling_schedule

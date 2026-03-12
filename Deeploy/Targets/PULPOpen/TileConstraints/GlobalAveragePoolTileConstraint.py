# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, List, Tuple

from Deeploy.AbstractDataTypes import PointerClass
from Deeploy.CommonExtensions.DataTypes import uint16_t
from Deeploy.DeeployTypes import NetworkContext, OperatorRepresentation
from Deeploy.TilingExtension.MemoryConstraints import NodeMemoryConstraint
from Deeploy.TilingExtension.TileConstraint import TileConstraint
from Deeploy.TilingExtension.TilerModel import TilerModel
from Deeploy.TilingExtension.TilingCodegen import AbsoluteHyperRectangle, HyperRectangle, TilingSchedule, \
    VariableReplacementScheme


class GlobalAveragePoolTileConstraint(TileConstraint):
    """Tile constraint for GlobalAveragePool (NCHW).

    Input:  data_in  [N, C, H, W]
    Output: data_out [N, C, 1, 1]  (N*C elements)

    Tiling strategy: tile along C (channels are independent).
      - N, H, W are pinned to full: GAP needs all spatial elements per channel.
      - C is free: each channel's mean is computed independently.
    """

    @staticmethod
    def addGeometricalConstraint(tilerModel: TilerModel, parseDict: Dict, ctxt: NetworkContext) -> TilerModel:
        data_in_name = parseDict['data_in']
        data_out_name = parseDict['data_out']

        for name in [data_in_name, data_out_name]:
            tilerModel.addTensorDimToModel(ctxt, name)

        input_shape = ctxt.lookup(data_in_name).shape
        N = input_shape[0]
        H = input_shape[2]
        W = input_shape[3]

        # Pin N, H, W to full
        tilerModel.addConstraint(tilerModel.getTensorDimVar(tensorName=data_in_name, dimIdx=0) == N)
        tilerModel.addConstraint(tilerModel.getTensorDimVar(tensorName=data_in_name, dimIdx=2) == H)
        tilerModel.addConstraint(tilerModel.getTensorDimVar(tensorName=data_in_name, dimIdx=3) == W)

        # Output shape [N, C, 1, 1]: dim 0 = N (pinned), dim 1 = C (free)
        # data_out is stored as [N, C] effectively; tilerModel sees it as 4D [N,C,1,1]
        tilerModel.addConstraint(
            tilerModel.getTensorDimVar(tensorName=data_out_name, dimIdx=0) ==
            tilerModel.getTensorDimVar(tensorName=data_in_name, dimIdx=0))
        tilerModel.addConstraint(
            tilerModel.getTensorDimVar(tensorName=data_out_name, dimIdx=1) ==
            tilerModel.getTensorDimVar(tensorName=data_in_name, dimIdx=1))

        return tilerModel

    @classmethod
    def serializeTilingSolution(
            cls, tilingSolution: NodeMemoryConstraint, absoluteOutputCubes: List[AbsoluteHyperRectangle],
            targetMemLevel: str, ctxt: NetworkContext,
            operatorRepresentation: OperatorRepresentation) -> Tuple[VariableReplacementScheme, TilingSchedule]:
        output_cubes = [cube.rectangle for cube in absoluteOutputCubes]

        addr_names = ['data_in', 'data_out']
        input_base_offsets, output_base_offsets = cls.extractBaseAddr(tilingSolution, targetMemLevel,
                                                                       operatorRepresentation, addr_names)

        replacements = {"channels": []}
        replacement_types = {"channels": PointerClass(uint16_t)}

        input_load_schedule = []
        output_load_schedule = []

        input_shape = ctxt.lookup(operatorRepresentation['data_in']).shape
        H = input_shape[2]
        W = input_shape[3]

        for cube in output_cubes:
            # cube is the tile of data_out: [N, C_tile, 1, 1]
            C_tile = cube.dims[1]
            c_start = cube.offset[1]

            replacements["channels"].append(C_tile)

            # Input tile: [N, C_tile, H, W]
            in_cube = HyperRectangle(
                (cube.offset[0], c_start, 0, 0),
                (cube.dims[0], C_tile, H, W)
            )
            input_load_schedule.append({"data_in": in_cube})
            output_load_schedule.append({"data_out": cube})

        tiling_schedule = TilingSchedule(input_base_offsets, output_base_offsets, input_load_schedule,
                                          output_load_schedule)
        variable_replacement_schedule = VariableReplacementScheme(replacements, replacement_types)
        return variable_replacement_schedule, tiling_schedule


class GlobalAveragePoolGradTileConstraint(TileConstraint):
    """Tile constraint for GlobalAveragePoolGrad (NCHW).

    Input:  dY [N, C, 1, 1]  (N*C elements)
    Output: dX [N, C, H, W]

    Tiling strategy: tile along C (channels are independent).
      - N, H, W are pinned to full.
      - C is free.
    """

    @staticmethod
    def addGeometricalConstraint(tilerModel: TilerModel, parseDict: Dict, ctxt: NetworkContext) -> TilerModel:
        dY_name = parseDict['dY']
        dX_name = parseDict['dX']

        for name in [dY_name, dX_name]:
            tilerModel.addTensorDimToModel(ctxt, name)

        output_shape = ctxt.lookup(dX_name).shape
        N = output_shape[0]
        H = output_shape[2]
        W = output_shape[3]

        # Pin N, H, W to full
        tilerModel.addConstraint(tilerModel.getTensorDimVar(tensorName=dX_name, dimIdx=0) == N)
        tilerModel.addConstraint(tilerModel.getTensorDimVar(tensorName=dX_name, dimIdx=2) == H)
        tilerModel.addConstraint(tilerModel.getTensorDimVar(tensorName=dX_name, dimIdx=3) == W)

        # dY [N, C, 1, 1]: N pinned, C follows dX C
        tilerModel.addConstraint(
            tilerModel.getTensorDimVar(tensorName=dY_name, dimIdx=0) ==
            tilerModel.getTensorDimVar(tensorName=dX_name, dimIdx=0))
        tilerModel.addConstraint(
            tilerModel.getTensorDimVar(tensorName=dY_name, dimIdx=1) ==
            tilerModel.getTensorDimVar(tensorName=dX_name, dimIdx=1))

        return tilerModel

    @classmethod
    def serializeTilingSolution(
            cls, tilingSolution: NodeMemoryConstraint, absoluteOutputCubes: List[AbsoluteHyperRectangle],
            targetMemLevel: str, ctxt: NetworkContext,
            operatorRepresentation: OperatorRepresentation) -> Tuple[VariableReplacementScheme, TilingSchedule]:
        output_cubes = [cube.rectangle for cube in absoluteOutputCubes]

        addr_names = ['dY', 'dX']
        input_base_offsets, output_base_offsets = cls.extractBaseAddr(tilingSolution, targetMemLevel,
                                                                       operatorRepresentation, addr_names)

        replacements = {"channels": []}
        replacement_types = {"channels": PointerClass(uint16_t)}

        input_load_schedule = []
        output_load_schedule = []

        for cube in output_cubes:
            # cube is the tile of dX: [N, C_tile, H, W]
            C_tile = cube.dims[1]
            c_start = cube.offset[1]

            replacements["channels"].append(C_tile)

            # dY tile: [N, C_tile, 1, 1]
            dy_cube = HyperRectangle(
                (cube.offset[0], c_start, 0, 0),
                (cube.dims[0], C_tile, 1, 1)
            )
            input_load_schedule.append({"dY": dy_cube})
            output_load_schedule.append({"dX": cube})

        tiling_schedule = TilingSchedule(input_base_offsets, output_base_offsets, input_load_schedule,
                                          output_load_schedule)
        variable_replacement_schedule = VariableReplacementScheme(replacements, replacement_types)
        return variable_replacement_schedule, tiling_schedule

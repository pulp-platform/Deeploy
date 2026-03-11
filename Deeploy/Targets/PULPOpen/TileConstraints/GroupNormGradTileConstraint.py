# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

import copy
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


class GroupNormGradTileConstraint(TileConstraint):
    """Tile constraint for merged GroupNormGrad node.

    Inputs:  dY[N,C,H,W], X[N,C,H,W], gamma[C], stat[N,G,2]
    Outputs: dX[N,C,H,W] (primary), weight_grad[C], bias_grad[C]

    Tiling strategy:
      - C, H, W are pinned to full size: GradXStat must sum over all channels per
        group and all spatial positions before dX can be computed.
      - N is free: each batch element is fully independent.
      - stat tiles with N (first dimension).
      - weight_grad / bias_grad are full-size [C] outputs accumulated with a
        static-flag memset + inline C loops in the template (ConvGradW pattern).
    """

    @staticmethod
    def addGeometricalConstraint(tilerModel: TilerModel, parseDict: Dict, ctxt: NetworkContext) -> TilerModel:
        dY_name = parseDict['dY']
        X_name = parseDict['X']
        gamma_name = parseDict['gamma']
        stat_name = parseDict['stat']
        dX_name = parseDict['dX']
        weight_grad_name = parseDict['weight_grad']
        bias_grad_name = parseDict['bias_grad']

        for name in [dY_name, X_name, gamma_name, stat_name, dX_name, weight_grad_name, bias_grad_name]:
            tilerModel.addTensorDimToModel(ctxt, name)

        input_shape = ctxt.lookup(dY_name).shape
        N = input_shape[0]
        C = input_shape[1]
        H = input_shape[2] if len(input_shape) > 2 else 1
        W = input_shape[3] if len(input_shape) > 3 else 1
        num_groups = parseDict['num_groups']

        # Pin C, H, W to full (GradXStat needs all spatial+channel positions per group)
        tilerModel.addConstraint(tilerModel.getTensorDimVar(tensorName=dY_name, dimIdx=1) == C)
        tilerModel.addConstraint(tilerModel.getTensorDimVar(tensorName=dY_name, dimIdx=2) == H)
        if len(input_shape) > 3:
            tilerModel.addConstraint(tilerModel.getTensorDimVar(tensorName=dY_name, dimIdx=3) == W)

        # dY, X, dX must have the same shape (N tiles freely)
        for idx in range(len(input_shape)):
            tilerModel.addConstraint(
                tilerModel.getTensorDimVar(tensorName=dY_name, dimIdx=idx) ==
                tilerModel.getTensorDimVar(tensorName=X_name, dimIdx=idx))
            tilerModel.addConstraint(
                tilerModel.getTensorDimVar(tensorName=dY_name, dimIdx=idx) ==
                tilerModel.getTensorDimVar(tensorName=dX_name, dimIdx=idx))

        # gamma: full C (constant, not tiled)
        tilerModel.addConstraint(tilerModel.getTensorDimVar(tensorName=gamma_name, dimIdx=0) == C)

        # stat[N, G, 2]: first dim tiles with N; G and 2 are pinned to full
        tilerModel.addConstraint(
            tilerModel.getTensorDimVar(tensorName=stat_name, dimIdx=0) ==
            tilerModel.getTensorDimVar(tensorName=dY_name, dimIdx=0))
        tilerModel.addConstraint(tilerModel.getTensorDimVar(tensorName=stat_name, dimIdx=1) == num_groups)
        tilerModel.addConstraint(tilerModel.getTensorDimVar(tensorName=stat_name, dimIdx=2) == 2)

        # weight_grad, bias_grad: full C (accumulated across N tiles)
        tilerModel.addConstraint(tilerModel.getTensorDimVar(tensorName=weight_grad_name, dimIdx=0) == C)
        tilerModel.addConstraint(tilerModel.getTensorDimVar(tensorName=bias_grad_name, dimIdx=0) == C)

        return tilerModel

    @classmethod
    def wrapTilingSolution(
            cls, tilingSolution: NodeMemoryConstraint, targetMemLevel: str, ctxt: NetworkContext,
            operatorRepresentation: OperatorRepresentation) -> Tuple[VariableReplacementScheme, List[TilingSchedule]]:

        dXVar = operatorRepresentation['dX']

        # Build a single-output copy to pass the base-class assertion
        singleOutputSolution = copy.deepcopy(tilingSolution)
        singleOutputSolution.outputTensorMemoryConstraints = {
            dXVar: tilingSolution.outputTensorMemoryConstraints[dXVar]
        }

        varReplacement, tilingSchedules = super().wrapTilingSolution(singleOutputSolution, targetMemLevel, ctxt,
                                                                      operatorRepresentation)

        # Extend each tiling schedule to include weight_grad and bias_grad.
        # These are always full-size [C] tensors (accumulated with static flag).
        for secondary in ['weight_grad', 'bias_grad']:
            secondaryVar = operatorRepresentation.get(secondary, '')
            if not secondaryVar:
                continue
            if secondaryVar not in tilingSolution.outputTensorMemoryConstraints:
                continue
            addr = TileConstraint.getBaseAddr(tilingSolution, targetMemLevel, secondaryVar)
            if addr == [None]:
                continue
            buf = ctxt.lookup(secondaryVar)
            full_rect = HyperRectangle((0,) * len(buf.shape), tuple(buf.shape))
            for schedule in tilingSchedules:
                schedule.outputBaseOffsets[secondary] = addr
                for step in schedule.outputLoadSchedule:
                    step[secondary] = full_rect

        return varReplacement, tilingSchedules

    @classmethod
    def serializeTilingSolution(
            cls, tilingSolution: NodeMemoryConstraint, absoluteOutputCubes: List[AbsoluteHyperRectangle],
            targetMemLevel: str, ctxt: NetworkContext,
            operatorRepresentation: OperatorRepresentation) -> Tuple[VariableReplacementScheme, TilingSchedule]:

        output_cubes = [cube.rectangle for cube in absoluteOutputCubes]
        addr_names = ['dY', 'X', 'gamma', 'stat', 'dX']
        input_base_offsets, output_base_offsets = cls.extractBaseAddr(tilingSolution, targetMemLevel,
                                                                       operatorRepresentation, addr_names)

        replacements = {"size": [], "N": []}
        replacement_types = {"size": PointerClass(uint16_t), "N": PointerClass(uint16_t)}

        input_load_schedule = []
        output_load_schedule = []

        for cube in output_cubes:
            new_size = np.prod(cube.dims)
            N_tile = cube.dims[0]
            num_groups = operatorRepresentation['num_groups']

            replacements["size"].append(new_size)
            replacements["N"].append(N_tile)

            gamma_cube = HyperRectangle((0,), (cube.dims[1],))  # full C
            # stat tiles along the N dimension: offset = (N_start, 0, 0), dims = (N_tile, G, 2)
            N_start = cube.offset[0]
            stat_cube = HyperRectangle((N_start, 0, 0), (N_tile, num_groups, 2))

            input_load_schedule.append({
                "dY": cube,
                "X": cube,
                "gamma": gamma_cube,
                "stat": stat_cube,
            })
            output_load_schedule.append({"dX": cube})

        tiling_schedule = TilingSchedule(input_base_offsets, output_base_offsets, input_load_schedule,
                                          output_load_schedule)
        variable_replacement_schedule = VariableReplacementScheme(replacements, replacement_types)

        return variable_replacement_schedule, tiling_schedule

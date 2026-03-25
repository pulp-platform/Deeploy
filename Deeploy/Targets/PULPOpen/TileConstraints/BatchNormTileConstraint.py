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


class BatchNormInternalTileConstraint(TileConstraint):
    """Tile constraint for BatchNormInternal (ORT training-mode BN forward pass).

    Inputs:  X[N,C,H,W], gamma[C], beta[C], running_mean[C], running_var[C]
    Outputs: Y[N,C,H,W] (primary), saved_mean[C], saved_inv_std[C] (secondary)
             (updated_running_mean/var are not registered — no consumers)

    Tiling strategy: tile along C (channels are independent).
      - N, H_in, W_in are pinned to full size: BN needs all N*H*W elements
        per channel to compute per-channel mean and variance.
      - C is free: channels are fully independent.
      - Per-channel vectors (gamma, beta, running_mean, running_var,
        saved_mean, saved_inv_std) tile with the C dimension of data_in.
    """

    @staticmethod
    def addGeometricalConstraint(tilerModel: TilerModel, parseDict: Dict, ctxt: NetworkContext) -> TilerModel:
        data_in_name = parseDict['data_in']
        data_out_name = parseDict['data_out']
        scale_name = parseDict['scale']
        bias_name = parseDict['bias']
        running_mean_name = parseDict['running_mean']
        running_var_name = parseDict['running_var']
        saved_mean_name = parseDict['saved_mean']
        saved_inv_std_name = parseDict['saved_inv_std']
        updated_running_mean_name = parseDict.get('updated_running_mean', '')
        updated_running_var_name = parseDict.get('updated_running_var', '')

        for name in [
                data_in_name, data_out_name, scale_name, bias_name, running_mean_name, running_var_name,
                saved_mean_name, saved_inv_std_name
        ]:
            tilerModel.addTensorDimToModel(ctxt, name)

        for name in [updated_running_mean_name, updated_running_var_name]:
            if name:
                tilerModel.addTensorDimToModel(ctxt, name)

        input_shape = ctxt.lookup(data_in_name).shape
        N = input_shape[0]
        H_in = input_shape[2]
        W_in = input_shape[3]

        # Pin N, H_in, W_in: BN statistics require all spatial/batch elements per channel
        tilerModel.addConstraint(tilerModel.getTensorDimVar(tensorName=data_in_name, dimIdx=0) == N)
        tilerModel.addConstraint(tilerModel.getTensorDimVar(tensorName=data_in_name, dimIdx=2) == H_in)
        tilerModel.addConstraint(tilerModel.getTensorDimVar(tensorName=data_in_name, dimIdx=3) == W_in)

        # data_out has the same shape as data_in
        for idx in range(4):
            tilerModel.addConstraint(
                tilerModel.getTensorDimVar(tensorName=data_in_name, dimIdx=idx) ==
                tilerModel.getTensorDimVar(tensorName=data_out_name, dimIdx=idx))

        # Per-channel vectors: single dimension follows C (dim 1 of data_in)
        for vec_name in [scale_name, bias_name, running_mean_name, running_var_name, saved_mean_name,
                         saved_inv_std_name]:
            tilerModel.addConstraint(
                tilerModel.getTensorDimVar(tensorName=vec_name, dimIdx=0) ==
                tilerModel.getTensorDimVar(tensorName=data_in_name, dimIdx=1))

        for vec_name in [updated_running_mean_name, updated_running_var_name]:
            if vec_name:
                tilerModel.addConstraint(
                    tilerModel.getTensorDimVar(tensorName=vec_name, dimIdx=0) ==
                    tilerModel.getTensorDimVar(tensorName=data_in_name, dimIdx=1))

        return tilerModel

    @classmethod
    def wrapTilingSolution(
            cls, tilingSolution: NodeMemoryConstraint, targetMemLevel: str, ctxt: NetworkContext,
            operatorRepresentation: OperatorRepresentation) -> Tuple[VariableReplacementScheme, List[TilingSchedule]]:

        dataOutVar = operatorRepresentation['data_out']

        # Pass a single-output solution to satisfy the base-class assertion
        singleOutputSolution = copy.deepcopy(tilingSolution)
        singleOutputSolution.outputTensorMemoryConstraints = {
            dataOutVar: tilingSolution.outputTensorMemoryConstraints[dataOutVar]
        }

        varReplacement, tilingSchedules = super().wrapTilingSolution(singleOutputSolution, targetMemLevel, ctxt,
                                                                      operatorRepresentation)

        # Extend each schedule to include saved_mean and saved_inv_std outputs
        for secondary in ['saved_mean', 'saved_inv_std']:
            secondaryVar = operatorRepresentation.get(secondary, '')
            if not secondaryVar:
                continue
            if secondaryVar not in tilingSolution.outputTensorMemoryConstraints:
                continue
            addr = TileConstraint.getBaseAddr(tilingSolution, targetMemLevel, secondaryVar)
            if addr == [None]:
                continue
            for schedule in tilingSchedules:
                schedule.outputBaseOffsets[secondary] = addr
                for step in schedule.outputLoadSchedule:
                    data_out_rect = step['data_out']
                    # Per-channel slice corresponding to the C tile in data_out
                    c_start = data_out_rect.offset[1]
                    c_tile = data_out_rect.dims[1]
                    step[secondary] = HyperRectangle((c_start,), (c_tile,))

        return varReplacement, tilingSchedules

    @classmethod
    def serializeTilingSolution(
            cls, tilingSolution: NodeMemoryConstraint, absoluteOutputCubes: List[AbsoluteHyperRectangle],
            targetMemLevel: str, ctxt: NetworkContext,
            operatorRepresentation: OperatorRepresentation) -> Tuple[VariableReplacementScheme, TilingSchedule]:

        output_cubes = [cube.rectangle for cube in absoluteOutputCubes]

        # Global model parameters (scale/bias/running_mean/running_var) may be excluded
        # from the tiling solution by _checkResolve (global tensors with 1 memory level).
        # Apply the same _in_solution guard as ConvTileConstraint.bias_in_solution.
        scale_in_solution = operatorRepresentation['scale'] in tilingSolution.tensorMemoryConstraints
        bias_in_solution = operatorRepresentation['bias'] in tilingSolution.tensorMemoryConstraints
        running_mean_in_solution = operatorRepresentation['running_mean'] in tilingSolution.tensorMemoryConstraints
        running_var_in_solution = operatorRepresentation['running_var'] in tilingSolution.tensorMemoryConstraints

        addr_names = ['data_in', 'data_out']
        if scale_in_solution:
            addr_names.append('scale')
        if bias_in_solution:
            addr_names.append('bias')
        if running_mean_in_solution:
            addr_names.append('running_mean')
        if running_var_in_solution:
            addr_names.append('running_var')

        input_base_offsets, output_base_offsets = cls.extractBaseAddr(tilingSolution, targetMemLevel,
                                                                       operatorRepresentation, addr_names)

        replacements = {"C": []}
        replacement_types = {"C": PointerClass(uint16_t)}

        input_load_schedule = []
        output_load_schedule = []

        for cube in output_cubes:
            # cube is the tile of data_out: [N, C_tile, H_in, W_in]
            C_tile = cube.dims[1]
            c_start = cube.offset[1]

            replacements["C"].append(C_tile)

            # Per-channel vector tile: offset=(c_start,), dims=(C_tile,)
            vec_cube = HyperRectangle((c_start,), (C_tile,))

            entry = {"data_in": cube}
            if scale_in_solution:
                entry["scale"] = vec_cube
            if bias_in_solution:
                entry["bias"] = vec_cube
            if running_mean_in_solution:
                entry["running_mean"] = vec_cube
            if running_var_in_solution:
                entry["running_var"] = vec_cube
            input_load_schedule.append(entry)
            output_load_schedule.append({"data_out": cube})

        tiling_schedule = TilingSchedule(input_base_offsets, output_base_offsets, input_load_schedule,
                                          output_load_schedule)
        variable_replacement_schedule = VariableReplacementScheme(replacements, replacement_types)
        return variable_replacement_schedule, tiling_schedule


class WelfordReduceTileConstraint(TileConstraint):
    """Tile constraint for WelfordReduce (split BN forward reduction).

    Inputs:  X[N,C,H,W]
    Outputs: saved_mean[C], saved_inv_std[C]

    Tiling: C is free; N, H, W are pinned (reduction over spatial).
    Memory per tile: X_tile[1,C_tile,H,W] + mean[C_tile] + inv_std[C_tile]
    → much smaller than monolithic BN (no Y needed during reduction).
    """

    @staticmethod
    def addGeometricalConstraint(tilerModel: TilerModel, parseDict: Dict, ctxt: NetworkContext) -> TilerModel:
        data_in_name = parseDict['data_in']
        saved_mean_name = parseDict['saved_mean']
        saved_inv_std_name = parseDict['saved_inv_std']

        for name in [data_in_name, saved_mean_name, saved_inv_std_name]:
            tilerModel.addTensorDimToModel(ctxt, name)

        input_shape = ctxt.lookup(data_in_name).shape
        N = input_shape[0]
        H_in = input_shape[2]
        W_in = input_shape[3]

        # Pin N, H, W: reduction needs full spatial
        tilerModel.addConstraint(tilerModel.getTensorDimVar(tensorName=data_in_name, dimIdx=0) == N)
        tilerModel.addConstraint(tilerModel.getTensorDimVar(tensorName=data_in_name, dimIdx=2) == H_in)
        tilerModel.addConstraint(tilerModel.getTensorDimVar(tensorName=data_in_name, dimIdx=3) == W_in)

        # Per-channel outputs follow C
        for vec_name in [saved_mean_name, saved_inv_std_name]:
            tilerModel.addConstraint(
                tilerModel.getTensorDimVar(tensorName=vec_name, dimIdx=0) ==
                tilerModel.getTensorDimVar(tensorName=data_in_name, dimIdx=1))

        return tilerModel

    @classmethod
    def wrapTilingSolution(
            cls, tilingSolution: NodeMemoryConstraint, targetMemLevel: str, ctxt: NetworkContext,
            operatorRepresentation: OperatorRepresentation) -> Tuple[VariableReplacementScheme, List[TilingSchedule]]:

        # Primary output is saved_mean; add saved_inv_std as secondary
        savedMeanVar = operatorRepresentation['saved_mean']
        singleOutputSolution = copy.deepcopy(tilingSolution)
        singleOutputSolution.outputTensorMemoryConstraints = {
            savedMeanVar: tilingSolution.outputTensorMemoryConstraints[savedMeanVar]
        }

        varReplacement, tilingSchedules = super().wrapTilingSolution(singleOutputSolution, targetMemLevel, ctxt,
                                                                      operatorRepresentation)

        secondaryVar = operatorRepresentation['saved_inv_std']
        if secondaryVar in tilingSolution.outputTensorMemoryConstraints:
            addr = TileConstraint.getBaseAddr(tilingSolution, targetMemLevel, secondaryVar)
            if addr != [None]:
                for schedule in tilingSchedules:
                    schedule.outputBaseOffsets['saved_inv_std'] = addr
                    for step in schedule.outputLoadSchedule:
                        mean_rect = step['saved_mean']
                        step['saved_inv_std'] = HyperRectangle(mean_rect.offset, mean_rect.dims)

        return varReplacement, tilingSchedules

    @classmethod
    def serializeTilingSolution(
            cls, tilingSolution: NodeMemoryConstraint, absoluteOutputCubes: List[AbsoluteHyperRectangle],
            targetMemLevel: str, ctxt: NetworkContext,
            operatorRepresentation: OperatorRepresentation) -> Tuple[VariableReplacementScheme, TilingSchedule]:

        output_cubes = [cube.rectangle for cube in absoluteOutputCubes]

        addr_names = ['data_in']
        input_base_offsets, output_base_offsets = cls.extractBaseAddr(tilingSolution, targetMemLevel,
                                                                       operatorRepresentation, addr_names)

        replacements = {"C": []}
        replacement_types = {"C": PointerClass(uint16_t)}

        input_load_schedule = []
        output_load_schedule = []

        for cube in output_cubes:
            # cube is the tile of saved_mean: (C_tile,)
            C_tile = cube.dims[0]
            c_start = cube.offset[0]

            replacements["C"].append(C_tile)

            input_shape = ctxt.lookup(operatorRepresentation['data_in']).shape
            N = input_shape[0]
            H_in = input_shape[2]
            W_in = input_shape[3]
            data_in_cube = HyperRectangle((0, c_start, 0, 0), (N, C_tile, H_in, W_in))

            input_load_schedule.append({"data_in": data_in_cube})
            output_load_schedule.append({"saved_mean": cube})

        tiling_schedule = TilingSchedule(input_base_offsets, output_base_offsets, input_load_schedule,
                                          output_load_schedule)
        variable_replacement_schedule = VariableReplacementScheme(replacements, replacement_types)
        return variable_replacement_schedule, tiling_schedule


class ChannelNormalizeTileConstraint(TileConstraint):
    """Tile constraint for ChannelNormalize (split BN forward elementwise).

    Inputs:  X[N,C,H,W], saved_mean[C], saved_inv_std[C], gamma[C], beta[C]
    Outputs: Y[N,C,H,W]

    Tiling: C, H, W are all free (elementwise per-channel op).
    """

    @staticmethod
    def addGeometricalConstraint(tilerModel: TilerModel, parseDict: Dict, ctxt: NetworkContext) -> TilerModel:
        data_in_name = parseDict['data_in']
        data_out_name = parseDict['data_out']
        saved_mean_name = parseDict['saved_mean']
        saved_inv_std_name = parseDict['saved_inv_std']
        gamma_name = parseDict['gamma']
        beta_name = parseDict['beta']

        for name in [data_in_name, data_out_name, saved_mean_name, saved_inv_std_name, gamma_name, beta_name]:
            tilerModel.addTensorDimToModel(ctxt, name)

        # data_out has the same shape as data_in
        for idx in range(4):
            tilerModel.addConstraint(
                tilerModel.getTensorDimVar(tensorName=data_in_name, dimIdx=idx) ==
                tilerModel.getTensorDimVar(tensorName=data_out_name, dimIdx=idx))

        # Per-channel vectors follow C (dim 1)
        for vec_name in [saved_mean_name, saved_inv_std_name, gamma_name, beta_name]:
            tilerModel.addConstraint(
                tilerModel.getTensorDimVar(tensorName=vec_name, dimIdx=0) ==
                tilerModel.getTensorDimVar(tensorName=data_in_name, dimIdx=1))

        return tilerModel

    @classmethod
    def serializeTilingSolution(
            cls, tilingSolution: NodeMemoryConstraint, absoluteOutputCubes: List[AbsoluteHyperRectangle],
            targetMemLevel: str, ctxt: NetworkContext,
            operatorRepresentation: OperatorRepresentation) -> Tuple[VariableReplacementScheme, TilingSchedule]:

        output_cubes = [cube.rectangle for cube in absoluteOutputCubes]

        saved_mean_in_solution = operatorRepresentation['saved_mean'] in tilingSolution.tensorMemoryConstraints
        saved_inv_std_in_solution = operatorRepresentation['saved_inv_std'] in tilingSolution.tensorMemoryConstraints
        gamma_in_solution = operatorRepresentation['gamma'] in tilingSolution.tensorMemoryConstraints
        beta_in_solution = operatorRepresentation['beta'] in tilingSolution.tensorMemoryConstraints

        addr_names = ['data_in', 'data_out']
        if saved_mean_in_solution:
            addr_names.append('saved_mean')
        if saved_inv_std_in_solution:
            addr_names.append('saved_inv_std')
        if gamma_in_solution:
            addr_names.append('gamma')
        if beta_in_solution:
            addr_names.append('beta')

        input_base_offsets, output_base_offsets = cls.extractBaseAddr(tilingSolution, targetMemLevel,
                                                                       operatorRepresentation, addr_names)

        replacements = {"C": [], "H_in": [], "W_in": []}
        replacement_types = {"C": PointerClass(uint16_t), "H_in": PointerClass(uint16_t),
                             "W_in": PointerClass(uint16_t)}

        input_load_schedule = []
        output_load_schedule = []

        for cube in output_cubes:
            C_tile = cube.dims[1]
            H_tile = cube.dims[2]
            W_tile = cube.dims[3]
            c_start = cube.offset[1]

            replacements["C"].append(C_tile)
            replacements["H_in"].append(H_tile)
            replacements["W_in"].append(W_tile)

            vec_cube = HyperRectangle((c_start,), (C_tile,))

            entry = {"data_in": cube}
            if saved_mean_in_solution:
                entry["saved_mean"] = vec_cube
            if saved_inv_std_in_solution:
                entry["saved_inv_std"] = vec_cube
            if gamma_in_solution:
                entry["gamma"] = vec_cube
            if beta_in_solution:
                entry["beta"] = vec_cube
            input_load_schedule.append(entry)
            output_load_schedule.append({"data_out": cube})

        tiling_schedule = TilingSchedule(input_base_offsets, output_base_offsets, input_load_schedule,
                                          output_load_schedule)
        variable_replacement_schedule = VariableReplacementScheme(replacements, replacement_types)
        return variable_replacement_schedule, tiling_schedule


class BNGradReduceTileConstraint(TileConstraint):
    """Tile constraint for BNGradReduce (split BN backward reduction).

    Inputs:  dY[N,C,H,W], X[N,C,H,W], saved_mean[C], saved_inv_std[C]
    Outputs: dgamma[C], dbeta[C]

    Tiling: C is free; N, H, W are pinned (reduction over spatial).
    """

    @staticmethod
    def addGeometricalConstraint(tilerModel: TilerModel, parseDict: Dict, ctxt: NetworkContext) -> TilerModel:
        dY_name = parseDict['dY']
        X_name = parseDict['X']
        saved_mean_name = parseDict['saved_mean']
        saved_inv_std_name = parseDict['saved_inv_std']
        dgamma_name = parseDict['dgamma']
        dbeta_name = parseDict['dbeta']

        for name in [dY_name, X_name, saved_mean_name, saved_inv_std_name, dgamma_name, dbeta_name]:
            tilerModel.addTensorDimToModel(ctxt, name)

        input_shape = ctxt.lookup(dY_name).shape
        N = input_shape[0]
        H_in = input_shape[2]
        W_in = input_shape[3]

        # Pin N, H, W
        tilerModel.addConstraint(tilerModel.getTensorDimVar(tensorName=dY_name, dimIdx=0) == N)
        tilerModel.addConstraint(tilerModel.getTensorDimVar(tensorName=dY_name, dimIdx=2) == H_in)
        tilerModel.addConstraint(tilerModel.getTensorDimVar(tensorName=dY_name, dimIdx=3) == W_in)

        # X has the same shape as dY
        for idx in range(4):
            tilerModel.addConstraint(
                tilerModel.getTensorDimVar(tensorName=dY_name, dimIdx=idx) ==
                tilerModel.getTensorDimVar(tensorName=X_name, dimIdx=idx))

        # Per-channel vectors follow C
        for vec_name in [saved_mean_name, saved_inv_std_name, dgamma_name, dbeta_name]:
            tilerModel.addConstraint(
                tilerModel.getTensorDimVar(tensorName=vec_name, dimIdx=0) ==
                tilerModel.getTensorDimVar(tensorName=dY_name, dimIdx=1))

        return tilerModel

    @classmethod
    def wrapTilingSolution(
            cls, tilingSolution: NodeMemoryConstraint, targetMemLevel: str, ctxt: NetworkContext,
            operatorRepresentation: OperatorRepresentation) -> Tuple[VariableReplacementScheme, List[TilingSchedule]]:

        dgammaVar = operatorRepresentation['dgamma']
        singleOutputSolution = copy.deepcopy(tilingSolution)
        singleOutputSolution.outputTensorMemoryConstraints = {
            dgammaVar: tilingSolution.outputTensorMemoryConstraints[dgammaVar]
        }

        varReplacement, tilingSchedules = super().wrapTilingSolution(singleOutputSolution, targetMemLevel, ctxt,
                                                                      operatorRepresentation)

        secondaryVar = operatorRepresentation['dbeta']
        if secondaryVar in tilingSolution.outputTensorMemoryConstraints:
            addr = TileConstraint.getBaseAddr(tilingSolution, targetMemLevel, secondaryVar)
            if addr != [None]:
                for schedule in tilingSchedules:
                    schedule.outputBaseOffsets['dbeta'] = addr
                    for step in schedule.outputLoadSchedule:
                        dgamma_rect = step['dgamma']
                        step['dbeta'] = HyperRectangle(dgamma_rect.offset, dgamma_rect.dims)

        return varReplacement, tilingSchedules

    @classmethod
    def serializeTilingSolution(
            cls, tilingSolution: NodeMemoryConstraint, absoluteOutputCubes: List[AbsoluteHyperRectangle],
            targetMemLevel: str, ctxt: NetworkContext,
            operatorRepresentation: OperatorRepresentation) -> Tuple[VariableReplacementScheme, TilingSchedule]:

        output_cubes = [cube.rectangle for cube in absoluteOutputCubes]

        saved_mean_in_solution = operatorRepresentation['saved_mean'] in tilingSolution.tensorMemoryConstraints
        saved_inv_std_in_solution = operatorRepresentation['saved_inv_std'] in tilingSolution.tensorMemoryConstraints

        addr_names = ['dY', 'X']
        if saved_mean_in_solution:
            addr_names.append('saved_mean')
        if saved_inv_std_in_solution:
            addr_names.append('saved_inv_std')

        input_base_offsets, output_base_offsets = cls.extractBaseAddr(tilingSolution, targetMemLevel,
                                                                       operatorRepresentation, addr_names)

        replacements = {"C": []}
        replacement_types = {"C": PointerClass(uint16_t)}

        input_load_schedule = []
        output_load_schedule = []

        for cube in output_cubes:
            # cube is the tile of dgamma: (C_tile,)
            C_tile = cube.dims[0]
            c_start = cube.offset[0]

            replacements["C"].append(C_tile)

            input_shape = ctxt.lookup(operatorRepresentation['dY']).shape
            N = input_shape[0]
            H_in = input_shape[2]
            W_in = input_shape[3]
            data_cube = HyperRectangle((0, c_start, 0, 0), (N, C_tile, H_in, W_in))
            vec_cube = HyperRectangle((c_start,), (C_tile,))

            entry = {"dY": data_cube, "X": data_cube}
            if saved_mean_in_solution:
                entry["saved_mean"] = vec_cube
            if saved_inv_std_in_solution:
                entry["saved_inv_std"] = vec_cube
            input_load_schedule.append(entry)
            output_load_schedule.append({"dgamma": cube})

        tiling_schedule = TilingSchedule(input_base_offsets, output_base_offsets, input_load_schedule,
                                          output_load_schedule)
        variable_replacement_schedule = VariableReplacementScheme(replacements, replacement_types)
        return variable_replacement_schedule, tiling_schedule


class BNGradNormalizeTileConstraint(TileConstraint):
    """Tile constraint for BNGradNormalize (split BN backward elementwise).

    Inputs:  dY[N,C,H,W], X[N,C,H,W], saved_mean[C], saved_inv_std[C],
             gamma[C], dgamma[C], dbeta[C]
    Outputs: dX[N,C,H,W]

    Tiling: C, H, W are all free (elementwise per-channel op).
    """

    @staticmethod
    def addGeometricalConstraint(tilerModel: TilerModel, parseDict: Dict, ctxt: NetworkContext) -> TilerModel:
        dY_name = parseDict['dY']
        X_name = parseDict['X']
        saved_mean_name = parseDict['saved_mean']
        saved_inv_std_name = parseDict['saved_inv_std']
        gamma_name = parseDict['gamma']
        dgamma_name = parseDict['dgamma']
        dbeta_name = parseDict['dbeta']
        dX_name = parseDict['dX']

        for name in [dY_name, X_name, saved_mean_name, saved_inv_std_name, gamma_name, dgamma_name, dbeta_name,
                     dX_name]:
            tilerModel.addTensorDimToModel(ctxt, name)

        # dY, X, dX must have the same shape
        for idx in range(4):
            tilerModel.addConstraint(
                tilerModel.getTensorDimVar(tensorName=dY_name, dimIdx=idx) ==
                tilerModel.getTensorDimVar(tensorName=X_name, dimIdx=idx))
            tilerModel.addConstraint(
                tilerModel.getTensorDimVar(tensorName=dY_name, dimIdx=idx) ==
                tilerModel.getTensorDimVar(tensorName=dX_name, dimIdx=idx))

        # Per-channel vectors follow C (dim 1)
        for vec_name in [saved_mean_name, saved_inv_std_name, gamma_name, dgamma_name, dbeta_name]:
            tilerModel.addConstraint(
                tilerModel.getTensorDimVar(tensorName=vec_name, dimIdx=0) ==
                tilerModel.getTensorDimVar(tensorName=dY_name, dimIdx=1))

        return tilerModel

    @classmethod
    def serializeTilingSolution(
            cls, tilingSolution: NodeMemoryConstraint, absoluteOutputCubes: List[AbsoluteHyperRectangle],
            targetMemLevel: str, ctxt: NetworkContext,
            operatorRepresentation: OperatorRepresentation) -> Tuple[VariableReplacementScheme, TilingSchedule]:

        output_cubes = [cube.rectangle for cube in absoluteOutputCubes]

        saved_mean_in_solution = operatorRepresentation['saved_mean'] in tilingSolution.tensorMemoryConstraints
        saved_inv_std_in_solution = operatorRepresentation['saved_inv_std'] in tilingSolution.tensorMemoryConstraints
        gamma_in_solution = operatorRepresentation['gamma'] in tilingSolution.tensorMemoryConstraints
        dgamma_in_solution = operatorRepresentation['dgamma'] in tilingSolution.tensorMemoryConstraints
        dbeta_in_solution = operatorRepresentation['dbeta'] in tilingSolution.tensorMemoryConstraints

        addr_names = ['dY', 'X', 'dX']
        if saved_mean_in_solution:
            addr_names.append('saved_mean')
        if saved_inv_std_in_solution:
            addr_names.append('saved_inv_std')
        if gamma_in_solution:
            addr_names.append('gamma')
        if dgamma_in_solution:
            addr_names.append('dgamma')
        if dbeta_in_solution:
            addr_names.append('dbeta')

        input_base_offsets, output_base_offsets = cls.extractBaseAddr(tilingSolution, targetMemLevel,
                                                                       operatorRepresentation, addr_names)

        replacements = {"C": [], "H_in": [], "W_in": []}
        replacement_types = {"C": PointerClass(uint16_t), "H_in": PointerClass(uint16_t),
                             "W_in": PointerClass(uint16_t)}

        input_load_schedule = []
        output_load_schedule = []

        for cube in output_cubes:
            C_tile = cube.dims[1]
            H_tile = cube.dims[2]
            W_tile = cube.dims[3]
            c_start = cube.offset[1]

            replacements["C"].append(C_tile)
            replacements["H_in"].append(H_tile)
            replacements["W_in"].append(W_tile)

            vec_cube = HyperRectangle((c_start,), (C_tile,))

            entry = {"dY": cube, "X": cube}
            if saved_mean_in_solution:
                entry["saved_mean"] = vec_cube
            if saved_inv_std_in_solution:
                entry["saved_inv_std"] = vec_cube
            if gamma_in_solution:
                entry["gamma"] = vec_cube
            if dgamma_in_solution:
                entry["dgamma"] = vec_cube
            if dbeta_in_solution:
                entry["dbeta"] = vec_cube
            input_load_schedule.append(entry)
            output_load_schedule.append({"dX": cube})

        tiling_schedule = TilingSchedule(input_base_offsets, output_base_offsets, input_load_schedule,
                                          output_load_schedule)
        variable_replacement_schedule = VariableReplacementScheme(replacements, replacement_types)
        return variable_replacement_schedule, tiling_schedule


class BatchNormalizationGradTileConstraint(TileConstraint):
    """Tile constraint for BatchNormalizationGrad (BN backward pass).

    Inputs:  dY[N,C,H,W], X[N,C,H,W], gamma[C], saved_mean[C], saved_inv_std[C]
    Outputs: dX[N,C,H,W] (primary), dgamma[C], dbeta[C] (secondary)

    Tiling strategy: tile along C (channels are independent).
      - N, H_in, W_in are pinned to full size: backward BN needs all N*H*W
        elements per channel for dgamma/dbeta reductions.
      - C is free: each channel's gradient is computed independently.
    """

    @staticmethod
    def addGeometricalConstraint(tilerModel: TilerModel, parseDict: Dict, ctxt: NetworkContext) -> TilerModel:
        dY_name = parseDict['dY']
        X_name = parseDict['X']
        gamma_name = parseDict['gamma']
        saved_mean_name = parseDict['saved_mean']
        saved_inv_std_name = parseDict['saved_inv_std']
        dX_name = parseDict['dX']
        dgamma_name = parseDict['dgamma']
        dbeta_name = parseDict['dbeta']

        for name in [dY_name, X_name, gamma_name, saved_mean_name, saved_inv_std_name, dX_name, dgamma_name,
                     dbeta_name]:
            tilerModel.addTensorDimToModel(ctxt, name)

        input_shape = ctxt.lookup(dY_name).shape
        N = input_shape[0]
        H_in = input_shape[2]
        W_in = input_shape[3]

        # Pin N, H_in, W_in: backward BN needs all spatial/batch elements for reductions
        tilerModel.addConstraint(tilerModel.getTensorDimVar(tensorName=dY_name, dimIdx=0) == N)
        tilerModel.addConstraint(tilerModel.getTensorDimVar(tensorName=dY_name, dimIdx=2) == H_in)
        tilerModel.addConstraint(tilerModel.getTensorDimVar(tensorName=dY_name, dimIdx=3) == W_in)

        # X, dX must have the same shape as dY
        for idx in range(4):
            tilerModel.addConstraint(
                tilerModel.getTensorDimVar(tensorName=dY_name, dimIdx=idx) ==
                tilerModel.getTensorDimVar(tensorName=X_name, dimIdx=idx))
            tilerModel.addConstraint(
                tilerModel.getTensorDimVar(tensorName=dY_name, dimIdx=idx) ==
                tilerModel.getTensorDimVar(tensorName=dX_name, dimIdx=idx))

        # Per-channel vectors: single dimension follows C (dim 1 of dY)
        for vec_name in [gamma_name, saved_mean_name, saved_inv_std_name, dgamma_name, dbeta_name]:
            tilerModel.addConstraint(
                tilerModel.getTensorDimVar(tensorName=vec_name, dimIdx=0) ==
                tilerModel.getTensorDimVar(tensorName=dY_name, dimIdx=1))

        return tilerModel

    @classmethod
    def wrapTilingSolution(
            cls, tilingSolution: NodeMemoryConstraint, targetMemLevel: str, ctxt: NetworkContext,
            operatorRepresentation: OperatorRepresentation) -> Tuple[VariableReplacementScheme, List[TilingSchedule]]:

        dXVar = operatorRepresentation['dX']

        # Pass a single-output solution to satisfy the base-class assertion
        singleOutputSolution = copy.deepcopy(tilingSolution)
        singleOutputSolution.outputTensorMemoryConstraints = {
            dXVar: tilingSolution.outputTensorMemoryConstraints[dXVar]
        }

        varReplacement, tilingSchedules = super().wrapTilingSolution(singleOutputSolution, targetMemLevel, ctxt,
                                                                      operatorRepresentation)

        # Extend each schedule to include dgamma and dbeta outputs
        for secondary in ['dgamma', 'dbeta']:
            secondaryVar = operatorRepresentation.get(secondary, '')
            if not secondaryVar:
                continue
            if secondaryVar not in tilingSolution.outputTensorMemoryConstraints:
                continue
            addr = TileConstraint.getBaseAddr(tilingSolution, targetMemLevel, secondaryVar)
            if addr == [None]:
                continue
            for schedule in tilingSchedules:
                schedule.outputBaseOffsets[secondary] = addr
                for step in schedule.outputLoadSchedule:
                    dX_rect = step['dX']
                    c_start = dX_rect.offset[1]
                    c_tile = dX_rect.dims[1]
                    step[secondary] = HyperRectangle((c_start,), (c_tile,))

        return varReplacement, tilingSchedules

    @classmethod
    def serializeTilingSolution(
            cls, tilingSolution: NodeMemoryConstraint, absoluteOutputCubes: List[AbsoluteHyperRectangle],
            targetMemLevel: str, ctxt: NetworkContext,
            operatorRepresentation: OperatorRepresentation) -> Tuple[VariableReplacementScheme, TilingSchedule]:

        output_cubes = [cube.rectangle for cube in absoluteOutputCubes]

        # gamma is a learnable model parameter (global) and may be excluded from the tiling
        # solution by _checkResolve. saved_mean/saved_inv_std are transient BN forward outputs
        # and are normally in the solution, but guard them too for robustness.
        gamma_in_solution = operatorRepresentation['gamma'] in tilingSolution.tensorMemoryConstraints
        saved_mean_in_solution = operatorRepresentation['saved_mean'] in tilingSolution.tensorMemoryConstraints
        saved_inv_std_in_solution = operatorRepresentation['saved_inv_std'] in tilingSolution.tensorMemoryConstraints

        addr_names = ['dY', 'X', 'dX']
        if gamma_in_solution:
            addr_names.append('gamma')
        if saved_mean_in_solution:
            addr_names.append('saved_mean')
        if saved_inv_std_in_solution:
            addr_names.append('saved_inv_std')

        input_base_offsets, output_base_offsets = cls.extractBaseAddr(tilingSolution, targetMemLevel,
                                                                       operatorRepresentation, addr_names)

        replacements = {"C": []}
        replacement_types = {"C": PointerClass(uint16_t)}

        input_load_schedule = []
        output_load_schedule = []

        for cube in output_cubes:
            # cube is the tile of dX: [N, C_tile, H_in, W_in]
            C_tile = cube.dims[1]
            c_start = cube.offset[1]

            replacements["C"].append(C_tile)

            vec_cube = HyperRectangle((c_start,), (C_tile,))

            entry = {"dY": cube, "X": cube}
            if gamma_in_solution:
                entry["gamma"] = vec_cube
            if saved_mean_in_solution:
                entry["saved_mean"] = vec_cube
            if saved_inv_std_in_solution:
                entry["saved_inv_std"] = vec_cube
            input_load_schedule.append(entry)
            output_load_schedule.append({"dX": cube})

        tiling_schedule = TilingSchedule(input_base_offsets, output_base_offsets, input_load_schedule,
                                          output_load_schedule)
        variable_replacement_schedule = VariableReplacementScheme(replacements, replacement_types)
        return variable_replacement_schedule, tiling_schedule

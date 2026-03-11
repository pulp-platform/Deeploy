# SPDX-FileCopyrightText: 2024 ETH Zurich and University of Bologna
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


class LayernormTileConstraint(TileConstraint):

    @staticmethod
    def addGeometricalConstraint(tilerModel: TilerModel, parseDict: Dict, ctxt: NetworkContext) -> TilerModel:
        inputBufferName = parseDict['data_in']
        outputBufferName = parseDict['data_out']
        scaleBufferName = parseDict['weight']
        biasBufferName = parseDict['bias']

        for bufferName in [inputBufferName, outputBufferName, scaleBufferName, biasBufferName]:
            tilerModel.addTensorDimToModel(ctxt, bufferName)

        inputShape = ctxt.lookup(inputBufferName).shape
        lastDimIdx = len(inputShape) - 1
        lastDimLen = inputShape[-1]

        tilerModel.addConstraint(
            tilerModel.getTensorDimVar(tensorName = inputBufferName, dimIdx = lastDimIdx) == lastDimLen)
        tilerModel.addConstraint(
            tilerModel.getTensorDimVar(tensorName = inputBufferName, dimIdx = lastDimIdx) == tilerModel.getTensorDimVar(
                tensorName = scaleBufferName, dimIdx = 0))
        tilerModel.addConstraint(
            tilerModel.getTensorDimVar(tensorName = inputBufferName, dimIdx = lastDimIdx) == tilerModel.getTensorDimVar(
                tensorName = biasBufferName, dimIdx = 0))

        for idx, dim in enumerate(inputShape):
            tilerModel.addConstraint(
                tilerModel.getTensorDimVar(tensorName = inputBufferName, dimIdx = idx) == tilerModel.getTensorDimVar(
                    tensorName = outputBufferName, dimIdx = idx))

        # Register mean/inv_std_dev (secondary outputs, shape = inputShape[:-1])
        # They tile along all dims except features, so constrain them to match data_in.
        for secondary in ['mean', 'inv_std_dev']:
            secondary_name = parseDict.get(secondary, '')
            if secondary_name:
                tilerModel.addTensorDimToModel(ctxt, secondary_name)
                for idx in range(len(inputShape) - 1):
                    tilerModel.addConstraint(
                        tilerModel.getTensorDimVar(tensorName = secondary_name, dimIdx = idx) ==
                        tilerModel.getTensorDimVar(tensorName = inputBufferName, dimIdx = idx))

        return tilerModel

    @classmethod
    def wrapTilingSolution(
            cls, tilingSolution: NodeMemoryConstraint, targetMemLevel: str, ctxt: NetworkContext,
            operatorRepresentation: OperatorRepresentation) -> Tuple[VariableReplacementScheme, List[TilingSchedule]]:

        dataOutVar = operatorRepresentation['data_out']

        # Build a single-output copy to bypass the base-class assertion
        # that len(outputTensorMemoryConstraints) == 1.
        singleOutputSolution = copy.deepcopy(tilingSolution)
        singleOutputSolution.outputTensorMemoryConstraints = {
            dataOutVar: tilingSolution.outputTensorMemoryConstraints[dataOutVar]
        }

        varReplacement, tilingSchedules = super().wrapTilingSolution(singleOutputSolution, targetMemLevel, ctxt,
                                                                      operatorRepresentation)

        # Extend each tiling schedule to include mean and inv_std_dev outputs.
        # Their tile rectangles are derived from data_out by dropping the features dim.
        for secondary in ['mean', 'inv_std_dev']:
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
                    # mean/inv_std_dev: drop the last (features) dim from data_out tile
                    step[secondary] = HyperRectangle(data_out_rect.offset[:-1], data_out_rect.dims[:-1])

        return varReplacement, tilingSchedules

    @classmethod
    def serializeTilingSolution(
            cls, tilingSolution: NodeMemoryConstraint, absoluteOutputCubes: List[AbsoluteHyperRectangle],
            targetMemLevel: str, ctxt: NetworkContext,
            operatorRepresentation: OperatorRepresentation) -> Tuple[VariableReplacementScheme, TilingSchedule]:

        outputCubes = [cube.rectangle for cube in absoluteOutputCubes]
        addrNames = ['data_in', 'data_out', 'weight', 'bias']
        inputBaseOffsets, outputBaseOffsets = cls.extractBaseAddr(tilingSolution, targetMemLevel,
                                                                  operatorRepresentation, addrNames)

        replacements = {"size": []}

        replacementTypes = {"size": PointerClass(uint16_t)}

        inputLoadSchedule = []
        outputLoadSchedule = []

        for cube in outputCubes:
            newSize = np.prod(cube.dims)
            replacements["size"].append(newSize)
            weightCube = HyperRectangle((0,), (cube.dims[-1],))
            biasCube = HyperRectangle((0,), (cube.dims[-1],))
            inputLoadSchedule.append({"data_in": cube, "weight": weightCube, "bias": biasCube})
            outputLoadSchedule.append({"data_out": cube})

        tilingSchedule = TilingSchedule(inputBaseOffsets, outputBaseOffsets, inputLoadSchedule, outputLoadSchedule)
        variableReplacementSchedule = VariableReplacementScheme(replacements, replacementTypes)

        return variableReplacementSchedule, tilingSchedule


class LayernormGradTileConstraint(TileConstraint):

    @staticmethod
    def addGeometricalConstraint(tilerModel: TilerModel, parseDict: Dict, ctxt: NetworkContext) -> TilerModel:
        grad_in_buffer_name = parseDict['grad_in']
        data_in_buffer_name = parseDict['data_in']
        weight_buffer_name = parseDict['weight']
        grad_out_buffer_name = parseDict['grad_out']

        for buffer_name in [grad_in_buffer_name, data_in_buffer_name, weight_buffer_name, grad_out_buffer_name]:
            tilerModel.addTensorDimToModel(ctxt, buffer_name)

        input_shape = ctxt.lookup(data_in_buffer_name).shape
        last_dim_idx = len(input_shape) - 1
        last_dim_len = input_shape[-1]

        tilerModel.addConstraint(
            tilerModel.getTensorDimVar(tensorName = data_in_buffer_name, dimIdx = last_dim_idx) == last_dim_len)

        tilerModel.addConstraint(
            tilerModel.getTensorDimVar(tensorName = data_in_buffer_name, dimIdx = last_dim_idx) ==
            tilerModel.getTensorDimVar(tensorName = weight_buffer_name, dimIdx = 0))

        for idx, dim in enumerate(input_shape):
            tilerModel.addConstraint(
                tilerModel.getTensorDimVar(tensorName = data_in_buffer_name, dimIdx = idx) ==
                tilerModel.getTensorDimVar(tensorName = grad_in_buffer_name, dimIdx = idx))

        for idx, dim in enumerate(input_shape):
            tilerModel.addConstraint(
                tilerModel.getTensorDimVar(tensorName = data_in_buffer_name, dimIdx = idx) ==
                tilerModel.getTensorDimVar(tensorName = grad_out_buffer_name, dimIdx = idx))

        # Register mean/inv_std_dev inputs (shape = inputShape[:-1]).
        for secondary in ['mean', 'inv_std_dev']:
            secondary_name = parseDict.get(secondary, '')
            if secondary_name:
                tilerModel.addTensorDimToModel(ctxt, secondary_name)
                for idx in range(len(input_shape) - 1):
                    tilerModel.addConstraint(
                        tilerModel.getTensorDimVar(tensorName = secondary_name, dimIdx = idx) ==
                        tilerModel.getTensorDimVar(tensorName = data_in_buffer_name, dimIdx = idx))

        # Register weight_grad/bias_grad (secondary outputs, shape = [features]).
        # Their single dimension (features) is already pinned to full size via last_dim_len above.
        for secondary in ['weight_grad', 'bias_grad']:
            secondary_name = parseDict.get(secondary, '')
            if secondary_name:
                tilerModel.addTensorDimToModel(ctxt, secondary_name)
                tilerModel.addConstraint(
                    tilerModel.getTensorDimVar(tensorName = secondary_name, dimIdx = 0) == last_dim_len)

        return tilerModel

    @staticmethod
    def addPolicyConstraint(tilerModel: TilerModel, parseDict: Dict, ctxt: NetworkContext) -> TilerModel:
        # Only pin the feature (last) dim — already done in addGeometricalConstraint.
        # Seq dims are left free so the solver can tile along the sequence dimension.
        # weight_grad/bias_grad accumulation across seq tiles is handled in the template
        # via a static-flag memset + inline accumulation loop (ConvGradW pattern).
        return tilerModel

    @classmethod
    def wrapTilingSolution(
            cls, tilingSolution: NodeMemoryConstraint, targetMemLevel: str, ctxt: NetworkContext,
            operatorRepresentation: OperatorRepresentation) -> Tuple[VariableReplacementScheme, List[TilingSchedule]]:

        gradOutVar = operatorRepresentation['grad_out']

        # Build a single-output copy to bypass the base-class assertion
        # that len(outputTensorMemoryConstraints) == 1.
        singleOutputSolution = copy.deepcopy(tilingSolution)
        singleOutputSolution.outputTensorMemoryConstraints = {
            gradOutVar: tilingSolution.outputTensorMemoryConstraints[gradOutVar]
        }

        varReplacement, tilingSchedules = super().wrapTilingSolution(singleOutputSolution, targetMemLevel, ctxt,
                                                                      operatorRepresentation)

        # Extend each tiling schedule to include weight_grad and bias_grad outputs.
        # Since batch is pinned to full size (addPolicyConstraint), there is effectively
        # one tile step and these are always full-size tensors.
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
        addr_names = ['grad_in', 'data_in', 'weight', 'mean', 'inv_std_dev', 'grad_out']
        input_base_offsets, output_base_offsets = cls.extractBaseAddr(tilingSolution, targetMemLevel,
                                                                      operatorRepresentation, addr_names)

        replacements = {"size": []}
        replacement_types = {"size": PointerClass(uint16_t)}

        input_load_schedule = []
        output_load_schedule = []

        for cube in output_cubes:
            new_size = np.prod(cube.dims)
            replacements["size"].append(new_size)

            feature_size = cube.dims[-1]
            seq_dims = cube.dims[:-1]
            seq_offset = cube.offset[:-1] if len(cube.offset) > 1 else (0,)

            weight_cube = HyperRectangle((0,), (feature_size,))
            mean_cube = HyperRectangle(seq_offset, seq_dims)
            inv_std_dev_cube = HyperRectangle(seq_offset, seq_dims)

            input_load_schedule.append({
                "grad_in": cube,
                "data_in": cube,
                "weight": weight_cube,
                "mean": mean_cube,
                "inv_std_dev": inv_std_dev_cube,
            })

            output_load_schedule.append({"grad_out": cube})

        tiling_schedule = TilingSchedule(input_base_offsets, output_base_offsets, input_load_schedule,
                                         output_load_schedule)
        variable_replacement_schedule = VariableReplacementScheme(replacements, replacement_types)

        return variable_replacement_schedule, tiling_schedule

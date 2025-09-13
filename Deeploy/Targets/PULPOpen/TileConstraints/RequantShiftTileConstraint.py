# Copyright (C) 2025, ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
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


class RequantShiftTileConstraint(TileConstraint):

    @staticmethod
    def addGeometricalConstraint(tilerModel: TilerModel, parseDict: Dict, ctxt: NetworkContext) -> TilerModel:

        inputBufferName = parseDict['data_in']
        mulBufferName = parseDict['mul']
        addBufferName = parseDict['add']
        outputBufferName = parseDict['data_out']

        # Add I/O dimensions to the model as variables
        for bufferName in [inputBufferName, mulBufferName, addBufferName, outputBufferName]:
            tilerModel.addTensorDimToModel(ctxt, bufferName)

        inputShape = ctxt.lookup(inputBufferName).shape

        mulBufferShapeLen = len(ctxt.lookup(mulBufferName).shape)
        addBufferShapeLen = len(ctxt.lookup(addBufferName).shape)

        mulChannelVar = tilerModel.getTensorDimVar(tensorName = mulBufferName, dimIdx = mulBufferShapeLen - 1)
        addChannelVar = tilerModel.getTensorDimVar(tensorName = addBufferName, dimIdx = addBufferShapeLen - 1)

        tilerModel.addConstraint(mulChannelVar == addChannelVar)

        channels_first = parseDict['channels_first']
        if not channels_first:
            inChannelVar = tilerModel.getTensorDimVar(tensorName = inputBufferName, dimIdx = len(inputShape) - 1)
        else:
            inChannelVar = tilerModel.getTensorDimVar(tensorName = inputBufferName, dimIdx = 1)

        tilerModel.addConstraint(mulChannelVar == inChannelVar)

        for dim in range(len(inputShape)):
            inputDimVar = tilerModel.getTensorDimVar(tensorName = inputBufferName, dimIdx = dim)
            outputDimVar = tilerModel.getTensorDimVar(tensorName = outputBufferName, dimIdx = dim)
            tilerModel.addConstraint(inputDimVar == outputDimVar)  # Batch

        return tilerModel

    @classmethod
    def serializeTilingSolution(
            cls, tilingSolution: NodeMemoryConstraint, absoluteOutputCubes: List[AbsoluteHyperRectangle],
            targetMemLevel: str, ctxt: NetworkContext,
            operatorRepresentation: OperatorRepresentation) -> Tuple[VariableReplacementScheme, TilingSchedule]:
        outputCubes = [cube.rectangle for cube in absoluteOutputCubes]

        addrNames = ['data_in', 'mul', 'add', 'data_out']
        inputBaseOffsets, outputBaseOffsets = cls.extractBaseAddr(tilingSolution, targetMemLevel,
                                                                  operatorRepresentation, addrNames)

        inputCubes = outputCubes

        rqCubes = []

        replacements = {"size": [], "channel_width": [], "channels": []}
        replacementTypes = {
            "size": PointerClass(uint16_t),
            "channel_width": PointerClass(uint16_t),
            "channels": PointerClass(uint16_t)
        }

        for cube in inputCubes:

            if operatorRepresentation['channels_first']:
                rqCube = HyperRectangle((cube.offset[1],), (cube.dims[1],))
                channelDim = cube.dims[1]
            else:
                rqCube = HyperRectangle((cube.offset[-1],), (cube.dims[-1],))
                channelDim = cube.dims[-1]

            rqCubes.append(rqCube)

            size = np.prod(cube.dims[1:])
            channelWidth = size // channelDim
            channels = channelDim

            replacements['size'].append(size)
            replacements['channel_width'].append(channelWidth)
            replacements['channels'].append(channels)

        inputLoadSchedule = []
        outputLoadSchedule = []

        for a, rq in zip(inputCubes, rqCubes):
            inputLoadSchedule.append({"data_in": a, "add": rq, "mul": rq})

        for out in outputCubes:
            outputLoadSchedule.append({"data_out": out})

        tilingSchedule = TilingSchedule(inputBaseOffsets, outputBaseOffsets, inputLoadSchedule, outputLoadSchedule)
        variableReplacementSchedule = VariableReplacementScheme(replacements, replacementTypes)

        return variableReplacementSchedule, tilingSchedule

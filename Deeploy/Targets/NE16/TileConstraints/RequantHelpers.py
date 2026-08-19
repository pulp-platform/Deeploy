# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, List

from Deeploy.DeeployTypes import NetworkContext, OperatorRepresentation
from Deeploy.TilingExtension.TilerModel import TilerModel
from Deeploy.TilingExtension.TilingCodegen import AbsoluteHyperRectangle, HyperRectangle


def requantAddGeometricalConstraint(tilerModel: TilerModel, operatorRepresentation: OperatorRepresentation,
                                    ctxt: NetworkContext) -> TilerModel:
    outputBufferName = operatorRepresentation['data_out']
    mulBufferName = operatorRepresentation['mul']
    addBufferName = operatorRepresentation['add']

    # Add I/O dimensions to the model as variables
    for bufferName in [mulBufferName, addBufferName]:
        tilerModel.addTensorDimToModel(ctxt, bufferName)

    outputChannelVar = tilerModel.getTensorDimVar(tensorName = outputBufferName, dimIdx = 3)

    addBuffer = ctxt.lookup(addBufferName)
    addChannelVar = tilerModel.getTensorDimVar(tensorName = addBufferName, dimIdx = len(addBuffer.shape) - 1)
    mulBuffer = ctxt.lookup(mulBufferName)
    mulChannelVar = tilerModel.getTensorDimVar(tensorName = mulBufferName, dimIdx = len(mulBuffer.shape) - 1)

    tilerModel.addConstraint(outputChannelVar == addChannelVar)
    tilerModel.addConstraint(outputChannelVar == mulChannelVar)

    return tilerModel


def requantLoadSchedule(
    absoluteOutputCubes: List[AbsoluteHyperRectangle],
    ctxt: NetworkContext,
    operatorRepresentation: OperatorRepresentation,
) -> List[Dict[str, HyperRectangle]]:
    outputCubes = [cube.rectangle for cube in absoluteOutputCubes]

    shapeMul = ctxt.lookup(operatorRepresentation["mul"]).shape
    shapeAdd = ctxt.lookup(operatorRepresentation["add"]).shape

    schedule = []
    for cube in outputCubes:
        (_, _, _, COffset) = cube.offset
        (_, _, _, CSize) = cube.dims
        MulCube = HyperRectangle((0,) * (len(shapeMul) - 1) + (COffset,), (1,) * (len(shapeMul) - 1) + (CSize,))
        AddCube = HyperRectangle((0,) * (len(shapeAdd) - 1) + (COffset,), (1,) * (len(shapeAdd) - 1) + (CSize,))
        schedule.append({"mul": MulCube, "add": AddCube})

    return schedule

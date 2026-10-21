# SPDX-FileCopyrightText: 2023 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, List, Tuple, Union

import numpy as np
from ortools.constraint_solver.pywrapcp import IntVar

from Deeploy.AbstractDataTypes import PointerClass
from Deeploy.CommonExtensions.DataTypes import uint32_t
from Deeploy.DeeployTypes import NetworkContext, OperatorRepresentation
from Deeploy.TilingExtension.MemoryConstraints import NodeMemoryConstraint
from Deeploy.TilingExtension.TileConstraint import TileConstraint
from Deeploy.TilingExtension.TilerModel import TilerModel
from Deeploy.TilingExtension.TilingCodegen import AbsoluteHyperRectangle, HyperRectangle, TilingSchedule, \
    VariableReplacementScheme


class SliceTileConstraint(TileConstraint):

    @staticmethod
    def _resolveImmediate(value, ctxt: NetworkContext):
        if isinstance(value, str):
            return np.asarray(ctxt.lookup(value).values).astype(np.int64).reshape(-1)
        return np.asarray(value).astype(np.int64).reshape(-1)

    @classmethod
    def normalizedParameters(cls, parseDict: Dict, ctxt: NetworkContext):
        """Resolve constant names and normalize ONNX indices against the full input shape."""
        shape = parseDict['data_in_shape']
        axes, starts, ends, steps = [
            cls._resolveImmediate(parseDict[key], ctxt) for key in ('axes', 'starts', 'ends', 'steps')
        ]
        if not (len(axes) == len(starts) == len(ends) == len(steps)):
            raise ValueError("Slice parameter lengths must match")
        normalizedStarts = [0] * len(shape)
        normalizedSteps = [1] * len(shape)
        seen = set()
        for axis, start, end, step in zip(axes, starts, ends, steps):
            axis = int(axis)
            if axis < 0:
                axis += len(shape)
            if axis < 0 or axis >= len(shape) or axis in seen or step == 0:
                raise ValueError("Invalid Slice axis or step")
            seen.add(axis)
            start, end, step = slice(int(start), int(end), int(step)).indices(shape[axis])
            if not range(start, end, step):
                raise ValueError("Tiled Slice does not support empty outputs")
            normalizedStarts[axis] = start
            normalizedSteps[axis] = step
        return normalizedStarts, normalizedSteps

    @staticmethod
    def addGeometricalConstraint(tilerModel: TilerModel, parseDict: Dict, ctxt: NetworkContext) -> TilerModel:

        # ===== GET NECESSARY INFORMATION =====
        #   Get I/O buffer names
        inputBufferName = parseDict['data_in']
        outputBufferName = parseDict['data_out']

        #   Get I/O shapes
        inputShape = parseDict['data_in_shape']

        #   Get other necessary information
        _, sliceSteps = SliceTileConstraint.normalizedParameters(parseDict, ctxt)

        # ===== ADD I/O DIMENSIONS TO THE MODEL AS VARIABLES =====
        for bufferName in [inputBufferName, outputBufferName]:
            tilerModel.addTensorDimToModel(ctxt, bufferName)

        # ===== ADD CONSTRAINTS =====
        #   Add constraints for the I/O dimensions
        for idx in range(len(inputShape)):
            # Get current dimension variables
            inputDimensionVar = tilerModel.getTensorDimVar(tensorName = inputBufferName, dimIdx = idx)
            outputDimensionVar = tilerModel.getTensorDimVar(tensorName = outputBufferName, dimIdx = idx)

            step = abs(sliceSteps[idx])
            tilerModel.addConstraint(inputDimensionVar == ((outputDimensionVar - 1) * step + 1))

        return tilerModel

    @staticmethod
    def constructSymbolicNodeRep(tilerModel: TilerModel, parseDict: Dict,
                                 ctxt: NetworkContext) -> Dict[str, Union[int, IntVar]]:
        symbolicParseDict = parseDict.copy()

        return symbolicParseDict

    @staticmethod
    def computeInputCubeFromOutputCube(outputCube: HyperRectangle, parseDict: Dict) -> HyperRectangle:
        """Return the bounding input rectangle for normalized per-axis starts and steps."""
        starts, steps = parseDict['slice_starts'], parseDict['slice_steps']
        offsets, dims = [], []
        for offset, size, start, step in zip(outputCube.offset, outputCube.dims, starts, steps):
            first = start + offset * step
            last = first + (size - 1) * step
            offsets.append(min(first, last))
            dims.append(abs(last - first) + 1)
        return HyperRectangle(tuple(offsets), tuple(dims))

    @classmethod
    def serializeTilingSolution(
            cls, tilingSolution: NodeMemoryConstraint, absoluteOutputCubes: List[AbsoluteHyperRectangle],
            targetMemLevel: str, ctxt: NetworkContext,
            operatorRepresentation: OperatorRepresentation) -> Tuple[VariableReplacementScheme, TilingSchedule]:
        starts, steps = cls.normalizedParameters(operatorRepresentation, ctxt)
        # Keep the original representation intact: other passes still need its buffer names.
        immediateParameters = {'slice_starts': starts, 'slice_steps': steps}
        inputBaseOffsets, outputBaseOffsets = cls.extractBaseAddr(tilingSolution, targetMemLevel,
                                                                  operatorRepresentation, ['data_in', 'data_out'])
        replacements = {
            f'{name}_{axis}': [] for axis in range(len(starts))
            for name in ('slice_in_dim', 'slice_out_dim', 'slice_start')
        }
        replacementTypes = {key: PointerClass(uint32_t) for key in replacements}
        inputLoadSchedule, outputLoadSchedule = [], []
        for absoluteCube in absoluteOutputCubes:
            outCube = absoluteCube.rectangle
            inCube = cls.computeInputCubeFromOutputCube(outCube, immediateParameters)
            for axis, step in enumerate(steps):
                replacements[f'slice_in_dim_{axis}'].append(inCube.dims[axis])
                replacements[f'slice_out_dim_{axis}'].append(outCube.dims[axis])
                # The DMA already applies the global slice offset. Reverse slices
                # start at the last element of the local bounding rectangle.
                replacements[f'slice_start_{axis}'].append(0 if step > 0 else inCube.dims[axis] - 1)
            inputLoadSchedule.append({'data_in': inCube})
            outputLoadSchedule.append({'data_out': outCube})
        return (VariableReplacementScheme(replacements, replacementTypes),
                TilingSchedule(inputBaseOffsets, outputBaseOffsets, inputLoadSchedule, outputLoadSchedule))

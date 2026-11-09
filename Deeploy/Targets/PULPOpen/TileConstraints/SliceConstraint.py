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
                raise ValueError("PULP Slice does not support empty outputs")
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
        """Return the input bounding box of ``outputCube`` in **full-tensor** coordinates.

        ``outputCube.offset`` must be expressed in full output coordinates. For every axis the selected
        input elements are ``first, first + step, ..., last`` with ``first = start + offset * step`` and
        ``last = first + (size - 1) * step``; the box spans ``min(first, last)`` and ``|last - first| + 1``.
        """
        starts, steps = parseDict['slice_starts'], parseDict['slice_steps']
        offsets, dims = [], []
        for offset, size, start, step in zip(outputCube.offset, outputCube.dims, starts, steps):
            first = start + offset * step
            last = first + (size - 1) * step
            offsets.append(min(first, last))
            dims.append(abs(last - first) + 1)
        return HyperRectangle(tuple(offsets), tuple(dims))

    @staticmethod
    def computeLocalInputCubeFromOutputCube(outputCube: HyperRectangle, parentOutputDims: Tuple[int, ...],
                                            parseDict: Dict) -> HyperRectangle:
        """Return the input bounding box of ``outputCube`` relative to its parent **input tile**.

        ``outputCube.offset`` is relative to a parent output tile of extent ``parentOutputDims``, and the
        parent input buffer holds exactly the bounding box of that parent output tile (it was produced by
        :meth:`computeInputCubeFromOutputCube` or by this method one level up). The slice start is therefore
        already contained in the parent buffer and must not be applied again:

        * ``step > 0``: the parent box starts at the parent's first selected element, so the local offset is
          ``off * step``.
        * ``step < 0``: the parent box starts at the parent's *last* selected element, so the local offset is
          measured from the end of the parent output tile: ``(P - off - size) * |step|``.

        The extent is ``(size - 1) * |step| + 1`` in both cases.
        """
        steps = parseDict['slice_steps']
        offsets, dims = [], []
        for offset, size, parentSize, step in zip(outputCube.offset, outputCube.dims, parentOutputDims, steps):
            assert 0 <= offset and offset + size <= parentSize, "Output tile exceeds its parent tile"
            if step > 0:
                offsets.append(offset * step)
            else:
                offsets.append((parentSize - offset - size) * -step)
            dims.append((size - 1) * abs(step) + 1)
        return HyperRectangle(tuple(offsets), tuple(dims))

    @classmethod
    def serializeTilingSolution(
            cls, tilingSolution: NodeMemoryConstraint, absoluteOutputCubes: List[AbsoluteHyperRectangle],
            targetMemLevel: str, ctxt: NetworkContext,
            operatorRepresentation: OperatorRepresentation) -> Tuple[VariableReplacementScheme, TilingSchedule]:
        """Serialize the output tiles of one parent tile.

        ``wrapTilingSolution`` calls this once per parent output tile, with that parent's children in
        ``absoluteOutputCubes``. ``cube.rectangle.offset`` is relative to the parent output tile and
        ``cube.absoluteOffset`` is relative to the full output tensor. Two cases:

        * The parent ``data_in`` buffer is the full input tensor (first tiling level): the input cube is the
          global bounding box computed from the absolute output offset, which includes the slice start.
        * The parent ``data_in`` buffer is itself a tile (nested tiling, e.g. L3 -> L2 -> L1): it is the
          bounding box of the parent output tile, so the input cube is computed relative to it and the start
          is not added again. The parent output extent per axis is ``max(offset + dims)`` over the children,
          since they partition the parent (``computeTileHyperRectangles``), including partial edge tiles.

        In both cases ``slice_start_k`` is local to the transferred input tile: ``0`` for ``step > 0`` (the
        box starts at the first selected element) and ``dims - 1`` for ``step < 0`` (it ends there). If the
        input is not transferred at all (resident at the target level), the kernel indexes the full tensor, so
        the full input shape and the global first index are used instead.
        """
        starts, steps = cls.normalizedParameters(operatorRepresentation, ctxt)
        # Keep the original representation intact: other passes still need its buffer names.
        immediateParameters = {'slice_starts': starts, 'slice_steps': steps}
        inputBaseOffsets, outputBaseOffsets = cls.extractBaseAddr(tilingSolution, targetMemLevel,
                                                                  operatorRepresentation, ['data_in', 'data_out'])

        inputLevelIdx = cls.inputLevelIndex(tilingSolution, targetMemLevel, operatorRepresentation)
        inputResident = inputLevelIdx <= 0
        globalInput = inputLevelIdx <= 1
        fullInputShape = tuple(operatorRepresentation['data_in_shape'])
        parentOutputDims = tuple(
            max(cube.rectangle.offset[axis] + cube.rectangle.dims[axis]
                for cube in absoluteOutputCubes)
            for axis in range(len(starts)))

        replacements = {
            f'{name}_{axis}': [] for axis in range(len(starts))
            for name in ('slice_in_dim', 'slice_out_dim', 'slice_start')
        }
        replacementTypes = {key: PointerClass(uint32_t) for key in replacements}
        inputLoadSchedule, outputLoadSchedule = [], []
        for absoluteCube in absoluteOutputCubes:
            outCube = absoluteCube.rectangle
            if globalInput:
                globalOutCube = HyperRectangle(tuple(absoluteCube.absoluteOffset), tuple(outCube.dims))
                inCube = cls.computeInputCubeFromOutputCube(globalOutCube, immediateParameters)
            else:
                inCube = cls.computeLocalInputCubeFromOutputCube(outCube, parentOutputDims, immediateParameters)
            for axis, step in enumerate(steps):
                replacements[f'slice_out_dim_{axis}'].append(outCube.dims[axis])
                if inputResident:
                    # No transfer: the kernel reads the full tensor, so use global coordinates.
                    replacements[f'slice_in_dim_{axis}'].append(fullInputShape[axis])
                    replacements[f'slice_start_{axis}'].append(starts[axis] + absoluteCube.absoluteOffset[axis] * step)
                else:
                    # The DMA already applied the tile offset. Reverse slices start at the last element of
                    # the local bounding box.
                    replacements[f'slice_in_dim_{axis}'].append(inCube.dims[axis])
                    replacements[f'slice_start_{axis}'].append(0 if step > 0 else inCube.dims[axis] - 1)
            inputLoadSchedule.append({'data_in': inCube})
            outputLoadSchedule.append({'data_out': outCube})
        return (VariableReplacementScheme(replacements, replacementTypes),
                TilingSchedule(inputBaseOffsets, outputBaseOffsets, inputLoadSchedule, outputLoadSchedule))

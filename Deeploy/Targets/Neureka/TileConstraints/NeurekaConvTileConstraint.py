# SPDX-FileCopyrightText: 2024 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, List, Tuple, Type

from Deeploy.AbstractDataTypes import Pointer, PointerClass
from Deeploy.CommonExtensions.DataTypes import uint8_t, uint16_t, uint32_t
from Deeploy.DeeployTypes import NetworkContext, OperatorRepresentation, VariableBuffer
from Deeploy.Targets.Neureka.Templates.ConvTemplate import NeurekaConvTemplate, getInputAddrOffset, \
    ioStridesFromDimensions
from Deeploy.Targets.Neureka.TileConstraints.RequantHelpers import requantAddGeometricalConstraint, requantLoadSchedule
from Deeploy.Targets.PULPOpen.TileConstraints.ConvTileConstraint import Conv2DTileConstraint
from Deeploy.TilingExtension.MemoryConstraints import NodeMemoryConstraint
from Deeploy.TilingExtension.TileConstraint import TileConstraint
from Deeploy.TilingExtension.TilingCodegen import AbsoluteHyperRectangle, HyperRectangle, TilingSchedule, \
    VariableReplacementScheme

# Order in which the N-EUREKA hardware subtile counters are returned by every template's getCounters().
_COUNTER_NAMES = ("nKo", "nKi", "nHo", "nWo", "bKo", "bKi", "bHo", "bWo", "bHi", "bWi")


class PerTileReplacements:
    """Accumulate per-tile template-variable replacements together with their C type.

    Each ``append`` records both the value and the variable's pointer type.
    Call ``scheme`` once at the end to materialize the :class:`VariableReplacementScheme` the tiler expects.
    """

    def __init__(self) -> None:
        self._types: Dict[str, Type] = {}
        self._values: Dict[str, List] = {}

    def append(self, name: str, dtype: Type, value) -> None:
        if name not in self._types:
            self._types[name] = dtype
            self._values[name] = []
        self._values[name].append(value)

    def scheme(self) -> VariableReplacementScheme:
        replacementTypes: Dict[str, Type[Pointer]] = {name: PointerClass(dtype) for name, dtype in self._types.items()}
        return VariableReplacementScheme(self._values, replacementTypes)


class NeurekaConvTileConstraint(TileConstraint):
    """Shared tiling logic for the N-EUREKA convolution variants (pointwise, depthwise, dense).

    The serialization skeleton (input-cube computation, I/O strides, subtile counters, load
    schedules) is identical across the three; the parts that genuinely differ are exposed as hooks:

    - ``_ConvTemplate``   : the template class providing ``getCounters`` for this variant.
    - ``_adjustInputCube``: post-process the computed input cube (depthwise slices channels).
    - ``_addWeightSchedule``: emit the weight base offset / load schedule (packing differs per variant).
    """

    # Set by each concrete variant to its Neureka2D*ConvTemplate subclass.
    _ConvTemplate: Type[NeurekaConvTemplate]

    @classmethod
    def _adjustInputCube(cls, inCube: HyperRectangle, outputCube: HyperRectangle) -> HyperRectangle:
        """Adjust the input cube derived from an output tile. Identity by default."""
        return inCube

    @classmethod
    def _addWeightSchedule(cls, rep: PerTileReplacements, inputLoadSchedule: List[Dict[str, HyperRectangle]],
                           inputBaseOffsets: Dict[str, List[int]], outputBaseOffsets: Dict[str, List[int]],
                           absoluteOutputCubes: List[AbsoluteHyperRectangle], tilingSolution: NodeMemoryConstraint,
                           targetMemLevel: str, ctxt: NetworkContext,
                           operatorRepresentation: OperatorRepresentation) -> None:
        """Emit the per-tile weight addressing (offset and/or load schedule). Variant-specific."""
        raise NotImplementedError(f"{cls.__name__} must implement _addWeightSchedule")

    @classmethod
    def serializeTilingSolution(
            cls, tilingSolution: NodeMemoryConstraint, absoluteOutputCubes: List[AbsoluteHyperRectangle],
            targetMemLevel: str, ctxt: NetworkContext,
            operatorRepresentation: OperatorRepresentation) -> Tuple[VariableReplacementScheme, TilingSchedule]:
        outputCubes = [cube.rectangle for cube in absoluteOutputCubes]

        inputBaseOffsets, outputBaseOffsets = cls.extractBaseAddr(tilingSolution, targetMemLevel,
                                                                  operatorRepresentation, ['data_in', 'data_out'])

        outputBuffer = ctxt.lookup(operatorRepresentation['data_out'])
        assert isinstance(outputBuffer, VariableBuffer)

        weightH: int = operatorRepresentation['dim_kernel_y']
        weightW: int = operatorRepresentation['dim_kernel_x']
        weightC: int = operatorRepresentation['ch_im_in']
        pads: tuple[int, int, int, int] = operatorRepresentation['pads']
        strides: tuple[int, int] = operatorRepresentation['strides']

        input_bits: int = operatorRepresentation["input_bits"]
        output_bits: int = operatorRepresentation["output_bits"]

        rep = PerTileReplacements()
        inputCubes = []

        for cube in outputCubes:
            (_, _, _, COffset) = cube.offset
            (_, HSize, WSize, CSize) = cube.dims

            inCube, pads_tuple = Conv2DTileConstraint.computeInputCube((weightH, weightW), pads, strides, weightC, cube,
                                                                       outputBuffer.shape)
            inCube = cls._adjustInputCube(inCube, cube)

            pad_left, pad_right, pad_top, pad_bottom = pads_tuple
            rep.append('padding_y_top', uint8_t, pad_top)
            rep.append('padding_y_bottom', uint8_t, pad_bottom)
            rep.append('padding_x_left', uint8_t, pad_left)
            rep.append('padding_x_right', uint8_t, pad_right)

            _, _, inWSize, inCSize = inCube.dims
            dim_im_in_x_stride, dim_im_in_y_stride = ioStridesFromDimensions(inWSize, inCSize, input_bits)
            rep.append('dim_im_in_x_stride', uint32_t, dim_im_in_x_stride)
            rep.append('dim_im_in_y_stride', uint32_t, dim_im_in_y_stride)
            dim_im_out_x_stride, dim_im_out_y_stride = ioStridesFromDimensions(WSize, CSize, output_bits)
            rep.append('dim_im_out_x_stride', uint32_t, dim_im_out_x_stride)
            rep.append('dim_im_out_y_stride', uint32_t, dim_im_out_y_stride)

            rep.append('input_addr_offset', uint32_t, getInputAddrOffset(inWSize, dim_im_in_y_stride, pad_top,
                                                                         pad_left))

            counters = cls._ConvTemplate.getCounters(inCSize, HSize, WSize, CSize, pad_bottom, pad_right,
                                                     operatorRepresentation)
            for name, value in zip(_COUNTER_NAMES, counters):
                rep.append(name, uint16_t, value)

            inputCubes.append(inCube)

        inputLoadSchedule = [{"data_in": cube} for cube in inputCubes]
        outputLoadSchedule = [{"data_out": cube} for cube in outputCubes]

        cls._addWeightSchedule(rep, inputLoadSchedule, inputBaseOffsets, outputBaseOffsets, absoluteOutputCubes,
                               tilingSolution, targetMemLevel, ctxt, operatorRepresentation)

        tilingSchedule = TilingSchedule(inputBaseOffsets, outputBaseOffsets, inputLoadSchedule, outputLoadSchedule)

        return rep.scheme(), tilingSchedule


class NeurekaRQSConvTileConstraint(NeurekaConvTileConstraint):
    """Mixin adding requantization to any :class:`NeurekaConvTileConstraint` variant.

    Combine it (listed first) with a concrete variant, e.g.::

        class NeurekaRQSDWConv2DTileConstraint(NeurekaRQSConvTileConstraint, NeurekaDWConv2DTileConstraint):
            pass

    Cooperative ``super()`` dispatch then routes to the variant's geometrical constraint and
    serialization before layering the requant offsets/loads on top.
    """

    @classmethod
    def addGeometricalConstraint(cls, tilerModel, parseDict: Dict, ctxt: NetworkContext):
        tilerModel = super().addGeometricalConstraint(tilerModel, parseDict, ctxt)
        return requantAddGeometricalConstraint(tilerModel, parseDict, ctxt)

    @classmethod
    def serializeTilingSolution(
            cls, tilingSolution: NodeMemoryConstraint, absoluteOutputCubes: List[AbsoluteHyperRectangle],
            targetMemLevel: str, ctxt: NetworkContext,
            operatorRepresentation: OperatorRepresentation) -> Tuple[VariableReplacementScheme, TilingSchedule]:

        variableReplacementSchedule, tilingSchedule = super().serializeTilingSolution(
            tilingSolution, absoluteOutputCubes, targetMemLevel, ctxt, operatorRepresentation)

        inputRequantBaseOffsets, _ = cls.extractBaseAddr(tilingSolution, targetMemLevel, operatorRepresentation,
                                                         ['mul', 'add'])
        newInputBaseOffsets = {**tilingSchedule.inputBaseOffsets, **inputRequantBaseOffsets}

        requantSchedule = requantLoadSchedule(absoluteOutputCubes, ctxt, operatorRepresentation)
        newInputLoadSchedule = [{
            **load,
            **rqLoad
        } for load, rqLoad in zip(tilingSchedule.inputLoadSchedule, requantSchedule)]

        newTilingSchedule = TilingSchedule(newInputBaseOffsets, tilingSchedule.outputBaseOffsets, newInputLoadSchedule,
                                           tilingSchedule.outputLoadSchedule)

        return variableReplacementSchedule, newTilingSchedule

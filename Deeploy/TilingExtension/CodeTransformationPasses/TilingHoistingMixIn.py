# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

import math
from typing import List, Mapping, Optional, Sequence, Tuple, Type, TypeVar, Union

import Deeploy.CommonExtensions.DataTypes as BasicDataTypes
from Deeploy.AbstractDataTypes import BaseType, PointerClass, VoidType
from Deeploy.DeeployTypes import ConstantBuffer, NetworkContext, NodeTemplate, OperatorRepresentation, VariableBuffer, \
    _ReferenceBuffer
from Deeploy.TilingExtension.MemoryConstraints import TensorMemoryConstraint
from Deeploy.TilingExtension.TilingCodegen import TilingSchedule

KT = TypeVar('KT')
VT = TypeVar('VT')


def dictOfArrays(arrayOfDicts: Sequence[Mapping[KT, VT]]) -> Mapping[KT, List[VT]]:
    ret: Mapping[KT, List[VT]] = {}
    for i, _dict in enumerate(arrayOfDicts):
        if i == 0:
            ret.update({key: [value] for key, value in _dict.items()})
        else:
            assert set(ret.keys()) == set(_dict.keys()), "Keys should be the same"
            for key, value in _dict.items():
                ret[key].append(value)
    return ret


class TilingHoistingMixIn:

    _DEFAULT_HOIST_PREFIX = "TILING_CODEGEN_"

    def __init__(self, memory: str) -> None:
        self.memory = memory
        self._prefix = None

    def _initPrefix(self, nodeName: str) -> None:
        self._prefix = f"{self._DEFAULT_HOIST_PREFIX}{self.memory}_{nodeName}_"

    def _deinitPrefix(self) -> None:
        self._prefix = None

    @property
    def prefix(self) -> str:
        assert self._prefix is not None, "Prefix is not initialized!"
        return self._prefix

    def _hoistValues(self,
                     ctxt: NetworkContext,
                     name: str,
                     values: List[int],
                     override_type: Optional[Type[BaseType]] = None) -> ConstantBuffer:
        assert all(isinstance(value, int) for value in values)
        cb = ctxt.ConstantBuffer(self.prefix + name, [len(values)], values)
        ctxt.add(cb, 'global')
        if override_type is not None:
            cb._type = PointerClass(override_type)
        else:
            cb._type = PointerClass(BasicDataTypes.minimalIntegerType(values))
        cb._instance = cb._type(cb.name, ctxt)
        # These are constant tile *control* tables (numTiles / DMA cmd / size /
        # dims / offsets) read by the (cluster) controller to drive the tiling
        # loop and program DMAs -- not bulk tile data. Putting them in the
        # innermost tile memory (L1/TCDM) wastes scarce L1 and, on GAP9, places
        # them in the contended L1 region next to the cluster master stack: a
        # deep stack write can clobber a single table entry, turning a DMA `cmd`
        # into a garbage code pointer so mchan_transfer_wait() hangs forever
        # (observed on MobileNetV1 training, OptimizerNetwork _sgd_blocks_5...).
        # Keep them in the controller-addressable outer memory (L2) instead.
        # Only redirect the L2->L1 pass; the L3->L2 pass must keep its tables in
        # L2 (== self.memory), never L3. Platforms that don't tile into "L1"
        # (different level naming) are unaffected.
        cb._memoryLevel = "L2" if self.memory == "L1" else self.memory
        return cb

    def _hoistReference(self,
                        ctxt: NetworkContext,
                        name: str,
                        reference: VariableBuffer,
                        shape: Tuple[int, ...] = (1,),
                        offset: Union[int, str, VariableBuffer] = 0,
                        override_type: Optional[Type[BaseType]] = None) -> _ReferenceBuffer:
        ref = ctxt.hoistReference(self.prefix + name, reference, shape, offset, override_type)
        ref._memoryLevel = self.memory
        return ref

    def _hoistTileNumAndIdxPtr(
            self,
            ctxt: NetworkContext,
            tilingSchedules: List[TilingSchedule],
            nodeMemoryConstraint: Optional['NodeMemoryConstraint'] = None) -> Tuple[ConstantBuffer, VariableBuffer]:
        stepsNumTiles = [len(tilingSchedule.outputLoadSchedule) for tilingSchedule in tilingSchedules]

        # Core extension: at the innermost memory level (L1), emit a per-tile
        # boundary so each invocation of the inner closure processes exactly one
        # tile. The outer-level (L3->L2) closure iterates N_outer times and calls
        # inner once per iter; with the baseline cumulative layout
        # `{0, N1, N1+N2, ...}` inner would process many tiles per call and only
        # tolerate `len(tilingSchedules)` outer iters before reading numTiles
        # OOB. Per-tile layout `{0,1,2,...,total}` keeps
        # outer_iters == inner_calls == total_tiles. Outer memory levels keep
        # cumulative layout to iterate per L2 tile.
        # When L1 is the innermost AND there's an outer (L3) driver, each outer
        # iter invokes the L1 closure once → use per-tile {0,1,2,...,total} so
        # one inner-call processes one tile. When L1 is the OUTERMOST tiling
        # level (defaultMemLevel=L2, no L3 driver), the L1 closure is called
        # exactly once from RunNetwork and must walk all tiles itself → use
        # cumulative {0, total}.
        #
        # Detect "has outer driver": at L1 hoist time, L2 hoist has not yet run
        # (PULPClusterTiling[L2->L1] runs before PULPL3Tiling[L3->L2] in the
        # CodeTransformation pipeline), so we can't probe L2_numTiles buffers.
        # Use the operator's tensor placements instead: if any of the tensors
        # referenced by this tiling schedule has _memoryLevel == "L3", then a
        # downstream L3 hoist will wrap this L1 closure in an L3 loop that
        # drives `total_tiles` invocations — use per-tile {0, 1, ..., total}.
        # If no tensors are L3-resident (defaultMemLevel=L2), the L1 closure
        # is the outermost tile loop, called once from RunNetwork; emit
        # cumulative {0, total} so the single call walks every tile.
        l1_has_outer_driver = False
        if self.memory == "L1" and nodeMemoryConstraint is not None:
            # Only probe input/output tensors, not intermediate (spilled) ones.
            # Intermediate buffers (e.g. transposed activations kept alive for the
            # backward pass) can spill to L3 under memory pressure without creating
            # an L3 outer loop.  Including them would falsely trigger per-tile
            # layout for nodes whose main I/O stays in L2 (e.g. DSCNN at L1=64000).
            io_tmcs = (list(nodeMemoryConstraint.inputTensorMemoryConstraints.values()) +
                       list(nodeMemoryConstraint.outputTensorMemoryConstraints.values()))
            for tmc in io_tmcs:
                if "L3" in tmc.memoryConstraints:
                    l1_has_outer_driver = True
                    break
        if self.memory == "L1" and l1_has_outer_driver:
            total = sum(stepsNumTiles)
            cumulativeNumTiles = list(range(total + 1))
        else:
            cumulativeNumTiles = [0]
            for numTiles in stepsNumTiles:
                cumulativeNumTiles.append(cumulativeNumTiles[-1] + numTiles)

        tileNum = self._hoistValues(ctxt, "numTiles", cumulativeNumTiles)

        tileIdxPtrName = f"{self.prefix}tileIdxPtr"
        # Idempotent: reuse if pre-hoisted by a template's alignToContext
        # (see _ConvGradWTemplate). That lets template rendering -- which
        # Closure passes trigger BEFORE this tiling pass runs -- already see
        # the real tileIdxPtr buffer name instead of a 'NULL' sentinel.
        # Keep whatever type the pre-hoister chose so the earlier captured
        # closure-struct field types stay consistent with the later outer-
        # scope initTemplate declaration.
        if ctxt.is_buffer(tileIdxPtrName):
            tileIdxPtr = ctxt.lookup(tileIdxPtrName)
            return (tileNum, tileIdxPtr)

        tileIdxPtr = ctxt.VariableBuffer(tileIdxPtrName, shape = [1])
        ctxt.add(tileIdxPtr, "local")

        tileIdxPtr._type = tileNum._type
        tileIdxPtr._instance = tileIdxPtr._type(tileIdxPtr.name, ctxt)
        # LMACAN: Intentionally don't annotate memory level so it gets allocated
        # outside of the tiling loops

        tileIdxPtr.allocTemplate = NodeTemplate("""
        ${type.referencedType.typeName} bu_${name} = 0;
        ${type.referencedType.typeName}* ${name} = &bu_${name};""")
        tileIdxPtr.deallocTemplate = NodeTemplate("")
        tileIdxPtr.initTemplate = NodeTemplate("")

        return (tileNum, tileIdxPtr)

    def _hoistOpReprUpdates(self,
                            ctxt: NetworkContext,
                            opReprs: List[OperatorRepresentation],
                            prefix: str = "") -> Tuple[OperatorRepresentation, List[str]]:
        # Early exit if the opReprs list is empty because the following code assumes at least 1 opRepr is in the list
        if len(opReprs) == 0:
            return {}, []

        newOpRepr = {}
        hoistedReprNames = []
        for var, updates in dictOfArrays(opReprs).items():
            if all(update == updates[0] for update in updates):
                newOpRepr[var] = updates[0]
            else:
                cb = self._hoistValues(ctxt, f"{prefix}{var}", updates)
                newOpRepr[var] = cb.name
                hoistedReprNames.append(var)
        return newOpRepr, hoistedReprNames

    def _hoistMultibufferReferences(self, ctxt: NetworkContext, buffer: VariableBuffer,
                                    tensorMemoryConstraint: TensorMemoryConstraint) -> List[_ReferenceBuffer]:
        tensorName = tensorMemoryConstraint.tensorName
        memoryConstraint = tensorMemoryConstraint.memoryConstraints[self.memory]
        assert memoryConstraint.addrSpace is not None, "Assuming address space is set"
        totalSize = memoryConstraint.addrSpace[1] - memoryConstraint.addrSpace[0]
        assert isinstance(memoryConstraint.multiBufferCoefficient,
                          int), "Assuming multi buffer coefficient has been assigned"
        assert totalSize % memoryConstraint.multiBufferCoefficient == 0, "Assuming total size is divisible by the multi buffer coefficient"
        bufferSize = totalSize // memoryConstraint.multiBufferCoefficient

        assert memoryConstraint.multiBufferCoefficient == 2, "Multi buffer coefficient has to be equal to 2 since this is for double buffering"
        assert memoryConstraint.shape is not None
        assert len(memoryConstraint.shape) > 0
        assert isinstance(memoryConstraint.shape[0], int)
        tileLength = math.prod(memoryConstraint.shape)
        tileSize = int(math.ceil(tileLength * buffer._type.referencedType.typeWidth / 8))

        assert bufferSize >= tileSize, f"Provided buffer size is not enough to fit the tile. Buffer size: {bufferSize}, tile size: {tileSize}"

        refs = [
            self._hoistReference(
                ctxt,
                f"{tensorName}_buffer_{i}",
                buffer,
                memoryConstraint.shape,
                offset = i * bufferSize,
                override_type = VoidType,
            ) for i in range(memoryConstraint.multiBufferCoefficient)
        ]

        return refs
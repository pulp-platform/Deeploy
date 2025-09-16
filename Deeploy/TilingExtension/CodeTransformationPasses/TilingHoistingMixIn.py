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
        cb._memoryLevel = self.memory
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

    def _hoistTileNumAndIdxPtr(self, ctxt: NetworkContext,
                               tilingSchedules: List[TilingSchedule]) -> Tuple[ConstantBuffer, VariableBuffer]:
        stepsNumTiles = [len(tilingSchedule.outputLoadSchedule) for tilingSchedule in tilingSchedules]

        cumulativeNumTiles = [0]
        for numTiles in stepsNumTiles:
            cumulativeNumTiles.append(cumulativeNumTiles[-1] + numTiles)

        tileNum = self._hoistValues(ctxt, "numTiles", cumulativeNumTiles)

        tileIdxPtr = ctxt.VariableBuffer(f"{self.prefix}tileIdxPtr", shape = [1])
        ctxt.add(tileIdxPtr, "local")

        tileIdxPtr._type = tileNum._type
        tileIdxPtr._instance = tileIdxPtr._type(tileIdxPtr.name, ctxt)
        # LMACAN: Intentionally don't annotate memory level so it gets allocated
        # outside of the tiling loops

        tileIdxPtr.allocTemplate = NodeTemplate("")
        tileIdxPtr.deallocTemplate = NodeTemplate("")
        tileIdxPtr.initTemplate = NodeTemplate("""
        ${type.referencedType.typeName} bu_${name} = 0;
        ${type.referencedType.typeName}* ${name} = &bu_${name};""")

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

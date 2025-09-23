# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

import math
from abc import ABC, abstractmethod
from typing import Dict, List, Literal, Optional, Set, Tuple, Type

from Deeploy.DeeployTypes import CodeSnippet, NetworkContext, NodeTemplate, OperatorRepresentation, VariableBuffer, \
    _ReferenceBuffer
from Deeploy.TilingExtension.TilingCodegen import padShape, padStride

DmaDirection = Literal["ExternalToLocal", "LocalToExternal"]


class Future:

    _initTemplate: NodeTemplate
    _allocTemplate: NodeTemplate
    _deinitTemplate: NodeTemplate
    _waitTemplate: NodeTemplate

    _registry: Dict[str, "Future"] = {}

    @classmethod
    def _buildName(cls, name: str, copyIdx: Optional[int] = None) -> str:
        name += "_future"
        if copyIdx is not None:
            name += f"_{copyIdx}"
        return name

    def __new__(cls, name: str, copyIdx: Optional[int] = None) -> "Future":
        futureName = cls._buildName(name, copyIdx)
        if futureName in cls._registry:
            return cls._registry[futureName]
        else:
            inst = super().__new__(cls)
            cls._registry[futureName] = inst
            return inst

    def __init__(self, name: str, copyIdx: Optional[int] = None):
        # LMACAN: __init__ is always called after __new__.
        #         This guards against reinitialization in case the future already exists in the registry.
        if not hasattr(self, "name"):
            self.name = name
            self._allocated = False
            self._waited = False

    def _operatorRepresentation(self, comment: str = "") -> OperatorRepresentation:
        return {"name": self.name, "comment": comment}

    def init(self, comment: str = "") -> CodeSnippet:
        return CodeSnippet(self._initTemplate, self._operatorRepresentation(comment))

    def alloc(self, comment: str = "") -> CodeSnippet:
        if self._allocated:
            return CodeSnippet(NodeTemplate(""), self._operatorRepresentation(comment))
        self._allocated = True
        return CodeSnippet(self._allocTemplate, self._operatorRepresentation(comment))

    def deinit(self, comment: str = "") -> CodeSnippet:
        return CodeSnippet(self._deinitTemplate, self._operatorRepresentation(comment))

    def wait(self, comment: str = "") -> CodeSnippet:
        if self._waited:
            return CodeSnippet(NodeTemplate(""), self._operatorRepresentation(comment))
        self._waited = True
        return CodeSnippet(self._waitTemplate, self._operatorRepresentation(comment))

    def __hash__(self) -> int:
        return hash(self.name)


class AsyncDmaWaitingStrategy(ABC):

    def __init__(self, FutureCls: Type[Future]) -> None:
        self.FutureCls = FutureCls

    @abstractmethod
    def getFuture(self, tensorName: str, direction: DmaDirection, copyIdx: Optional[int] = None) -> Future:
        pass

    @abstractmethod
    def resetState(self):
        pass


class PerTensorWaitingStrategy(AsyncDmaWaitingStrategy):

    def getFuture(self, tensorName: str, direction: DmaDirection, copyIdx: Optional[int] = None) -> Future:
        _ = direction
        return self.FutureCls(tensorName, copyIdx)

    def resetState(self):
        return


class DirectionWaitingStrategy(AsyncDmaWaitingStrategy):

    def __init__(self, FutureCls: Type[Future], asyncGroupName: str) -> None:
        super().__init__(FutureCls)
        self.asyncGroupName = asyncGroupName

    def getFuture(self, tensorName: str, direction: DmaDirection, copyIdx: Optional[int] = None) -> Future:
        _ = tensorName
        name = self.asyncGroupName
        if direction == "ExternalToLocal":
            name += "_input"
        else:
            name += "_output"
        return self.FutureCls(name, copyIdx)

    def resetState(self):
        for future in self.FutureCls._registry.values():
            if future.name.startswith(self.asyncGroupName):
                future._allocated = False
                future._waited = False


class BarrierWaitingStrategy(AsyncDmaWaitingStrategy):

    def __init__(self, FutureCls: Type[Future], barrierName: str) -> None:
        super().__init__(FutureCls)
        self.barrier = FutureCls(barrierName)

    def getFuture(self, tensorName: str, direction: DmaDirection, copyIdx: Optional[int] = None) -> Future:
        _ = tensorName, direction, copyIdx
        return self.barrier


class AsyncDma(ABC):

    _waitingStrategy: AsyncDmaWaitingStrategy

    def __init__(self, transferTemplates: Dict[int, NodeTemplate]) -> None:
        self._transferTemplates = transferTemplates

    def getFuture(self, tensorName: str, direction: DmaDirection, copyIdx: Optional[int] = None) -> Future:
        return self._waitingStrategy.getFuture(tensorName, direction, copyIdx)

    def supportedTransferRanks(self) -> Set[int]:
        return set(self._transferTemplates.keys())

    def checkTransfer(self, ctxt: NetworkContext, externalBuffer: VariableBuffer, localBuffer: VariableBuffer,
                      shape: Tuple[int, ...], strideExt: Tuple[int, ...], strideLoc: Tuple[int, ...],
                      direction: DmaDirection) -> None:
        transferRank = len(shape)
        assert transferRank == len(strideLoc) and transferRank == len(
            strideExt), f"The shape and stride rank should match"
        assert transferRank in self.supportedTransferRanks(
        ), f"Unsupported transfer rank {transferRank}. Supported ranks are {self.supportedTransferRanks()}"

    @abstractmethod
    def transferOpRepr(self,
                       externalBuffer: VariableBuffer,
                       localBuffer: VariableBuffer,
                       shape: Tuple[int, ...],
                       strideExt: Tuple[int, ...],
                       strideLoc: Tuple[int, ...],
                       direction: DmaDirection,
                       future: Future,
                       comment: str = "") -> OperatorRepresentation:
        return {"loc": localBuffer.name, "ext": externalBuffer.name, "future": future.name, "comment": comment}

    def transfer(self,
                 ctxt: NetworkContext,
                 externalBuffer: VariableBuffer,
                 localBuffer: VariableBuffer,
                 shape: Tuple[int, ...],
                 strideExt: Tuple[int, ...],
                 strideLoc: Tuple[int, ...],
                 direction: DmaDirection,
                 future: Future,
                 comment: str = "") -> Tuple[List[CodeSnippet], List[CodeSnippet], List[CodeSnippet]]:
        self.checkTransfer(ctxt, externalBuffer, localBuffer, shape, strideExt, strideLoc, direction)
        opRepr = self.transferOpRepr(externalBuffer, localBuffer, shape, strideExt, strideLoc, direction, future,
                                     comment)
        template = self._transferTemplates[len(shape)]
        return [future.alloc(comment)], [CodeSnippet(template, opRepr)], []

    def resetState(self):
        self._waitingStrategy.resetState()


class EmptyFuture(Future):

    _initTemplate = NodeTemplate("")
    _allocTemplate = NodeTemplate("")
    _deinitTemplate = NodeTemplate("")
    _waitTemplate = NodeTemplate("")


class BlockingDmaFromAsyncDmaAdapter(AsyncDma):

    _waitingStrategy = PerTensorWaitingStrategy(EmptyFuture)

    def __init__(self, dma: AsyncDma) -> None:
        self.dma = dma

    @property
    def _transferTemplates(self) -> Dict[int, NodeTemplate]:
        return self.dma._transferTemplates

    def getFuture(self, tensorName: str, direction: DmaDirection, copyIdx: Optional[int] = None) -> Future:
        return self.dma.getFuture(tensorName, direction, copyIdx)

    def transferOpRepr(self,
                       externalBuffer: VariableBuffer,
                       localBuffer: VariableBuffer,
                       shape: Tuple[int, ...],
                       strideExt: Tuple[int, ...],
                       strideLoc: Tuple[int, ...],
                       direction: DmaDirection,
                       future: Future,
                       comment: str = "") -> OperatorRepresentation:
        return self.dma.transferOpRepr(externalBuffer, localBuffer, shape, strideExt, strideLoc, direction, future,
                                       comment)

    def transfer(self,
                 ctxt: NetworkContext,
                 externalBuffer: VariableBuffer,
                 localBuffer: VariableBuffer,
                 shape: Tuple[int, ...],
                 strideExt: Tuple[int, ...],
                 strideLoc: Tuple[int, ...],
                 direction: DmaDirection,
                 future: Future,
                 comment: str = "") -> Tuple[List[CodeSnippet], List[CodeSnippet], List[CodeSnippet]]:
        callStack = []
        callStack.extend(
            self.dma.transfer(ctxt, externalBuffer, localBuffer, shape, strideExt, strideLoc, direction, future))
        callStack.append(future.wait())

        return [], callStack, []


class AnydimAsyncDmaTransferAdapter:

    class NestedForLoopOpenTemplate(NodeTemplate):

        def __init__(self, depth: int):
            templateStr = ""
            for level in range(depth):
                iter = f"i_{level}"
                templateStr += f"for (uint32_t {iter} = 0; {iter} < ${{end_{level}}}; {iter}++) {{"
            super().__init__(templateStr)

    class NestedForLoopCloseTemplate(NodeTemplate):

        def __init__(self, depth: int):
            templateStr = ""
            for _ in range(depth):
                templateStr += "}"
            super().__init__(templateStr)

    class OffsetCalculationTemplate(NodeTemplate):

        def __init__(self, name: str, depth: int):
            templateStr = f"const uint32_t {name} = "
            for i in range(depth):
                templateStr += f"i_{i} * ${{stride_{i}}}"
                if i < depth - 1:
                    templateStr += " + "
            templateStr += ";"
            super().__init__(templateStr)

    offsetPtrTemplate = NodeTemplate("void * const ${resultPtr} = (void *)${basePtr} + ${offset};")

    def __init__(self, dma: AsyncDma) -> None:
        self.dma = dma

    def getFuture(self, tensorName: str, direction: DmaDirection, copyIdx: Optional[int] = None) -> Future:
        return self.dma.getFuture(tensorName, direction, copyIdx)

    def nearestSupportedTransferRank(self, transfer_rank: int) -> int:
        sortedRanks = sorted(self.dma.supportedTransferRanks())

        # Find nearest smaller
        for rank in reversed(sortedRanks):
            if rank <= transfer_rank:
                return rank

        # All supported ranks are bigger so return the smallest one
        return sortedRanks[0]

    def transfer(self,
                 ctxt: NetworkContext,
                 externalBuffer: VariableBuffer,
                 localBuffer: VariableBuffer,
                 shape: Tuple[int, ...],
                 strideExt: Tuple[int, ...],
                 strideLoc: Tuple[int, ...],
                 direction: DmaDirection,
                 future: Future,
                 strideExtPad: int = 0,
                 comment: str = "") -> Tuple[List[CodeSnippet], List[CodeSnippet], List[CodeSnippet]]:
        transferRank = len(shape)
        kernelRank = self.nearestSupportedTransferRank(transferRank)

        if kernelRank < transferRank:
            nestedLoopDepth = transferRank - kernelRank

            nestedLoopOpRepr = {f"end_{level}": shape[level] for level in range(nestedLoopDepth)}
            locOffsetCalculationOpRepr = {f"stride_{level}": strideLoc[level] for level in range(nestedLoopDepth)}
            extOffsetCalculationOpRepr = {f"stride_{level}": strideExt[level] for level in range(nestedLoopDepth)}

            callStack = []
            callStack.append(CodeSnippet(self.NestedForLoopOpenTemplate(nestedLoopDepth), nestedLoopOpRepr))
            callStack.append(
                CodeSnippet(self.OffsetCalculationTemplate("ext_offset", nestedLoopDepth), extOffsetCalculationOpRepr))
            callStack.append(
                CodeSnippet(self.OffsetCalculationTemplate("loc_offset", nestedLoopDepth), locOffsetCalculationOpRepr))

            localBufferOffseted = _ReferenceBuffer("local_buffer_offsetted", localBuffer)
            localBufferOffseted._memoryLevel = localBuffer._memoryLevel
            callStack.append(
                CodeSnippet(self.offsetPtrTemplate, {
                    "resultPtr": "local_buffer_offsetted",
                    "basePtr": localBuffer.name,
                    "offset": "loc_offset"
                }))

            externalBufferOffseted = _ReferenceBuffer("external_buffer_offsetted", externalBuffer)
            externalBufferOffseted._memoryLevel = externalBuffer._memoryLevel
            callStack.append(
                CodeSnippet(self.offsetPtrTemplate, {
                    "resultPtr": externalBufferOffseted.name,
                    "basePtr": externalBuffer.name,
                    "offset": "ext_offset"
                }))

            alloc_code, dma_code, deinit_code = self.dma.transfer(ctxt,
                                                                  externalBufferOffseted,
                                                                  localBufferOffseted,
                                                                  shape[-kernelRank:],
                                                                  strideExt[-kernelRank:],
                                                                  strideLoc[-kernelRank:],
                                                                  direction,
                                                                  future,
                                                                  comment = comment)

            callStack.extend(dma_code)
            callStack.append(CodeSnippet(self.NestedForLoopCloseTemplate(nestedLoopDepth), {}))
            return alloc_code, callStack, deinit_code
        elif kernelRank == transferRank:
            return self.dma.transfer(ctxt,
                                     externalBuffer,
                                     localBuffer,
                                     shape,
                                     strideExt,
                                     strideLoc,
                                     direction,
                                     future,
                                     comment = comment)
        else:
            return self.dma.transfer(ctxt,
                                     externalBuffer,
                                     localBuffer,
                                     padShape(shape, kernelRank),
                                     padStride(strideExt, kernelRank, strideExtPad),
                                     padStride(strideLoc, kernelRank, math.prod(shape)),
                                     direction,
                                     future,
                                     comment = comment)

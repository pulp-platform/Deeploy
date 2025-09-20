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
    _deinitTemplate: NodeTemplate
    _allocTemplate: NodeTemplate
    _waitTemplate: NodeTemplate

    def __init__(self, name: str):
        self.name = name

    def _operatorRepresentation(self) -> OperatorRepresentation:
        return {"name": self.name}

    def init(self) -> CodeSnippet:
        return CodeSnippet(self._initTemplate, self._operatorRepresentation())

    def deinit(self) -> CodeSnippet:
        return CodeSnippet(self._deinitTemplate, self._operatorRepresentation())

    def alloc(self) -> CodeSnippet:
        return CodeSnippet(self._allocTemplate, self._operatorRepresentation())

    def wait(self) -> CodeSnippet:
        return CodeSnippet(self._waitTemplate, self._operatorRepresentation())


class AsyncDmaWaitingStrategy(ABC):

    def __init__(self, FutureCls: Type[Future]) -> None:
        self.FutureCls = FutureCls

    @abstractmethod
    def getFuture(self, tensorName: str) -> Future:
        pass


class PerTensorWaitingStrategy(AsyncDmaWaitingStrategy):

    def getFuture(self, tensorName: str) -> Future:
        return self.FutureCls(tensorName + "_future")


class TensorGroupWaitingStrategy(AsyncDmaWaitingStrategy):

    def __init__(self, FutureCls: Type[Future], asyncGroupName: str) -> None:
        super().__init__(FutureCls)
        self.asyncGroupFuture = FutureCls(f"{asyncGroupName}_future")

    def getFuture(self, tensorName: str) -> Future:
        _ = tensorName
        return self.asyncGroupFuture


class AsyncDma(ABC):

    _waitingStrategy: AsyncDmaWaitingStrategy

    def __init__(self, transferTemplates: Dict[int, NodeTemplate]) -> None:
        self._transferTemplates = transferTemplates

    def getFuture(self, tensorName: str, copyIdx: Optional[int] = None) -> Future:
        name = tensorName
        if copyIdx is not None:
            name += f"_{copyIdx}"
        return self._waitingStrategy.getFuture(name)

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
    def transferOpRepr(self, externalBuffer: VariableBuffer, localBuffer: VariableBuffer, shape: Tuple[int, ...],
                       strideExt: Tuple[int, ...], strideLoc: Tuple[int, ...], direction: DmaDirection,
                       future: Future) -> OperatorRepresentation:
        return {"loc": localBuffer.name, "ext": externalBuffer.name, "future": future.name}

    def transfer(self, ctxt: NetworkContext, externalBuffer: VariableBuffer, localBuffer: VariableBuffer,
                 shape: Tuple[int, ...], strideExt: Tuple[int, ...], strideLoc: Tuple[int, ...],
                 direction: DmaDirection, future: Future) -> List[CodeSnippet]:
        self.checkTransfer(ctxt, externalBuffer, localBuffer, shape, strideExt, strideLoc, direction)
        opRepr = self.transferOpRepr(externalBuffer, localBuffer, shape, strideExt, strideLoc, direction, future)
        template = self._transferTemplates[len(shape)]
        return [CodeSnippet(template, opRepr)]

    def setup(self) -> List[CodeSnippet]:
        return []

    def teardown(self) -> List[CodeSnippet]:
        return []


class EmptyFuture(Future):

    _initTemplate = NodeTemplate("")
    _deinitTemplate = NodeTemplate("")
    _waitTemplate = NodeTemplate("")


class BlockingDmaFromAsyncDmaAdapter(AsyncDma):

    _waitingStrategy = PerTensorWaitingStrategy(EmptyFuture)

    def __init__(self, dma: AsyncDma) -> None:
        self.dma = dma

    @property
    def _transferTemplates(self) -> Dict[int, NodeTemplate]:
        return self.dma._transferTemplates

    def transferOpRepr(self, externalBuffer: VariableBuffer, localBuffer: VariableBuffer, shape: Tuple[int, ...],
                       strideExt: Tuple[int, ...], strideLoc: Tuple[int, ...], direction: DmaDirection,
                       future: Future) -> OperatorRepresentation:
        return self.dma.transferOpRepr(externalBuffer, localBuffer, shape, strideExt, strideLoc, direction, future)

    def transfer(self, ctxt: NetworkContext, externalBuffer: VariableBuffer, localBuffer: VariableBuffer,
                 shape: Tuple[int, ...], strideExt: Tuple[int, ...], strideLoc: Tuple[int, ...],
                 direction: DmaDirection, future: Future) -> List[CodeSnippet]:
        tmpFuture = self.dma.getFuture(future.name.removesuffix("_future"))
        callStack = []
        callStack.append(tmpFuture.init())
        callStack.extend(
            self.dma.transfer(ctxt, externalBuffer, localBuffer, shape, strideExt, strideLoc, direction, tmpFuture))
        callStack.append(tmpFuture.wait())
        callStack.append(tmpFuture.deinit())
        return callStack

    def setup(self) -> List[CodeSnippet]:
        return self.dma.setup()

    def teardown(self) -> List[CodeSnippet]:
        return self.dma.teardown()


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
                 strideExtPad: int = 0) -> List[CodeSnippet]:
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

            callStack.extend(
                self.dma.transfer(ctxt, externalBufferOffseted, localBufferOffseted, shape[-kernelRank:],
                                  strideExt[-kernelRank:], strideLoc[-kernelRank:], direction, future))
            callStack.append(CodeSnippet(self.NestedForLoopCloseTemplate(nestedLoopDepth), {}))
            return callStack
        elif kernelRank == transferRank:
            return self.dma.transfer(ctxt, externalBuffer, localBuffer, shape, strideExt, strideLoc, direction, future)
        else:
            return self.dma.transfer(ctxt, externalBuffer, localBuffer, padShape(shape, kernelRank),
                                     padStride(strideExt, kernelRank, strideExtPad),
                                     padStride(strideLoc, kernelRank, math.prod(shape)), direction, future)

# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

import math
from abc import ABC, abstractmethod
from typing import Dict, List, Literal, Set, Tuple, Type

from Deeploy.DeeployTypes import CodeSnippet, NetworkContext, NodeTemplate, OperatorRepresentation, VariableBuffer, \
    _ReferenceBuffer
from Deeploy.TilingExtension.TilingCodegen import padShape, padStride

DmaDirection = Literal["ExternalToLocal", "LocalToExternal"]


class Future:

    _initTemplate: NodeTemplate
    _allocTemplate: NodeTemplate
    _deinitTemplate: NodeTemplate
    _waitTemplate: NodeTemplate

    def __init__(self, name: str):
        self._allocated = False
        self.name = name

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
        return CodeSnippet(self._waitTemplate, self._operatorRepresentation(comment))


class AsyncDmaWaitingStrategy(ABC):

    def __init__(self, FutureCls: Type[Future]) -> None:
        self.FutureCls = FutureCls

    @abstractmethod
    def getFuture(self, tensorName: str, direction: DmaDirection, initial = False) -> Future:
        pass

    @abstractmethod
    def resetState(self):
        pass


class PerTensorWaitingStrategy(AsyncDmaWaitingStrategy):

    def getFuture(self, tensorName: str, direction: DmaDirection, initial = False) -> Future:
        _ = direction
        return self.FutureCls(tensorName + "_future")

    def resetState(self):
        return


class TensorGroupWaitingStrategy(AsyncDmaWaitingStrategy):

    def __init__(self, FutureCls: Type[Future], asyncGroupName: str) -> None:
        super().__init__(FutureCls)
        self.asyncGroupName = asyncGroupName

        self.asyncGroupFutures = {
            direction: [
                FutureCls(f"{self.asyncGroupName}_{direction}_future"),
                FutureCls(f"{self.asyncGroupName}_{direction}_future")
            ] for direction in ["ExternalToLocal", "LocalToExternal"]
        }

    def getFuture(self, tensorName: str, direction: DmaDirection, initial = False) -> Future:
        _ = tensorName
        return self.asyncGroupFutures[direction][initial]

    def resetState(self):
        for futures in self.asyncGroupFutures.values():
            for future in futures:
                future._allocated = False


class AsyncDma(ABC):

    _waitingStrategy: AsyncDmaWaitingStrategy

    def __init__(self, transferTemplates: Dict[int, NodeTemplate]) -> None:
        self._transferTemplates = transferTemplates

    def getFuture(self, tensorName: str, direction: DmaDirection, initial = False) -> Future:
        return self._waitingStrategy.getFuture(tensorName, direction, initial)

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
        tmpFuture = self.dma.getFuture(future.name.removesuffix("_future"), direction)
        callStack = []
        callStack.append(tmpFuture.init(comment))
        callStack.append(tmpFuture.alloc(comment))
        _, dma_code, _ = self.dma.transfer(ctxt,
                                           externalBuffer,
                                           localBuffer,
                                           shape,
                                           strideExt,
                                           strideLoc,
                                           direction,
                                           tmpFuture,
                                           comment = comment)
        callStack.extend(dma_code)
        callStack.append(tmpFuture.wait(comment))
        callStack.append(tmpFuture.deinit(comment))

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

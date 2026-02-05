# SPDX-FileCopyrightText: 2023 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

import math
from typing import List, Set, Tuple

from Deeploy.AbstractDataTypes import VoidType
from Deeploy.DeeployTypes import CodeSnippet, ExecutionBlock, NetworkContext, NodeTemplate, OperatorRepresentation, \
    VariableBuffer, _ReferenceBuffer
from Deeploy.TilingExtension.AsyncDma import AnydimAsyncDmaTransferAdapter, AsyncDma, Future
from Deeploy.TilingExtension.CodeTransformationPasses.TilingCodeGeneration import TilingCodeGeneration
from Deeploy.TilingExtension.CodeTransformationPasses.TilingHoistingMixIn import dictOfArrays
from Deeploy.TilingExtension.CodeTransformationPasses.TilingPrototypes import ProfilingPrototypeMixIn, \
    PrototypeTilingMixIn, TilingMetaInfo
from Deeploy.TilingExtension.MemoryConstraints import NodeMemoryConstraint
from Deeploy.TilingExtension.TilingCodegen import TilingSchedule, VariableReplacementScheme, stridesFromShape


class DoubleBufferingTilingCodeGeneration(TilingCodeGeneration):

    _moveTileInCheckOpenStatement = NodeTemplate("""
    // DOUBLE BUFFERING CHECK TILE LOAD
    if ((${tileIdxVar}) < ${numTiles}[*${tileIdxPtr}+1]) {
    """)

    _moveTileInCheckCloseStatement = NodeTemplate("""
    }
    """)

    # LMACAN: The brackets around ${tileIdxVar} are important to ensure correct order
    #         of the modulo operation. Breaking case without the brackets is when we
    #         put "TILING_I + 1" for tileIdxVar.
    _switchOpen = NodeTemplate("switch((${tileIdxVar}) % ${bufferCount}) {")
    _caseOpen = NodeTemplate("case ${case}:")
    _caseClose = NodeTemplate("break;")

    _blockClose = NodeTemplate("""
    }
    """)

    _referenceUpdate = NodeTemplate("${reference} = (${type})${update};")

    def __init__(self, externalMemory: str, localMemory: str, dma: AsyncDma):
        super().__init__(externalMemory, localMemory, dma, 2)

    def _switch(self, caseBlocks: List[List[CodeSnippet]], tileIdxVar: str) -> List[CodeSnippet]:
        assert len(caseBlocks) == self.bufferCount, f"Expected {self.bufferCount} cases, got {len(caseBlocks)}`"
        callStack = [CodeSnippet(self._switchOpen, {"tileIdxVar": tileIdxVar, "bufferCount": self.bufferCount})]
        for i, block in enumerate(caseBlocks):
            callStack.append(CodeSnippet(self._caseOpen, {"case": i}))
            callStack.extend(block)
            callStack.append(CodeSnippet(self._caseClose, {}))
        callStack.append(CodeSnippet(self._blockClose, {}))
        return callStack

    def _generateBufferChoice(self, reference: VariableBuffer,
                              buffers: List[_ReferenceBuffer]) -> List[List[CodeSnippet]]:
        return [[
            CodeSnippet(self._referenceUpdate, {
                "reference": reference.name,
                "type": reference._type.typeName,
                "update": buff.name
            })
        ] for buff in buffers]

    def _tilingLoop(self, ctxt: NetworkContext, executionBlock: ExecutionBlock,
                    nodeMemoryConstraint: NodeMemoryConstraint, tilingSchedule: TilingSchedule,
                    variableReplacement: VariableReplacementScheme,
                    operatorRepresentation: OperatorRepresentation) -> Tuple[NetworkContext, ExecutionBlock, bool]:

        # Double Buffering Tiling Loop Strategy
        # ===================================
        # - 1) Initialize all futures
        # - 2) Start transfer for first input tile
        # - 3) Update input reference for second tile
        # - 4) for TILING_I in numTiles:
        #   - 4.1) Choose buffers for current tile (inputs and outputs)
        #   - 4.2) Input data transfer for next tile (see "4.2) Input Data Transfers")
        #   - 4.3) Process current tile
        #   - 4.4) Output data transfer for current tile (see "4.4) Output Data Transfers")
        # - 5) Wait for final output tile to be ready
        # - 6) Deinitialize all futures

        # 4.2) Input Data Transfers
        # -----------------------------------
        # - for each input tensor:
        #   - 4.2.1) Wait for current input tile
        #   - 4.2.2) if there is a next tile:
        #     - 4.2.3) Choose buffers for next tile
        #     - 4.2.4) Start transfer for next input tile
        #     - 4.2.5) Update input reference for next tile

        # 4.4) Output Data Transfers
        # -----------------------------------
        # - for each output tensor:
        #   - 4.4.1) Wait for previous output tile
        #   - 4.4.2) Start transfer for current output tile
        #   - 4.4.3) Update outut reference for next tile

        setupStatements: List[CodeSnippet] = []
        openLoopStatements: List[CodeSnippet] = [CodeSnippet(self._openTileLoopTemplate, {**operatorRepresentation})]

        ingressDMAStatements: List[CodeSnippet] = []
        ingressFutures: Set[Future] = set()

        egressDMAStatements: List[CodeSnippet] = []
        egressFutures: Set[Future] = set()

        closeLoopStatements: List[CodeSnippet] = [CodeSnippet(self._closeTileLoopTemplate, {**operatorRepresentation})]
        teardownStatements: List[CodeSnippet] = []

        # 4.2) Input Data Transfers
        # -----------------------------------

        buffer_choices: List[List[CodeSnippet]] = [[], []]
        for tensorName, rectangles in dictOfArrays(tilingSchedule.inputLoadSchedule).items():
            localBuffer = ctxt.lookup(operatorRepresentation[tensorName])
            assert localBuffer._memoryLevel == self.localMemory
            assert isinstance(localBuffer, _ReferenceBuffer)
            externalBuffer = ctxt.lookup(localBuffer._referenceName)
            assert isinstance(externalBuffer, VariableBuffer)
            tensorMemoryConstraint = nodeMemoryConstraint.inputTensorMemoryConstraints[externalBuffer.name]
            externalBufferShape = tensorMemoryConstraint.memoryConstraints[self.externalMemory].shape
            assert externalBufferShape is not None

            rectangles, externalBufferShape = self._legalizeTransfers(rectangles, tuple(externalBufferShape),
                                                                      localBuffer._type.referencedType.typeWidth,
                                                                      self.isFinalMemoryLevel(tensorMemoryConstraint))

            externalBufferRef = self._hoistReference(ctxt,
                                                     externalBuffer.name + "_ref",
                                                     externalBuffer,
                                                     externalBufferShape,
                                                     override_type = VoidType)

            tensorMemoryConstraint = nodeMemoryConstraint.inputTensorMemoryConstraints[externalBuffer.name]
            l1BuffersReferences = self._hoistMultibufferReferences(ctxt, localBuffer, tensorMemoryConstraint)

            nextLocalBufferReference = self._hoistReference(ctxt, f"{tensorName}_next", l1BuffersReferences[1])

            future = self.dma.getFuture(tensorName, "ExternalToLocal")

            # 2) Load initial input tiles
            anydimAdapter = AnydimAsyncDmaTransferAdapter(self.dma)
            initialDmaTransferCalls = anydimAdapter.transfer(ctxt, externalBufferRef, localBuffer, rectangles[0].dims,
                                                             stridesFromShape(externalBufferShape),
                                                             stridesFromShape(rectangles[0].dims), "ExternalToLocal",
                                                             future, math.prod(externalBufferShape))
            if future not in ingressFutures:
                setupStatements.append(future.alloc())
            setupStatements.extend(initialDmaTransferCalls)

            # 4.1) Choose buffers for current tile (inputs and outputs)
            _buffer_choice = self._generateBufferChoice(localBuffer, l1BuffersReferences)
            for i in range(len(buffer_choices)):
                buffer_choices[i].extend(_buffer_choice[i])

            # 4.2.1) Wait for current input tile
            ingressDMAStatements.append(CodeSnippet(self._lineComment, {"comment": "Wait for current input tile"}))

            if future not in ingressFutures:
                ingressDMAStatements.append(future.wait())

            # 4.2.2) if there is a next tile:
            ingressDMAStatements.append(
                CodeSnippet(self._moveTileInCheckOpenStatement, {
                    **operatorRepresentation, "tileIdxVar": "TILING_I+1"
                }))

            # 4.2.3) Choose buffers for next tile
            ingressDMAStatements += self._switch(
                self._generateBufferChoice(nextLocalBufferReference, l1BuffersReferences), "TILING_I+1")

            # 4.2.4) Start transfer for next input tile
            ingressDMAStatements.append(CodeSnippet(self._lineComment, {"comment": "Transfer next input tile"}))

            # Allocate the future for the next transfer
            if future not in ingressFutures:
                ingressDMAStatements.append(future.alloc())

            ingressDMAStatements.extend(
                self._generateDmaTransferCalls(ctxt, tensorName, rectangles, "TILING_I+1", nextLocalBufferReference,
                                               externalBufferRef, "ExternalToLocal", future))
            # 4.2.5) Update external reference for next til
            referenceUpdate = self._generateExternalReferenceUpdate(ctxt, tensorName, rectangles, "TILING_I+1",
                                                                    externalBufferRef)
            if referenceUpdate is not None:
                ingressDMAStatements.append(referenceUpdate)

                # 3) Update input reference for second tile
                initialReferenceUpdate = CodeSnippet(referenceUpdate.template,
                                                     operatorRepresentation = {
                                                         **referenceUpdate.operatorRepresentation,
                                                         "tileIdxVar": 0,
                                                     })
                setupStatements.append(initialReferenceUpdate)

            # Close the "if there is a next tile" block
            ingressDMAStatements.append(CodeSnippet(self._moveTileInCheckCloseStatement, {}))

            # Add future to the set to prevent double wait/allocation
            ingressFutures.add(future)

        # 4.4) Output Data Transfers
        # -----------------------------------
        for tensorName, rectangles in dictOfArrays(tilingSchedule.outputLoadSchedule).items():
            localBuffer = ctxt.lookup(operatorRepresentation[tensorName])
            assert localBuffer._memoryLevel == self.localMemory
            assert isinstance(localBuffer, _ReferenceBuffer)
            externalBuffer = ctxt.lookup(localBuffer._referenceName)
            assert isinstance(externalBuffer, VariableBuffer)
            tensorMemoryConstraint = nodeMemoryConstraint.outputTensorMemoryConstraints[externalBuffer.name]
            externalBufferShape = tensorMemoryConstraint.memoryConstraints[self.externalMemory].shape
            assert externalBufferShape is not None

            rectangles, externalBufferShape = self._legalizeTransfers(rectangles, tuple(externalBufferShape),
                                                                      localBuffer._type.referencedType.typeWidth,
                                                                      self.isFinalMemoryLevel(tensorMemoryConstraint))

            externalBufferRef = self._hoistReference(ctxt,
                                                     externalBuffer.name + "_ref",
                                                     externalBuffer,
                                                     externalBufferShape,
                                                     override_type = VoidType)

            tensorMemoryConstraint = nodeMemoryConstraint.outputTensorMemoryConstraints[externalBuffer.name]
            l1BuffersReferences = self._hoistMultibufferReferences(ctxt, localBuffer, tensorMemoryConstraint)

            # 4.1) Choose buffers for current tile (inputs and outputs)
            _buffer_choice = self._generateBufferChoice(localBuffer, l1BuffersReferences)
            for i in range(len(buffer_choices)):
                buffer_choices[i].extend(_buffer_choice[i])

            # 4.4.1) Wait for previous output tile
            future = self.dma.getFuture(tensorName, "LocalToExternal")

            egressDMAStatements.append(CodeSnippet(self._lineComment, {"comment": "Wait for previous output tile"}))

            if future not in egressFutures:
                egressDMAStatements.append(future.wait())

            # 4.4.2) Start transfer for current output tile
            dmaTransferCalls = self._generateDmaTransferCalls(ctxt, tensorName, rectangles, "TILING_I", localBuffer,
                                                              externalBufferRef, "LocalToExternal", future)

            egressDMAStatements.append(CodeSnippet(self._lineComment, {"comment": "Transfer current output tile"}))
            # Allocate the future for the next transfer
            if future not in egressFutures:
                egressDMAStatements.append(future.alloc())

            egressDMAStatements.extend(dmaTransferCalls)

            # 4.4.3) Update outut reference for next tile
            referenceUpdate = self._generateExternalReferenceUpdate(ctxt, tensorName, rectangles, "TILING_I",
                                                                    externalBufferRef)
            if referenceUpdate is not None:
                egressDMAStatements.append(referenceUpdate)

            # Add future to the set to prevent double wait/allocation
            egressFutures.add(future)

        # 4.2.
        openLoopStatements += self._switch(buffer_choices, "TILING_I")

        # 1. Initialize all futures
        setupStatements = [f.init() for f in ingressFutures | egressFutures] + setupStatements
        setupStatements = [CodeSnippet(self._lineComment, {"comment": "Initialize DMA future"})] + setupStatements

        # 5. Wait for final output tile to be ready
        teardownStatements.append(CodeSnippet(self._lineComment, {"comment": "Wait for final output tile"}))
        teardownStatements.extend([f.wait() for f in egressFutures])

        # 6. Deinitialize all futures

        teardownStatements.append(CodeSnippet(self._lineComment, {"comment": "Deinitialize DMA future"}))
        teardownStatements.extend(f.deinit() for f in ingressFutures | egressFutures)

        metaInfo = TilingMetaInfo(nodeName = operatorRepresentation['nodeName'] + f"_{self.externalMemory}",
                                  nodeOps = operatorRepresentation['nodeOps'],
                                  numTiles = operatorRepresentation['numTiles'],
                                  totalNumTiles = len(tilingSchedule.outputLoadSchedule),
                                  tileIdxPtr = operatorRepresentation['tileIdxPtr'],
                                  tileIdxVar = "TILING_I",
                                  kernelLevelTiling = True)

        executionBlock = self.generateAllTilingCode(executionBlock, metaInfo, ingressDMAStatements, egressDMAStatements,
                                                    openLoopStatements, closeLoopStatements, setupStatements,
                                                    teardownStatements)

        return ctxt, executionBlock, True


class ProfilingDoubleBufferingTilingMixIn(PrototypeTilingMixIn, ProfilingPrototypeMixIn):

    @classmethod
    def generateSetupAndTeardownCode(cls, executionBlock: ExecutionBlock, metaInfo: TilingMetaInfo,
                                     setupStatements: List[CodeSnippet],
                                     teardownStatements: List[CodeSnippet]) -> ExecutionBlock:

        nodeName = metaInfo.nodeName
        totalNumTiles = metaInfo.totalNumTiles

        executionBlock.addLeft(cls._measureCycles, {
            "measurements": f"{nodeName}_ingress_dma_wait_start_measurements",
            "tileIdxVar": 0
        })

        executionBlock = cls.measurementArrayDeclaration(executionBlock, metaInfo, bufferingStr = "DB")

        executionBlock = super().generateSetupAndTeardownCode(executionBlock, metaInfo, setupStatements,
                                                              teardownStatements)
        executionBlock.addRight(cls._measureCycles, {
            "measurements": f"{nodeName}_egress_dma_wait_end_measurements",
            "tileIdxVar": totalNumTiles - 1
        })

        executionBlock = cls.injectPrintCycleDiff(executionBlock, metaInfo)

        return executionBlock

    @classmethod
    def generateLoopCode(cls, executionBlock: ExecutionBlock, metaInfo: TilingMetaInfo,
                         openLoopStatements: List[CodeSnippet], ingressDMAStatements: List[CodeSnippet],
                         egressDMAStatements: List[CodeSnippet],
                         closeLoopStatements: List[CodeSnippet]) -> ExecutionBlock:

        nodeName = metaInfo.nodeName
        tileIdxVar = metaInfo.tileIdxVar

        _openLoopStatements = [openLoopStatements[0]]
        _openLoopStatements.append(CodeSnippet(cls._measureConditionSetup, {"cond": f"{tileIdxVar} > 0"}))
        _openLoopStatements.append(
            CodeSnippet(cls._measureCycles, {
                "measurements": f"{nodeName}_ingress_dma_wait_start_measurements",
                "tileIdxVar": tileIdxVar
            }))
        _openLoopStatements.append(CodeSnippet(cls._measureConditionEnd, {}))
        _openLoopStatements += openLoopStatements[1:]

        _ingressDMAStatements = []
        _ingressDMAStatements += ingressDMAStatements
        _ingressDMAStatements.append(
            CodeSnippet(cls._measureCycles, {
                "measurements": f"{nodeName}_ingress_dma_wait_end_measurements",
                "tileIdxVar": tileIdxVar
            }))

        executionBlock = cls.kernelProfilingWrap(executionBlock, metaInfo)

        _egressDMAStatements = []
        _egressDMAStatements.append(
            CodeSnippet(cls._measureCycles, {
                "measurements": f"{nodeName}_egress_dma_wait_start_measurements",
                "tileIdxVar": f"{tileIdxVar}"
            }))
        _egressDMAStatements += egressDMAStatements
        _egressDMAStatements.append(
            CodeSnippet(cls._measureCycles, {
                "measurements": f"{nodeName}_egress_dma_wait_end_measurements",
                "tileIdxVar": f"{tileIdxVar}"
            }))

        executionBlock = super().generateLoopCode(executionBlock, metaInfo, _openLoopStatements, _ingressDMAStatements,
                                                  _egressDMAStatements, closeLoopStatements)
        return executionBlock

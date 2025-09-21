# SPDX-FileCopyrightText: 2023 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

import copy
import math
from typing import List, Set, Tuple

from Deeploy.AbstractDataTypes import VoidType
from Deeploy.DeeployTypes import CodeSnippet, ExecutionBlock, NetworkContext, NodeTemplate, OperatorRepresentation, \
    VariableBuffer, _ReferenceBuffer
from Deeploy.TilingExtension.AsyncDma import AnydimAsyncDmaTransferAdapter, AsyncDma, Future
from Deeploy.TilingExtension.CodeTransformationPasses.TilingCodeGeneration import TilingCodeGeneration
from Deeploy.TilingExtension.CodeTransformationPasses.TilingHoistingMixIn import dictOfArrays
from Deeploy.TilingExtension.CodeTransformationPasses.TilingPrototypes import TilingMetaInfo
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
    _chooseBufferTemplate = NodeTemplate("""
    // DOUBLE BUFFERING CHOOSE BUFFER
    switch((${tileIdxVar}) % 2) {
        case 0: ${reference} = (${type})${buffer_0}; break;
        case 1: ${reference} = (${type})${buffer_1}; break;
    }
    """)

    def __init__(self, externalMemory: str, localMemory: str, dma: AsyncDma):
        super().__init__(externalMemory, localMemory, dma, 2)

    def _generateBufferChoice(self, reference: VariableBuffer, buffers: List[_ReferenceBuffer],
                              tileIdxVar: str) -> CodeSnippet:
        assert len(buffers) == 2, f"Only double buffering supported. Received {len(buffers)} buffers."
        operatorRepresentation = {
            "tileIdxVar": tileIdxVar,
            "reference": reference.name,
            "type": reference._type.typeName,
            "buffer_0": buffers[0].name,
            "buffer_1": buffers[1].name,
        }
        template = self._chooseBufferTemplate
        return CodeSnippet(template, operatorRepresentation)

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
        #   - 4.4.1) Wait for current output tile
        #   - 4.4.2) Start transfer for current output tile
        #   - 4.4.3) Update outut reference for next tile

        setupStatements: List[CodeSnippet] = []
        openLoopStatements: List[CodeSnippet] = [CodeSnippet(self._openTileLoopTemplate, {**operatorRepresentation})]

        ingressDmaTransferCalls: List[CodeSnippet] = []
        ingressFutures: Set[Future] = set()

        egressDmaTransferCalls: List[CodeSnippet] = []
        egressFutures: Set[Future] = set()

        closeLoopStatements: List[CodeSnippet] = [CodeSnippet(self._closeTileLoopTemplate, {**operatorRepresentation})]
        teardownStatements: List[CodeSnippet] = []

        # 4.2) Input Data Transfers
        # -----------------------------------
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

            # 2) Load initial input tiles
            anydimAdapter = AnydimAsyncDmaTransferAdapter(self.dma)

            initialFuture = self.dma.getFuture(tensorName, "ExternalToLocal", initial = True)
            initialDmaTransferCalls = anydimAdapter.transfer(ctxt,
                                                             externalBufferRef,
                                                             localBuffer,
                                                             rectangles[0].dims,
                                                             stridesFromShape(externalBufferShape),
                                                             stridesFromShape(rectangles[0].dims),
                                                             "ExternalToLocal",
                                                             initialFuture,
                                                             math.prod(externalBufferShape),
                                                             comment = "Transfer initial input tile")

            initialDmaTransferCalls = [item for tup in initialDmaTransferCalls for item in tup]
            setupStatements.extend(initialDmaTransferCalls)

            # 4.1) Choose buffers for current tile (inputs and outputs)
            openLoopStatements.append(self._generateBufferChoice(localBuffer, l1BuffersReferences, "TILING_I"))

            # 4.2.1) Wait for current input tile
            future = self.dma.getFuture(tensorName, "ExternalToLocal")
            ingressFutures.add(future)
            ingressDmaTransferCalls.append(future.wait("Wait for current input tile"))

            # 4.2.2) if there is a next tile:
            ingressDmaTransferCalls.append(
                CodeSnippet(self._moveTileInCheckOpenStatement, {
                    **operatorRepresentation, "tileIdxVar": "TILING_I+1"
                }))

            # 4.2.3) Choose buffers for next tile
            ingressDmaTransferCalls.append(
                self._generateBufferChoice(nextLocalBufferReference, l1BuffersReferences, "TILING_I+1"))

            # 4.2.4) Start transfer for next input tile
            ingressDmaTransferCalls.extend(
                self._generateDmaTransferCalls(ctxt,
                                               tensorName,
                                               rectangles,
                                               "TILING_I+1",
                                               nextLocalBufferReference,
                                               externalBufferRef,
                                               "ExternalToLocal",
                                               future,
                                               comment = "Transfer next input tile"))
            # 4.2.5) Update external reference for next tile
            referenceUpdate = self._generateExternalReferenceUpdate(ctxt, tensorName, rectangles, "TILING_I+1",
                                                                    externalBufferRef)
            if referenceUpdate is not None:
                ingressDmaTransferCalls.append(referenceUpdate)

                # 3) Update input reference for second tile
                initialReferenceUpdate = CodeSnippet(referenceUpdate.template,
                                                     operatorRepresentation = {
                                                         **referenceUpdate.operatorRepresentation,
                                                         "tileIdxVar": 0,
                                                     })
                setupStatements.append(initialReferenceUpdate)

            # Close the "if there is a next tile" block
            ingressDmaTransferCalls.append(CodeSnippet(self._moveTileInCheckCloseStatement, {}))

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
            openLoopStatements.append(self._generateBufferChoice(localBuffer, l1BuffersReferences, "TILING_I"))

            future = self.dma.getFuture(tensorName, "LocalToExternal")
            egressFutures.add(future)

            # 4.4.1) Wait for current output tile
            egressDmaTransferCalls.append(future.wait("Wait for current output tile"))

            # 4.4.2) Start transfer for current output tile
            dmaTransferCalls = self._generateDmaTransferCalls(ctxt,
                                                              tensorName,
                                                              rectangles,
                                                              "TILING_I",
                                                              localBuffer,
                                                              externalBufferRef,
                                                              "LocalToExternal",
                                                              future,
                                                              comment = "Transfer current output tile")
            egressDmaTransferCalls.extend(dmaTransferCalls)

            # 4.4.3) Update outut reference for next tile
            referenceUpdate = self._generateExternalReferenceUpdate(ctxt, tensorName, rectangles, "TILING_I",
                                                                    externalBufferRef)
            if referenceUpdate is not None:
                egressDmaTransferCalls.append(referenceUpdate)

        # 1. Initialize all futures
        setupStatements = [f.init("Initialize DMA future") for f in ingressFutures | egressFutures] + setupStatements

        # 4. Wait for final output tile to be ready
        teardownStatements.extend([f.wait("Wait for final output tile") for f in egressFutures])

        # 5. Deinitialize all futures
        teardownStatements.extend(f.deinit("Deinitialize DMA future") for f in ingressFutures | egressFutures)

        metaInfo = TilingMetaInfo(
            nodeName = operatorRepresentation['nodeName'] + f"_{self.externalMemory}",
            nodeOps = operatorRepresentation['nodeOps'],
            numTiles = operatorRepresentation['numTiles'],
            totalNumTiles = len(tilingSchedule.outputLoadSchedule),
            tileIdxPtr = operatorRepresentation['tileIdxPtr'],
            tileIdxVar = "TILING_I",
            # TODO: The kernelLevelTiling field is used in profiling to know we are generating code around the kernel.
            #       The current implementation does this by checking whether we are at the lowest memory level,
            #       which is hardcoded by the value "L1". Change this to be memory level agnostic.
            kernelLevelTiling = self.localMemory == "L1")

        executionBlock = self.generateAllTilingCode(executionBlock, metaInfo, ingressDmaTransferCalls, [], [],
                                                    egressDmaTransferCalls, [], [], [], openLoopStatements,
                                                    closeLoopStatements, setupStatements, teardownStatements)

        return ctxt, executionBlock, True

    def generateTilingLoop(
            self, ctxt: NetworkContext, executionBlock: ExecutionBlock, nodeMemoryConstraint: NodeMemoryConstraint,
            tilingSchedules: List[TilingSchedule], variableReplacement: VariableReplacementScheme,
            operatorRepresentation: OperatorRepresentation) -> Tuple[NetworkContext, ExecutionBlock, bool]:

        flatTilingSchedule = copy.copy(tilingSchedules[0])
        for tilingSchedule in tilingSchedules[1:]:
            flatTilingSchedule += tilingSchedule

        offsetLists = list({**flatTilingSchedule.inputBaseOffsets, **flatTilingSchedule.outputBaseOffsets}.values())

        if len(offsetLists) == 0:
            return ctxt, executionBlock, False

        for offsetList in offsetLists:
            if not len(offsetList) == self.bufferCount:
                return ctxt, executionBlock, False

        numTiles, tileIdxPtr = self._hoistTileNumAndIdxPtr(ctxt, tilingSchedules)
        operatorRepresentation["numTiles"] = numTiles.name
        operatorRepresentation["tileIdxPtr"] = tileIdxPtr.name

        return self._tilingLoop(ctxt, executionBlock, nodeMemoryConstraint, flatTilingSchedule, variableReplacement,
                                operatorRepresentation)

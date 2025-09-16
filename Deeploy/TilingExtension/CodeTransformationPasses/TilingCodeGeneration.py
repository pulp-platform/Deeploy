# SPDX-FileCopyrightText: 2023 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

import copy
import math
from abc import abstractmethod
from typing import List, Optional, Tuple, TypeVar

import numpy as np

from Deeploy.CommonExtensions.CodeTransformationPasses.Closure import ClosureExecutionBlock
from Deeploy.CommonExtensions.CodeTransformationPasses.IntrospectiveCodeTransformation import \
    IntrospectiveCodeTransformationMixIn
from Deeploy.CommonExtensions.CodeTransformationPasses.MemoryAllocation import ArgumentStructGeneration
from Deeploy.DeeployTypes import CodeGenVerbosity, CodeSnippet, CodeTransformationPass, ExecutionBlock, \
    NetworkContext, NodeTemplate, OperatorRepresentation, VariableBuffer, _NoVerbosity
from Deeploy.TilingExtension.AsyncDma import AnydimAsyncDmaTransferAdapter, AsyncDma, DmaDirection, Future
from Deeploy.TilingExtension.CodeTransformationPasses.TilingHoistingMixIn import TilingHoistingMixIn
from Deeploy.TilingExtension.CodeTransformationPasses.TilingPrototypes import PrototypeTilingMixIn
from Deeploy.TilingExtension.MemoryConstraints import NodeMemoryConstraint, TensorMemoryConstraint
from Deeploy.TilingExtension.TilingCodegen import HyperRectangle, TilingSchedule, VariableReplacementScheme, \
    calculateFlatOffset, minimizeRectangle, minimizeVariableReplacement, padOffset, padShape, stridesFromShape

T = TypeVar('T')


def transposeListOfLists(listOfLists: List[List[T]]) -> List[List[T]]:
    transposedListOfLists = []
    for _list in listOfLists:
        for i, element in enumerate(_list):
            if i >= len(transposedListOfLists):
                assert i == len(transposedListOfLists)
                transposedListOfLists.append([element])
            else:
                transposedListOfLists[i].append(element)
    return transposedListOfLists


class TilingCodeGeneration(CodeTransformationPass, IntrospectiveCodeTransformationMixIn, PrototypeTilingMixIn,
                           TilingHoistingMixIn):

    _relativeOffsetReferenceUpdateTemplate = NodeTemplate("""
    // UPDATE VARIABLE ${reference}
    ${reference} += ${relativeOffset};
    """)

    _relativeOffsetReferenceUpdateTiledTemplate = NodeTemplate("""
    // UPDATE VARIABLE ${reference}
    ${reference} += ${relativeOffset}[${tileIdxVar}];
    """)

    _openTileLoopTemplate = NodeTemplate("""
    // TILING LOOP
    for (int TILING_I=${numTiles}[*${tileIdxPtr}]; TILING_I<${numTiles}[(*${tileIdxPtr})+1]; TILING_I++){
    """)

    _closeTileLoopTemplate = NodeTemplate("""
    // CLOSE TILING LOOP
    }
    *${tileIdxPtr} += 1;
    """)

    @abstractmethod
    def generateTilingLoop(
            self, ctxt: NetworkContext, executionBlock: ExecutionBlock, nodeMemoryConstraint: NodeMemoryConstraint,
            tilingSchedules: List[TilingSchedule], variableReplacement: VariableReplacementScheme,
            operatorRepresentation: OperatorRepresentation) -> Tuple[NetworkContext, ExecutionBlock, bool]:

        return ctxt, executionBlock, False

    def __init__(self, externalMemory: str, localMemory: str, dma: AsyncDma, bufferCount: int):
        self.externalMemory = externalMemory
        self.localMemory = localMemory
        self.dma = dma
        self.bufferCount = bufferCount
        TilingHoistingMixIn.__init__(self, localMemory)
        self.argStructGeneration = ArgumentStructGeneration()

    # SCHEREMO: internalPtr refers to the HIGHER memory level of a transfer,
    # e.g. in both an L2 -> L1 and L1 -> L2 transfer, the internalPtr is in L1.
    def isFinalMemoryLevel(self, tensorMemoryConstraint: TensorMemoryConstraint) -> bool:
        memoryOrder = list(tensorMemoryConstraint.memoryConstraints.keys())
        assert self.localMemory in memoryOrder, f"Memory {self.localMemory} does not exist in the tensor memory constraint {tensorMemoryConstraint}"
        if len(memoryOrder) < 2:
            return True
        return self.localMemory in memoryOrder[:2]

    def _generateDmaTransferCalls(self, ctxt: NetworkContext, tensorName: str, transfers: List[HyperRectangle],
                                  tileIdxVar: str, localBuffer: VariableBuffer, externalBuffer: VariableBuffer,
                                  direction: DmaDirection, future: Future) -> List[CodeSnippet]:
        assert all(len(transfers[0].dims) == len(rect.dims) for rect in transfers), \
            "Currently supporting only rectangles of same rank"

        assert len(transfers[0].dims) > 0, "Expecting transfers of rank greater than 0"

        assert len(transfers[0].dims) == len(externalBuffer.shape), \
            "External buffer's rank should be equal to the internal buffer's"

        anydimAdapter = AnydimAsyncDmaTransferAdapter(self.dma)

        initSnippets = anydimAdapter.transfer(ctxt, externalBuffer, localBuffer, transfers[0].dims,
                                              stridesFromShape(externalBuffer.shape),
                                              stridesFromShape(transfers[0].dims), direction, future,
                                              math.prod(externalBuffer.shape))

        templates = [snippet.template for snippet in initSnippets]
        opReprUpdates = [[] for _ in range(len(initSnippets))]

        for rect in transfers:
            snippets = anydimAdapter.transfer(ctxt, externalBuffer, localBuffer, rect.dims,
                                              stridesFromShape(externalBuffer.shape), stridesFromShape(rect.dims),
                                              direction, future, math.prod(externalBuffer.shape))
            for i, snippet in enumerate(snippets):
                opReprUpdates[i].append(snippet.operatorRepresentation)

        tiledSnippets: List[CodeSnippet] = [
            CodeSnippet(*self._tileTemplate(ctxt, opReprUpdate, template, tileIdxVar, f"{tensorName}_"))
            for template, opReprUpdate in zip(templates, opReprUpdates)
        ]

        return tiledSnippets

    def _generateExternalReferenceUpdate(self, ctxt: NetworkContext, tensorName: str, transfers: List[HyperRectangle],
                                         tileIdxVar: str, externalBuffer: VariableBuffer) -> Optional[CodeSnippet]:
        externalBufferStrides = stridesFromShape(externalBuffer.shape)
        offsets = [calculateFlatOffset(rect.offset, externalBufferStrides) for rect in transfers]
        relativeOffsets = [_next - _prev for _prev, _next in zip(offsets[:-1], offsets[1:])]

        if len(relativeOffsets) == 0 or all(offset == 0 for offset in relativeOffsets):
            return None

        operatorRepresentation: OperatorRepresentation = {"reference": externalBuffer.name, "tileIdxVar": tileIdxVar}

        if all(relativeOffsets[0] == offset for offset in relativeOffsets):
            operatorRepresentation["relativeOffset"] = relativeOffsets[0]
            template = self._relativeOffsetReferenceUpdateTemplate
        else:
            relativeOffsets.append(0)  # To have the same length as the number of tiles
            buffer = self._hoistValues(ctxt, f'{tensorName}_relativeOffset', relativeOffsets)
            operatorRepresentation["relativeOffset"] = buffer.name
            operatorRepresentation["tileIdxVar"] = tileIdxVar
            template = self._relativeOffsetReferenceUpdateTiledTemplate

        return CodeSnippet(template, operatorRepresentation)

    # TODO: Not super sure this should go here. It could be shared, but it seems a little bit too specific
    # with the `isFinalMemory` thing.
    def _legalizeTransfers(self, transfers: List[HyperRectangle], outerShape: Tuple[int, ...], typeWidth: int,
                           isFinalMemoryLevel: bool) -> Tuple[List[HyperRectangle], Tuple[int, ...]]:
        transfersCommonRank = max(len(rect.dims) for rect in transfers)
        commonRank = max(transfersCommonRank, len(outerShape))
        outerShape = padShape(outerShape, commonRank)

        minOuterShape = None

        if isFinalMemoryLevel:
            minimizedTransfers = []
            for rect in transfers:
                paddedRect = HyperRectangle(padOffset(rect.offset, commonRank), padShape(rect.dims, commonRank))
                minRect, newMinOuterShape = minimizeRectangle(paddedRect, outerShape)
                if minOuterShape is None:
                    minOuterShape = newMinOuterShape
                else:
                    if minOuterShape != newMinOuterShape:
                        rectStr = "\n".join(str(trans) for trans in transfers[:transfers.index(rect)])
                        raise RuntimeError(f"""Currently support a single minimal outer shape.
Old minOuterShape: {minOuterShape} vs. new minOuterShape {newMinOuterShape}.
New minOuterShape produced by outerDims: {outerShape} and rect: {rect}.
Old minOuterShape produced by outerDims: {outerShape} and rects:
{rectStr}""")
                minimizedTransfers.append(minRect)
        else:
            minimizedTransfers = [HyperRectangle((0,), (int(np.prod(rect.dims)),)) for rect in transfers]
            minOuterShape = (int(np.prod(outerShape)),)

        if minOuterShape is not None:
            outerShape = minOuterShape
        transfers = minimizedTransfers

        def sizeInBytes(length: int, typeWidth: int) -> int:
            return int(np.ceil((length * typeWidth) / 8))

        outerShape = outerShape[:-1] + (sizeInBytes(outerShape[-1], typeWidth),)

        inBytesTransfers = []
        for rect in transfers:
            newOffset = rect.offset[:-1] + (sizeInBytes(rect.offset[-1], typeWidth),)
            newDims = rect.dims[:-1] + (sizeInBytes(rect.dims[-1], typeWidth),)
            inBytesTransfers.append(HyperRectangle(newOffset, newDims))
        transfers = inBytesTransfers

        return transfers, outerShape

    def _tileTemplate(self, ctxt: NetworkContext, perTileOpReprs: List[OperatorRepresentation], template: NodeTemplate,
                      tileIdxVar: str, prefix: str) -> Tuple[NodeTemplate, OperatorRepresentation]:
        opRepr, hoistedNames = self._hoistOpReprUpdates(ctxt, perTileOpReprs, prefix)
        if len(hoistedNames) > 0:
            template = copy.deepcopy(template)
            self.indexVars(template.template, hoistedNames, "tileIdxVar")
            opRepr["tileIdxVar"] = tileIdxVar
        return template, opRepr

    def apply(self,
              ctxt: NetworkContext,
              executionBlock: ExecutionBlock,
              name: str,
              verbose: CodeGenVerbosity = _NoVerbosity) -> Tuple[NetworkContext, ExecutionBlock]:
        if isinstance(executionBlock, ClosureExecutionBlock):
            baseExecutionBlock = executionBlock.baseBlock
        else:
            baseExecutionBlock = executionBlock

        patternMemoryConstraint = baseExecutionBlock.patternMemoryConstraint

        if patternMemoryConstraint is None:
            return ctxt, executionBlock

        assert len(patternMemoryConstraint.nodeConstraints) == 1, "Only layerwise supported for now!"
        #assert len(baseExecutionBlock.codeSnippets) == 1, "Only layerwise supported for now!"

        nodeMemoryConstraint = patternMemoryConstraint.nodeConstraints[0]

        possibleTemplateNodes = [
            node for node in baseExecutionBlock.codeSnippets if hasattr(node.template, 'tileConstraint')
        ]

        assert len(possibleTemplateNodes) == 1, "More than one template node with TCF found"

        templateNode = possibleTemplateNodes[0]

        self._initPrefix(templateNode.operatorRepresentation['nodeName'])

        operatorRepresentation = templateNode.operatorRepresentation
        template = templateNode.template

        unraveledOpRepr = operatorRepresentation.copy()
        for key, value in unraveledOpRepr.items():
            if ctxt.is_buffer(value):
                buffer = ctxt.lookup(value)
                assert isinstance(buffer, VariableBuffer)
                unraveledOpRepr[key] = ctxt.unravelReference(buffer).name

        variableReplacement, tilingSchedules = template.tileConstraint.wrapTilingSolution(
            nodeMemoryConstraint, self.localMemory, ctxt, unraveledOpRepr)

        minimalVariableReplacement, newOpRepr = minimizeVariableReplacement(variableReplacement, operatorRepresentation)

        operatorRepresentation.update(newOpRepr)

        ctxt, executionBlock, applicable = self.generateTilingLoop(ctxt, executionBlock, nodeMemoryConstraint,
                                                                   tilingSchedules, minimalVariableReplacement,
                                                                   operatorRepresentation)
        if applicable:
            ctxt, executionBlock = self.argStructGeneration.apply(ctxt, executionBlock, name)

        self._deinitPrefix()

        return ctxt, executionBlock

# ----------------------------------------------------------------------
#
# File: TilingCodeGeneration.py
#
# Last edited: 24.10.2023
#
# Copyright (C) 2023, ETH Zurich and University of Bologna.
#
# Author: Moritz Scherer, ETH Zurich
#
# ----------------------------------------------------------------------
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the License); you may
# not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an AS IS BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from abc import abstractmethod
from typing import Dict, List, Optional, Tuple, Type

import Deeploy.CommonExtensions.DataTypes as BasicDataTypes
from Deeploy.AbstractDataTypes import Immediate, PointerClass
from Deeploy.CommonExtensions.CodeTransformationPasses.Closure import ClosureExecutionBlock
from Deeploy.CommonExtensions.CodeTransformationPasses.IntrospectiveCodeTransformation import \
    IntrospectiveCodeTransformationMixIn
from Deeploy.CommonExtensions.CodeTransformationPasses.MemoryAllocation import ArgumentStructGeneration
from Deeploy.DeeployTypes import CodeGenVerbosity, CodeTransformationPass, ConstantBuffer, ExecutionBlock, NetworkContext, \
    NodeTemplate, OperatorRepresentation, VariableBuffer, _NoVerbosity
from Deeploy.TilingExtension.CodeTransformationPasses.TilingPrototypes import PrototypeTilingMixIn
from Deeploy.TilingExtension.MemoryConstraints import NodeMemoryConstraint
from Deeploy.TilingExtension.TilingCodegen import TilingSchedule, VariableReplacementScheme, minimizeVariableReplacement


class TilingCodeGeneration(CodeTransformationPass, IntrospectiveCodeTransformationMixIn, PrototypeTilingMixIn):

    def __init__(self, targetMemLevel: str):
        self.targetMemLevel = targetMemLevel
        self.argStructGeneration = ArgumentStructGeneration()

    @abstractmethod
    def generateTilingLoop(
            self, ctxt: NetworkContext, executionBlock: ExecutionBlock, nodeMemoryConstraint: NodeMemoryConstraint,
            tilingSchedule: TilingSchedule, variableReplacement: VariableReplacementScheme,
            operatorRepresentation: OperatorRepresentation) -> Tuple[NetworkContext, ExecutionBlock, bool]:

        return ctxt, executionBlock, False

    # SCHEREMO: internalPtr refers to the HIGHER memory level of a transfer,
    # e.g. in both an L2 -> L1 and L1 -> L2 transfer, the internalPtr is in L1.
    @staticmethod
    def isFinalMemoryLevel(nodeMemoryConstraint: NodeMemoryConstraint, internalPtr: VariableBuffer) -> bool:
        externalName = internalPtr._referenceName
        tensorMemoryConstraint = nodeMemoryConstraint.tensorMemoryConstraints[externalName]
        if len(tensorMemoryConstraint.memoryConstraints.keys()) <= 2:
            return True

        finalMemoryLevels = list(tensorMemoryConstraint.memoryConstraints.keys())[:2]
        memoryLevel = internalPtr._memoryLevel

        return memoryLevel in finalMemoryLevels

    def _hoistTileIdxPtr(self,
                         ctxt: NetworkContext,
                         operatorRepresentation: OperatorRepresentation,
                         sourceMemoryLevel: str = "L2") -> str:

        newPtrName = self.prefix + operatorRepresentation['nodeName'] + "_tileIdxPtr"

        tilePtrBuffer = ctxt.VariableBuffer(newPtrName, shape = [1])
        ctxt.add(tilePtrBuffer, "local")

        _type = ctxt.lookup(self.prefix + operatorRepresentation['nodeName'] + "_numTiles")._type

        tilePtrBuffer._type = _type
        tilePtrBuffer._instance = tilePtrBuffer._type(newPtrName, ctxt)
        tilePtrBuffer._memoryLevel = sourceMemoryLevel

        tilePtrBuffer.allocTemplate = NodeTemplate("")
        tilePtrBuffer.deallocTemplate = NodeTemplate("")
        tilePtrBuffer.initTemplate = NodeTemplate("""
        ${type.referencedType.typeName} bu_${name} = 0;
        ${type.referencedType.typeName}* ${name} = &bu_${name};""")

        return newPtrName

    def _hoistNumTiles(self,
                       ctxt: NetworkContext,
                       nodeName: str,
                       tilingSchedules: List[TilingSchedule],
                       sourceMemoryLevel: str = "L2") -> str:

        newPtrName = self.prefix + nodeName + "_numTiles"

        numTiles = [len(tilingSchedule.outputLoadSchedule) for tilingSchedule in tilingSchedules]
        cumNumTiles = [0]
        for idx in list(range(len(numTiles))):
            cumNumTiles.append(cumNumTiles[-1] + numTiles[idx])

        cb = ctxt.ConstantBuffer(newPtrName, [len(cumNumTiles)], values = cumNumTiles)
        ctxt.add(cb, "global")

        minType = None
        if BasicDataTypes.uint8_t.checkValue(cumNumTiles):
            minType = BasicDataTypes.uint8_t
        elif BasicDataTypes.uint16_t.checkValue(cumNumTiles):
            minType = BasicDataTypes.uint16_t
        else:
            minType = BasicDataTypes.uint32_t

        cb._type = PointerClass(minType)
        cb._instance = cb._type(newPtrName, ctxt)
        cb._memoryLevel = sourceMemoryLevel

        return newPtrName

    def _hoistConstantAndReference(self,
                                   ctxt: NetworkContext,
                                   constBuf: ConstantBuffer,
                                   operatorRepresentation: OperatorRepresentation,
                                   nodeName: str,
                                   operatorRepresentationName: str,
                                   immediateType: Optional[Type[Immediate]] = None) -> Tuple[NetworkContext, Dict]:

        if immediateType is None:
            _type = PointerClass(BasicDataTypes.int32_t)
        else:
            _type = PointerClass(immediateType)

        name = constBuf.name

        ctxt.add(constBuf, "global")
        constBuf._type = _type
        constBuf._instance = constBuf._type(name, ctxt)
        constBuf._users = [nodeName]
        constBuf._memoryLevel = self.targetMemLevel

        refName = name + "_ref"
        reference = ctxt.hoistReference(name, refName)
        ctxt.lookup(reference)._memoryLevel = self.targetMemLevel

        operatorRepresentation[operatorRepresentationName] = refName

        return ctxt, operatorRepresentation

    
    def apply(self,
              ctxt: NetworkContext,
              executionBlock: ExecutionBlock,
              name: str,
              verbose: CodeGenVerbosity = _NoVerbosity) -> Tuple[NetworkContext, ExecutionBlock]:

        def unravelReference(ctxt: NetworkContext, name: str) -> str:

            if name not in ctxt.localObjects.keys() and name not in ctxt.globalObjects.keys():
                return name

            refBuffer = ctxt.lookup(name)
            if not hasattr(refBuffer, "_referenceName"):
                return name

            return unravelReference(ctxt, refBuffer._referenceName)

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

        operatorRepresentation = templateNode.operatorRepresentation
        unravelRep = operatorRepresentation.copy()
        for key in unravelRep.keys():

            val = unravelRep[key]
            if not isinstance(val, str):
                continue

            unravelRep[key] = unravelReference(ctxt, val)

        template = templateNode.template

        variableReplacement, tilingSchedules = template.tileConstraint.wrapTilingSolution(
            nodeMemoryConstraint, self.targetMemLevel, ctxt, unravelRep)

        minimalVariableReplacement, newNodeRep = minimizeVariableReplacement(variableReplacement,
                                                                             templateNode.operatorRepresentation)
        for key, value in newNodeRep.items():
            templateNode.operatorRepresentation[key] = value

        ctxt, executionBlock, applicable = self.generateTilingLoop(ctxt, executionBlock, nodeMemoryConstraint,
                                                                   tilingSchedules, minimalVariableReplacement,
                                                                   operatorRepresentation)
        if applicable:
            ctxt, executionBlock = self.argStructGeneration.apply(ctxt, executionBlock, name)

        return ctxt, executionBlock

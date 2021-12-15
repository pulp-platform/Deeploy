# ----------------------------------------------------------------------
#
# File: TilingVariableReplacement.py
#
# Last edited: 28.09.2023
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

import copy
from typing import Dict, List, Tuple, Type

from mako.parsetree import Expression, Node, Text

from Deeploy.AbstractDataTypes import Pointer
from Deeploy.CommonExtensions.CodeTransformationPasses.Closure import ClosureExecutionBlock
from Deeploy.CommonExtensions.CodeTransformationPasses.IntrospectiveCodeTransformation import \
    IntrospectiveCodeTransformationMixIn
from Deeploy.DeeployTypes import CodeGenVerbosity, CodeSnippet, CodeTransformationPass, ExecutionBlock, \
    NetworkContext, NodeTemplate, OperatorRepresentation, TransientBuffer, _NoVerbosity
from Deeploy.TilingExtension.MemoryConstraints import NodeMemoryConstraint
from Deeploy.TilingExtension.TilingCodegen import TilingSchedule, VariableReplacementScheme, minimizeVariableReplacement


class TilingVariableReplacement(CodeTransformationPass, IntrospectiveCodeTransformationMixIn):

    _prefix = "TILING_REPLACED_"

    def __init__(self, targetMemLevel: str):
        self.targetMemLevel = targetMemLevel
        self._name: str

    @property
    def prefix(self):
        return self._prefix + f"{self._name}_" + self.targetMemLevel + "_"

    def _dereferencePointer(self, nodes: List[Node], name: str) -> List[Node]:
        instanceIdxs = [idx for idx, node in enumerate(nodes) if isinstance(node, Expression) and node.text == name]

        for offset, idx in enumerate(instanceIdxs):
            text = Text("*", source = "*", lineno = 0, pos = 0, filename = None)
            nodes.insert(offset + idx, text)

        return nodes

    def _replaceImmediate(self, ctxt: NetworkContext, operatorRepresentation: OperatorRepresentation,
                          variableReplacement: Tuple[str,
                                                     List], dataType: Type[Pointer]) -> Tuple[NetworkContext, Dict]:

        varName = variableReplacement[0]
        varVal = variableReplacement[1]

        newConstName = self.prefix + varName
        newRefName = self.prefix + "ref_" + varName

        cb = ctxt.ConstantBuffer(newConstName, shape = (len(varVal),), values = varVal)
        ctxt.add(cb, "global")

        cb._type = dataType
        cb._instance = dataType(newConstName, ctxt)
        cb._memoryLevel = self.targetMemLevel

        reference = ctxt.hoistReference(newConstName, newRefName)
        ctxt.lookup(reference)._memoryLevel = self.targetMemLevel

        operatorRepresentation[varName] = reference

        return ctxt, operatorRepresentation

    def _hoistTileReference(self, ctxt: NetworkContext, reference: str, name: str, offset: int) -> NetworkContext:

        refName = ctxt.hoistReference(reference, name)
        refBuf = ctxt.lookup(refName)

        staticBuf = ctxt.lookup(f"MEMORYARENA_{self.targetMemLevel}")

        refBuf.allocTemplate = NodeTemplate(" \
        ${type.typeName} ${name} = (${type.typeName}) " + f"((char*){str(staticBuf._instance)} + {offset});")
        refBuf._memoryLevel = self.targetMemLevel

        return ctxt

    def _replaceReferences(self, ctxt: NetworkContext, operatorRepresentation: OperatorRepresentation,
                           tilingSchedule: TilingSchedule, name: str) -> Tuple[NetworkContext, Dict]:

        def unravelOldRef(refName):
            oldBuf = ctxt.lookup(refName)
            if hasattr(oldBuf, "_referenceName"):
                return unravelOldRef(oldBuf._referenceName)
            return oldBuf.name

        newRefName = self.prefix + "ref_" + name
        oldRefName = operatorRepresentation[name]

        if name in tilingSchedule.inputBaseOffsets:
            offset = tilingSchedule.inputBaseOffsets[name]
        elif name in tilingSchedule.outputBaseOffsets:
            offset = tilingSchedule.outputBaseOffsets[name]
        else:
            raise RuntimeError(f"Name {name} not found in TilingSchedule {tilingSchedule}")

        unravelRef = unravelOldRef(oldRefName)

        ctxt = self._hoistTileReference(ctxt, unravelRef, newRefName, offset[0])
        operatorRepresentation[name] = newRefName

        return ctxt, operatorRepresentation

    def _replaceTransients(self, ctxt: NetworkContext, operatorRepresentation: OperatorRepresentation,
                           nodeMemoryConstraint: NodeMemoryConstraint, name: str) -> Tuple[NetworkContext, Dict]:

        memoryConstraints = nodeMemoryConstraint.tensorMemoryConstraints[operatorRepresentation[name]].memoryConstraints
        assert len(memoryConstraints
                  ) == 1, f"Tiled transient buffer {operatorRepresentation[name]} has more than one memory level!"
        key = list(memoryConstraints.keys())[0]
        constraint = memoryConstraints[key]
        assert constraint.addrSpace is not None, f"Address space of {constraint} cannot be None!"
        offset = constraint.addrSpace[0]

        refBuf = ctxt.lookup(operatorRepresentation[name])

        if refBuf._memoryLevel != self.targetMemLevel:
            return ctxt, operatorRepresentation

        staticBuf = ctxt.lookup(f"MEMORYARENA_{self.targetMemLevel}")

        refBuf.allocTemplate = NodeTemplate(" \
        ${type.typeName} ${name} = (${type.typeName}) " + f"((char*){str(staticBuf._instance)} + {offset});")
        refBuf.deallocTemplate = NodeTemplate("")
        refBuf._memoryLevel = self.targetMemLevel

        return ctxt, operatorRepresentation

    def _replaceTiledExpressions(self, ctxt: NetworkContext, templateNode: CodeSnippet,
                                 variableReplacement: VariableReplacementScheme, tilingSchedule: TilingSchedule,
                                 nodeMemoryConstraint: NodeMemoryConstraint) -> NetworkContext:

        operatorRepresentation = templateNode.operatorRepresentation
        template = templateNode.template

        immediateList = [(key, value)
                         for key, value in variableReplacement.perTileReplacements.items()
                         if type(operatorRepresentation[key]) != str]

        inoutSchedule = {**tilingSchedule.inputBaseOffsets, **tilingSchedule.outputBaseOffsets}
        variableList = [key for key, value in inoutSchedule.items() if type(operatorRepresentation[key]) == str]

        transientBufferList = []
        for key, value in operatorRepresentation.items():
            if not isinstance(value, str):
                continue
            if (ctxt.is_local(value) and isinstance(ctxt.lookup(value), TransientBuffer)):
                transientBufferList.append(key)

        parseTree = IntrospectiveCodeTransformationMixIn._generateParseTree(template)
        newParseTree = copy.copy(parseTree)
        nodes = parseTree.nodes

        newNodes = copy.copy(nodes)

        for rep in immediateList:
            ctxt, operatorRepresentation = self._replaceImmediate(ctxt, operatorRepresentation, rep,
                                                                  variableReplacement.replacementTypes[rep[0]])
            newNodes = self._dereferencePointer(newNodes, rep[0])

        for rep in variableList:
            ctxt, operatorRepresentation = self._replaceReferences(ctxt, operatorRepresentation, tilingSchedule, rep)

        for rep in transientBufferList:
            ctxt, operatorRepresentation = self._replaceTransients(ctxt, operatorRepresentation, nodeMemoryConstraint,
                                                                   rep)

        newParseTree.nodes = newNodes
        IntrospectiveCodeTransformationMixIn._reconstructCode(template, newParseTree)

        return ctxt

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

        self._name = name

        if isinstance(executionBlock, ClosureExecutionBlock):
            baseExecutionBlock = executionBlock.baseBlock
        else:
            baseExecutionBlock = executionBlock

        patternMemoryConstraint = baseExecutionBlock.patternMemoryConstraint

        if patternMemoryConstraint is None:
            return ctxt, executionBlock

        assert len(patternMemoryConstraint.nodeConstraints) == 1, "Only layerwise supported for now!"
        #assert len(executionBlock.codeSnippets) == 1, "Only layerwise supported for now!"

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

        flatTilingSchedule = copy.copy(tilingSchedules[0])
        for tilingSchedule in tilingSchedules[1:]:
            flatTilingSchedule += tilingSchedule

        ctxt = self._replaceTiledExpressions(ctxt, templateNode, minimalVariableReplacement, flatTilingSchedule,
                                             nodeMemoryConstraint)

        for codeSnippet in executionBlock.codeSnippets:

            template, nRep = codeSnippet.template, codeSnippet.operatorRepresentation

            if not "closureStructArgs" in nRep:
                continue

            keyList = {}

            for key in list(flatTilingSchedule.inputBaseOffsets.keys()) + list(
                    flatTilingSchedule.outputBaseOffsets.keys()):
                keyList[unravelRep[key]] = operatorRepresentation[key]

            for key in copy.copy(nRep['closureStructArgs'].value).keys():
                if nRep['closureStructArgs'].value[key].referenceName in keyList.keys():
                    nRep['closureStructArgs'].value[key] = type(nRep['closureStructArgs'].value[key])(
                        keyList[nRep['closureStructArgs'].value[key].referenceName], ctxt)

        return ctxt, executionBlock

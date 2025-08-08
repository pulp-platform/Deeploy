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
import itertools
from typing import List, Tuple

from Deeploy.AbstractDataTypes import Struct
from Deeploy.CommonExtensions.CodeTransformationPasses.Closure import ClosureExecutionBlock
from Deeploy.CommonExtensions.CodeTransformationPasses.IntrospectiveCodeTransformation import \
    IntrospectiveCodeTransformationMixIn
from Deeploy.DeeployTypes import CodeGenVerbosity, CodeSnippet, CodeTransformationPass, ExecutionBlock, \
    NetworkContext, NodeTemplate, OperatorRepresentation, TransientBuffer, VariableBuffer, _NoVerbosity, \
    _ReferenceBuffer
from Deeploy.TilingExtension.CodeTransformationPasses.TilingHoistingMixIn import TilingHoistingMixIn
from Deeploy.TilingExtension.MemoryConstraints import NodeMemoryConstraint
from Deeploy.TilingExtension.TilingCodegen import TilingSchedule, VariableReplacementScheme, minimizeVariableReplacement


class TilingVariableReplacement(CodeTransformationPass, IntrospectiveCodeTransformationMixIn, TilingHoistingMixIn):

    def __init__(self, targetMemLevel: str):
        self.targetMemLevel = targetMemLevel
        TilingHoistingMixIn.__init__(self, targetMemLevel)

    @property
    def arenaName(self):
        return f"MEMORYARENA_{self.targetMemLevel}"

    def _arenaAllocate(self, ctxt: NetworkContext, buffer: VariableBuffer, offset: int) -> VariableBuffer:
        arena = ctxt.lookup(self.arenaName)
        buffer.allocTemplate = NodeTemplate(" \
        ${type.typeName} ${name} = (${type.typeName}) " + f"((char*){str(arena._instance)} + {offset});")
        buffer.deallocTemplate = NodeTemplate("")
        return buffer

    def _replaceTransients(self, ctxt: NetworkContext, operatorRepresentation: OperatorRepresentation,
                           nodeMemoryConstraint: NodeMemoryConstraint) -> NetworkContext:
        for value in operatorRepresentation.values():
            if not (isinstance(value, str) and ctxt.is_local(value)):
                continue

            buffer = ctxt.lookup(value)

            if not (isinstance(buffer, TransientBuffer) and buffer._memoryLevel == self.targetMemLevel):
                continue

            memoryConstraints = nodeMemoryConstraint.tensorMemoryConstraints[buffer.name].memoryConstraints
            assert len(memoryConstraints) == 1, f"Tiled transient buffer {buffer.name} has more than one memory level!"
            constraint = next(iter(memoryConstraints.values()))
            assert constraint.addrSpace is not None, f"Address space of {constraint} cannot be None!"
            offset = constraint.addrSpace[0]
            self._arenaAllocate(ctxt, buffer, offset)

        return ctxt

    def _replaceVariableReplacements(self, ctxt: NetworkContext, snippet: CodeSnippet,
                                     variableReplacement: VariableReplacementScheme) -> NetworkContext:
        operatorRepresentation = snippet.operatorRepresentation
        template = snippet.template

        replacedVars = []

        for name, values in variableReplacement.perTileReplacements.items():
            # Case where we have already replaced the variable
            if isinstance(operatorRepresentation[name], str):
                continue
            _type = variableReplacement.replacementTypes[name]
            # LMACAN: Hoist values expects integers (should be the only thing we deal with for now...)
            intValues = [int(v) for v in values]
            assert all(intV == v for intV, v in zip(intValues, values)), f"Received non-int values"
            buff = self._hoistValues(ctxt, name, intValues, _type.referencedType)
            ref = self._hoistReference(ctxt, name + "_ref", buff)
            operatorRepresentation[name] = ref.name
            replacedVars.append(name)

        self.dereferenceVars(template.template, replacedVars)

        return ctxt

    def _replaceTiledTensors(self, ctxt: NetworkContext, snippet: CodeSnippet,
                             tilingSchedule: TilingSchedule) -> NetworkContext:
        operatorRepresentation = snippet.operatorRepresentation

        for name, offsets in itertools.chain(tilingSchedule.inputBaseOffsets.items(),
                                             tilingSchedule.outputBaseOffsets.items()):
            buffer = ctxt.lookup(operatorRepresentation[name])
            assert isinstance(buffer, VariableBuffer)
            unraveledBuffer = ctxt.unravelReference(buffer)

            ref = self._hoistReference(ctxt, name + "_ref", unraveledBuffer)
            ref = self._arenaAllocate(ctxt, ref, offsets[0])
            operatorRepresentation[name] = ref.name

        return ctxt

    def apply(self,
              ctxt: NetworkContext,
              executionBlock: ExecutionBlock,
              name: str,
              verbose: CodeGenVerbosity = _NoVerbosity) -> Tuple[NetworkContext, ExecutionBlock]:
        self._initPrefix(name)

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

        possibleSnippets = [
            node for node in baseExecutionBlock.codeSnippets if hasattr(node.template, 'tileConstraint')
        ]

        assert len(possibleSnippets) == 1, "More than one template node with TCF found"

        snippet = possibleSnippets[0]
        operatorRepresentation = snippet.operatorRepresentation
        template = snippet.template

        unraveledOpRepr = {
            key: ctxt.unravelReference(ctxt.lookup(value)).name if ctxt.is_buffer(value) else value
            for key, value in operatorRepresentation.items()
        }

        variableReplacement, tilingSchedules = template.tileConstraint.wrapTilingSolution(
            nodeMemoryConstraint, self.targetMemLevel, ctxt, unraveledOpRepr)

        minimalVariableReplacement, newOpRepr = minimizeVariableReplacement(variableReplacement, operatorRepresentation)
        operatorRepresentation.update(newOpRepr)

        flatTilingSchedule = copy.copy(tilingSchedules[0])
        for tilingSchedule in tilingSchedules[1:]:
            flatTilingSchedule += tilingSchedule

        ctxt = self._replaceVariableReplacements(ctxt, snippet, minimalVariableReplacement)
        ctxt = self._replaceTiledTensors(ctxt, snippet, flatTilingSchedule)
        ctxt = self._replaceTransients(ctxt, operatorRepresentation, nodeMemoryConstraint)

        tilingReplacedRefMap = {}
        for key in list(flatTilingSchedule.inputBaseOffsets.keys()) + list(flatTilingSchedule.outputBaseOffsets.keys()):
            tilingReplacedRefMap[unraveledOpRepr[key]] = operatorRepresentation[key]

        # Swap any original tensor occurances with the tiled targetMemLevel-local tensor
        for codeSnippet in executionBlock.codeSnippets:
            template, opRepr = codeSnippet.template, codeSnippet.operatorRepresentation

            for key, value in opRepr.items():
                if isinstance(value, str) and value in tilingReplacedRefMap:
                    opRepr[key] = tilingReplacedRefMap[value]

            if "closureStructArgs" in opRepr:
                closureArgsStruct: Struct = opRepr['closureStructArgs']
                structDict = closureArgsStruct.value

                for key, value in structDict.items():
                    if value.referenceName in tilingReplacedRefMap:
                        structDict[key] = type(value)(tilingReplacedRefMap[value.referenceName], ctxt)

        self._deinitPrefix()

        return ctxt, executionBlock


class TilingVariableReplacementUpdate(CodeTransformationPass, IntrospectiveCodeTransformationMixIn,
                                      TilingHoistingMixIn):

    _updateReferenceTemplate = NodeTemplate("""
    // UPDATE VARIABLE ${reference}
    *${reference} = ${baseReference}[${tileIdxVar}];
    """)

    def __init__(self, targetMemLevel: str, tileIdxVar: str = "TILING_I"):
        super().__init__()
        self.tileIdxVar = tileIdxVar
        self.targetMemLevel = targetMemLevel

    def _generateVariableUpdates(self, variableReplacement: VariableReplacementScheme, ctxt: NetworkContext,
                                 operatorRepresentation: OperatorRepresentation) -> List[CodeSnippet]:
        updates = []
        for key in variableReplacement.perTileReplacements.keys():
            ref = ctxt.lookup(operatorRepresentation[key])
            assert isinstance(ref, _ReferenceBuffer)
            updates.append(
                CodeSnippet(self._updateReferenceTemplate, {
                    "reference": ref.name,
                    "tileIdxVar": self.tileIdxVar,
                    "baseReference": ref._referenceName
                }))
        return updates

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

        nodeMemoryConstraint = patternMemoryConstraint.nodeConstraints[0]

        possibleSnippets = [
            node for node in baseExecutionBlock.codeSnippets if hasattr(node.template, 'tileConstraint')
        ]

        assert len(possibleSnippets) == 1, "More than one template node with TCF found"

        snippet = possibleSnippets[0]
        operatorRepresentation = snippet.operatorRepresentation
        template = snippet.template

        unraveledOpRepr = {
            key: ctxt.unravelReference(ctxt.lookup(value)).name if ctxt.is_buffer(value) else value
            for key, value in operatorRepresentation.items()
        }

        variableReplacement, _ = template.tileConstraint.wrapTilingSolution(nodeMemoryConstraint, self.targetMemLevel,
                                                                            ctxt, unraveledOpRepr)

        minimalVariableReplacement, newOpRepr = minimizeVariableReplacement(variableReplacement, operatorRepresentation)
        operatorRepresentation.update(newOpRepr)

        updates = self._generateVariableUpdates(minimalVariableReplacement, ctxt, operatorRepresentation)

        for snippet in updates:
            executionBlock.addLeft(snippet.template, snippet.operatorRepresentation)

        return super().apply(ctxt, executionBlock, name, verbose)

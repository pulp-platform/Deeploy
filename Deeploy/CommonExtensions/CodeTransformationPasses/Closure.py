# ----------------------------------------------------------------------
#
# File: Closure.py
#
# Last edited: 12.06.2023
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

from typing import Dict, Optional, Tuple, Type, Union

from Deeploy.AbstractDataTypes import Immediate, Pointer, PointerClass, Struct, StructClass, VoidType
from Deeploy.CommonExtensions.CodeTransformationPasses.IntrospectiveCodeTransformation import \
    IntrospectiveCodeTransformationMixIn
from Deeploy.CommonExtensions.CodeTransformationPasses.MemoryAllocation import ArgumentStructGeneration
from Deeploy.DeeployTypes import CodeGenVerbosity, CodeTransformationPass, ExecutionBlock, NetworkContext, \
    NodeTemplate, StructBuffer, TransientBuffer, _NoVerbosity

# SCHEREMO: example template for a function closure call
_closureCallTemplate = NodeTemplate("""
// ${closureName} CLOSURE CALL
${closureName}(&${closureStructArgName});
""")

_closureTemplate = NodeTemplate("""
static void ${closureName}(void* ${closureName}_args){
// CLOSURE ARG CAST
${closureStructArgs.typeName}* args = (${closureStructArgs.typeName}*) ${closureStructArgName};
% for argName, argType in closureStructArgs.value.items():
${argType.typeName} ${argName} = args->${argName};
% endfor

// CLOSURE FUNCTION CALL
${functionCall}

// CLOSURE ARG WRITEBACK
% if writeback:
% for argName, argType in closureStructArgs.value.items():
args->${argName} = ${argName};
% endfor
% endif
}
""")

_closureWriteBackTemplate = NodeTemplate("""
// CLOSURE ARG WRITEBACK
% for argName, argType in closureStructArgs.value.items():
${argName} = ${closureStructArgName}.${argName};
% endfor
""")

_closureStructDefTemplate = NodeTemplate("""
typedef struct ${closureStructArgs._typeDefRepr()} ${closureStructArgName}_t;
""")


class ClosureExecutionBlock(ExecutionBlock):

    def __init__(self, nodeTemplate = None, closureBlock: Optional[ExecutionBlock] = None):
        super().__init__(nodeTemplate)
        self.closureBlock = closureBlock

    @property
    def baseBlock(self):
        if isinstance(self.closureBlock, ClosureExecutionBlock):
            return self.closureBlock.baseBlock
        return self.closureBlock


class ClosureGeneration(CodeTransformationPass, IntrospectiveCodeTransformationMixIn):

    closureStructArgs: Struct

    def __init__(self,
                 closureCallTemplate: NodeTemplate = _closureCallTemplate,
                 closureSuffix = "_closure",
                 writeback: bool = True,
                 generateStruct: bool = True):
        super().__init__()
        self.closureSuffix = closureSuffix
        self.closureTemplate = _closureTemplate
        self.closureCallTemplate = closureCallTemplate
        self.closureStructDefTemplate = _closureStructDefTemplate
        self.closureWriteBackTemplate = _closureWriteBackTemplate
        self.writeback = writeback
        self.generateStruct = generateStruct

    # Don't override this
    def _generateClosureStruct(self, ctxt: NetworkContext, executionBlock: ExecutionBlock):

        # Add closure struct info to operatorRepresentation
        closureStructArgsType: Dict[str, Type[Union[Pointer, Immediate, Struct]]] = {}
        closureStruct: Dict[str, Union[Pointer, Immediate, Struct]] = {}
        makoDynamicReferences = self.extractDynamicReferences(ctxt, executionBlock, True)

        for arg in list(dict.fromkeys(makoDynamicReferences)):
            ref = ctxt.lookup(arg)
            if isinstance(ref, TransientBuffer):
                closureStructArgsType[ctxt._mangle(arg)] = PointerClass(VoidType)
            elif not isinstance(ref, StructBuffer):
                closureStructArgsType[ctxt._mangle(arg)] = ref._type

            if not isinstance(ref, StructBuffer):
                closureStruct[ctxt._mangle(arg)] = arg

        structClass = StructClass(self.closureName + "_args_t", closureStructArgsType)
        self.closureStructArgType = structClass
        self.closureStructArgs = structClass(closureStruct, ctxt)

    # Don't override this
    def _generateClosureCtxt(self, ctxt: NetworkContext, nodeName: str) -> NetworkContext:

        ret = ctxt.hoistStruct(self.closureStructArgs, self.closureName + "_args", self.closureStructArgType)
        ctxt.lookup(ret)._users.append(nodeName)

        allArgs = {
            "closureName": self.closureName,
            "functionCall": self.functionCall,
            "closureStructArgs": ctxt.lookup(self.closureName + "_args").structDict,
            "closureStructArgName": self.closureName + "_args",
            "writeback": self.writeback
        }

        # SCHEREMO: These are global definitions
        closure = self.closureTemplate.generate(allArgs)
        closureStructDef = self.closureStructDefTemplate.generate(allArgs)
        closureStructName = self.closureName + '_args_t'

        ctxt.hoistGlobalDefinition(closureStructName, closureStructDef)
        ctxt.hoistGlobalDefinition(self.closureName, closure)

        return ctxt

    # Don't override this
    def _generateClosureCall(self, ctxt: NetworkContext, executionBlock: ExecutionBlock,
                             nodeName: str) -> Tuple[NetworkContext, ExecutionBlock]:

        allArgs = {
            "closureName": self.closureName,
            "functionCall": self.functionCall,
            "closureStructArgs": ctxt.lookup(self.closureName + "_args").structDict,
            "closureStructArgName": self.closureName + "_args",
            "writeback": self.writeback
        }

        executionBlock = ClosureExecutionBlock(None, executionBlock)

        # SCHEREMO: These replace the function call
        executionBlock.addLeft(self.closureCallTemplate, allArgs)
        if self.writeback:
            executionBlock.addRight(self.closureWriteBackTemplate, allArgs)
        if self.generateStruct:
            ctxt, executionBlock = ArgumentStructGeneration().apply(ctxt, executionBlock, nodeName, _NoVerbosity)

        return ctxt, executionBlock

    def apply(self,
              ctxt: NetworkContext,
              executionBlock: ExecutionBlock,
              name: str,
              verbose: CodeGenVerbosity = _NoVerbosity) -> Tuple[NetworkContext, ExecutionBlock]:
        self.closureName = name + self.closureSuffix
        self.functionCall = executionBlock.generate(ctxt)
        self._generateClosureStruct(ctxt, executionBlock)
        ctxt = self._generateClosureCtxt(ctxt, name)
        ctxt, executionBlock = self._generateClosureCall(ctxt, executionBlock, name)
        return ctxt, executionBlock


class MemoryAwareClosureGeneration(ClosureGeneration):

    def __init__(self,
                 closureCallTemplate: NodeTemplate = _closureCallTemplate,
                 closureSuffix = "_closure",
                 writeback: bool = True,
                 generateStruct: bool = True,
                 startRegion: str = "L2",
                 endRegion: str = "L1"):
        super().__init__(closureCallTemplate, closureSuffix, writeback, generateStruct)
        self.startRegion = startRegion
        self.endRegion = endRegion

    # Don't override this
    def _generateClosureStruct(self, ctxt: NetworkContext, executionBlock: ExecutionBlock):

        # Add closure struct info to operatorRepresentation
        closureStructArgsType = {}
        closureStruct = {}
        makoDynamicReferences = self.extractDynamicReferences(ctxt, executionBlock, True)

        filteredMakoDynamicReferences = []

        for ref in makoDynamicReferences:
            buf = ctxt.lookup(ref)
            if not hasattr(buf, "_memoryLevel") or buf._memoryLevel is None:
                filteredMakoDynamicReferences.append(ref)
                continue

            if buf._memoryLevel == self.startRegion or buf._memoryLevel != self.endRegion:
                filteredMakoDynamicReferences.append(ref)

        for arg in list(dict.fromkeys(filteredMakoDynamicReferences)):
            ref = ctxt.lookup(arg)
            if isinstance(ref, TransientBuffer):
                closureStructArgsType[ctxt._mangle(arg)] = PointerClass(VoidType)
            elif not isinstance(ref, StructBuffer):
                closureStructArgsType[ctxt._mangle(arg)] = ref._type

            if not isinstance(ref, StructBuffer):
                closureStruct[ctxt._mangle(arg)] = arg

        structClass = StructClass(self.closureName + "_args_t", closureStructArgsType)
        self.closureStructArgType = structClass
        self.closureStructArgs = self.closureStructArgType(closureStruct, ctxt)

# Copyright (C) 2025, ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

import types
from typing import Dict, List

import mako.codegen as codegen
from mako.lexer import Lexer
from mako.parsetree import Expression, TemplateNode, Text
from mako.template import Template

from Deeploy.AbstractDataTypes import Pointer, Struct
from Deeploy.DeeployTypes import ExecutionBlock, NetworkContext, OperatorRepresentation, VariableBuffer

_NULL: str = "NULL"


class IntrospectiveCodeTransformationMixIn():

    parseTreeDict: Dict[int, TemplateNode] = {}

    @staticmethod
    def _generateParseTree(template: Template) -> TemplateNode:
        return Lexer(template._source).parse()

    @staticmethod
    def _reconstructCode(template: Template, node: TemplateNode) -> Template:
        lexer = Lexer(template._source)
        source = codegen.compile(
            node,
            template.uri,
            None,
            default_filters = template.default_filters,
            buffer_filters = template.buffer_filters,
            imports = template.imports,
            future_imports = template.future_imports,
            source_encoding = lexer.encoding,
            generate_magic_comment = True,
            strict_undefined = template.strict_undefined,
            enable_loop = template.enable_loop,
            reserved_names = template.reserved_names,
        )
        module = types.ModuleType(template.module_id)
        code = compile(source, template.module_id, "exec")
        exec(code, module.__dict__, module.__dict__)

        template._code = code
        template.module = module
        template.callable_ = template.module.render_body
        return template

    @staticmethod
    def _indexPointer(parseTree: TemplateNode, ptrName: str, index: str) -> TemplateNode:
        indexes = [i for i, node in enumerate(parseTree.nodes) if isinstance(node, Expression) and node.text == ptrName]

        for offset, idx in enumerate(indexes):
            bracketOpen = Text("[", source = "[", lineno = 0, pos = 0, filename = None)
            indexExpr = Expression(index, '', source = index, lineno = 0, pos = 0, filename = None)
            bracketClose = Text("]", source = "]", lineno = 0, pos = 0, filename = None)
            parseTree.nodes.insert(idx + 3 * offset + 1, bracketOpen)
            parseTree.nodes.insert(idx + 3 * offset + 2, indexExpr)
            parseTree.nodes.insert(idx + 3 * offset + 3, bracketClose)

        return parseTree

    @staticmethod
    def indexVars(template: Template, varNames: List[str], index: str) -> None:
        if len(varNames) == 0:
            return
        parseTree = IntrospectiveCodeTransformationMixIn._generateParseTree(template)
        for name in varNames:
            parseTree = IntrospectiveCodeTransformationMixIn._indexPointer(parseTree, name, index)
        IntrospectiveCodeTransformationMixIn._reconstructCode(template, parseTree)

    @staticmethod
    def _dereferencePointer(parseTree: TemplateNode, ptrName: str) -> TemplateNode:
        indexes = [i for i, node in enumerate(parseTree.nodes) if isinstance(node, Expression) and node.text == ptrName]

        for offset, idx in enumerate(indexes):
            text = Text("*", source = "*", lineno = 0, pos = 0, filename = None)
            parseTree.nodes.insert(idx + offset, text)

        return parseTree

    @staticmethod
    def dereferenceVars(template: Template, varNames: List[str]) -> None:
        if len(varNames) == 0:
            return
        parseTree = IntrospectiveCodeTransformationMixIn._generateParseTree(template)
        for name in varNames:
            parseTree = IntrospectiveCodeTransformationMixIn._dereferencePointer(parseTree, name)
        IntrospectiveCodeTransformationMixIn._reconstructCode(template, parseTree)

    def extractDynamicReferences(self,
                                 ctxt: NetworkContext,
                                 executionBlock: ExecutionBlock = None,
                                 unrollStructs = False,
                                 includeGobalReferences = False):

        makoDynamicReferences = []
        for codeSnippet in executionBlock.codeSnippets:
            template, operatorRepresentation = codeSnippet.template, codeSnippet.operatorRepresentation

            newRefs = self._extractDynamicExpressions(ctxt, operatorRepresentation, template.template, unrollStructs,
                                                      includeGobalReferences)

            makoDynamicReferences += newRefs

        ret = IntrospectiveCodeTransformationMixIn._fixCtxtOrdering(ctxt, list(dict.fromkeys(makoDynamicReferences)))

        return ret

    @staticmethod
    def _fixCtxtOrdering(ctxt: NetworkContext, nameList: List[str]) -> List[str]:

        orderList = [*ctxt.globalObjects.keys(), *ctxt.localObjects.keys()]
        _nameList = sorted(nameList.copy(), key = lambda key: orderList.index(key))

        return _nameList

    def _extractDynamicExpressions(self,
                                   ctxt: NetworkContext,
                                   operatorRepresentation: OperatorRepresentation,
                                   template: Template,
                                   unrollStructs = False,
                                   includeGobalReferences = False):
        codeHash = hash(template._source)

        if codeHash in self.parseTreeDict.keys():
            makoParseTree = self.parseTreeDict[codeHash]
        else:
            # Parse the user-provided template
            makoParseTree = IntrospectiveCodeTransformationMixIn._generateParseTree(template)
            self.parseTreeDict[codeHash] = makoParseTree

        # Filter parsing tree for expressions
        makoExpressions = [node.text for node in makoParseTree.nodes if isinstance(node, Expression)]

        # Filter represented expressions
        representedExpressions = [
            operatorRepresentation[expr] for expr in makoExpressions if expr in operatorRepresentation
        ]

        # Filter buffers from expressions
        references = [expr for expr in representedExpressions if ctxt.is_buffer(expr)]

        if unrollStructs:

            def _unrollStructReferences(val: Struct) -> List[str]:
                assert isinstance(val, Struct)
                # Recursively unroll struct references
                structReferences = []
                for field in val.value.values():
                    if isinstance(field, Struct):
                        structReferences += _unrollStructReferences(field)
                    elif isinstance(field, Pointer) and field.referenceName != _NULL:
                        structReferences.append(field.referenceName)
                return structReferences

            # Unroll local struct references
            for ref in references:
                if hasattr(ctxt.lookup(ref), "structDict"):
                    references += _unrollStructReferences(ctxt.lookup(ref).structDict)

        # Filter expressions for local variables contained in operatorRepresentation
        localReferences = [ref for ref in references if ctxt.is_local(ref)]

        # Filter expressions for global variables contained in operatorRepresentation
        globalReferences = [ref for ref in references if ctxt.is_global(ref)]

        # Filter for dynamically allocated tensors
        dynamicLocalReferences = [ref for ref in localReferences if ctxt.lookup(ref)._deploy]
        dynamicGlobalReferences = [ref for ref in globalReferences if isinstance(ctxt.lookup(ref), VariableBuffer)]

        if includeGobalReferences:
            return dynamicLocalReferences + dynamicGlobalReferences
        else:
            return dynamicLocalReferences

# SPDX-FileCopyrightText: 2023 ETH Zurich and University of Bologna
#
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
    """A mix-in class providing introspective code transformation capabilities for template-based code generation.

    This class enables analysis and manipulation of template code by parsing it into an abstract syntax tree (AST),
    allowing for dynamic transformations such as variable indexing, dereferencing, and extraction of dynamic references.
    It is designed to work with template objects and their parse trees, supporting advanced code introspection and
    modification tasks commonly required in code generation frameworks.

    Key Features
    ------------
    - Parse template source code into a tree structure for introspection.
    - Programmatically index or dereference variables within templates.
    - Extract dynamic references (e.g., buffers, tensors) used in code blocks.
    - Support for unrolling struct references and distinguishing between local/global context.
    - Efficient caching of parse trees for repeated template analysis.

    Intended Usage
    --------------
    This mix-in is intended to be used with classes that manage code templates, enabling them to inspect and transform
    template code at runtime. It is particularly useful in scenarios where code generation must adapt dynamically to
    context or user input, such as in neural network frameworks or domain-specific languages.
    """

    parseTreeDict: Dict[int, TemplateNode] = {}

    @staticmethod
    def _generateParseTree(template: Template) -> TemplateNode:
        """Generate the parse tree for the given template.

        Parameters
        ----------
        template : Template
            The template to parse.

        Returns
        -------
        TemplateNode
            The root node of the parse tree.
        """
        return Lexer(template._source).parse()

    @staticmethod
    def _reconstructCode(template: Template, node: TemplateNode) -> Template:
        """Reconstruct the template from the parse tree.

        Parameters
        ----------
        template : Template
            The template to modify.
        node : TemplateNode
            The parse tree node to use.

        Returns
        -------
        Template
            The modified template.
        """
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

        # Execute the compiled code in the module's namespace
        exec(code, module.__dict__, module.__dict__)

        template._code = code
        template.module = module
        template.callable_ = template.module.render_body
        return template

    @staticmethod
    def _indexPointer(parseTree: TemplateNode, ptrName: str, index: str) -> TemplateNode:
        """Index a pointer in the parse tree.

        Parameters
        ----------
        parseTree : TemplateNode
            The parse tree to modify.
        ptrName : str
            The name of the pointer to index.
        index : str
            The index to use.

        Returns
        -------
        TemplateNode
            The modified parse tree.
        """
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
        """Index the specified variables in the given template.

        Modifies the template in place by indexing the specified variable names.

        Parameters
        ----------
        template : Template
            The template to modify.
        varNames : List[str]
            The variable names to index.
        index : str
            The index to use.
        """
        if len(varNames) == 0:
            return
        parseTree = IntrospectiveCodeTransformationMixIn._generateParseTree(template)
        for name in varNames:
            parseTree = IntrospectiveCodeTransformationMixIn._indexPointer(parseTree, name, index)
        IntrospectiveCodeTransformationMixIn._reconstructCode(template, parseTree)

    @staticmethod
    def _dereferencePointer(parseTree: TemplateNode, ptrName: str) -> TemplateNode:
        """Dereference a pointer in the parse tree.

        Parameters
        ----------
        parseTree : TemplateNode
            The parse tree to modify.
        ptrName : str
            The name of the pointer to dereference.

        Returns
        -------
        TemplateNode
            The modified parse tree with dereferenced pointers.
        """
        indexes = [i for i, node in enumerate(parseTree.nodes) if isinstance(node, Expression) and node.text == ptrName]

        for offset, idx in enumerate(indexes):
            text = Text("*", source = "*", lineno = 0, pos = 0, filename = None)
            parseTree.nodes.insert(idx + offset, text)

        return parseTree

    @staticmethod
    def dereferenceVars(template: Template, varNames: List[str]) -> None:
        """Dereference the specified variables in the given template.

        This function modifies the provided template in place by dereferencing
        the variables listed in `varNames`. The template is modified in place.

        Parameters
        ----------
        template : Template
            The template object to be modified.
        varNames : list of str
            List of variable names to dereference within the template.
        """
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
                                 includeGlobalReferences = False):
        """Extract all dynamic references from the given execution block.

        Parameters
        ----------
        ctxt : NetworkContext
            The network context.
        executionBlock : ExecutionBlock, optional
            The execution block.
        unrollStructs : bool, optional
            Whether to unroll structs.
        includeGlobalReferences : bool, optional
            Whether to include global references.

        Returns
        -------
        List[str]
            A list of dynamic references.
        """

        makoDynamicReferences = []
        for codeSnippet in executionBlock.codeSnippets:
            template, operatorRepresentation = codeSnippet.template, codeSnippet.operatorRepresentation

            newRefs = self._extractDynamicExpressions(ctxt, operatorRepresentation, template.template, unrollStructs,
                                                      includeGlobalReferences)

            makoDynamicReferences += newRefs

        ret = IntrospectiveCodeTransformationMixIn._fixCtxtOrdering(ctxt, list(dict.fromkeys(makoDynamicReferences)))

        return ret

    @staticmethod
    def _fixCtxtOrdering(ctxt: NetworkContext, nameList: List[str]) -> List[str]:
        """Fix the ordering of context names based on their appearance in the context.

        Parameters
        ----------
        ctxt : NetworkContext
            The network context.
        nameList : List[str]
            The list of context names to order.

        Returns
        -------
        List[str]
            The ordered list of context names.
        """
        orderList = [*ctxt.globalObjects.keys(), *ctxt.localObjects.keys()]
        _nameList = sorted(nameList.copy(), key = lambda key: orderList.index(key))

        return _nameList

    def _extractDynamicExpressions(self,
                                   ctxt: NetworkContext,
                                   operatorRepresentation: OperatorRepresentation,
                                   template: Template,
                                   unrollStructs = False,
                                   includeGlobalReferences = False):
        """Extract dynamic expressions from the given template.

        Parameters
        ----------
        ctxt : NetworkContext
            The network context.
        operatorRepresentation : OperatorRepresentation
            The operator representation mapping expressions to their representations.
        template : Template
            The template to extract expressions from.
        unrollStructs : bool, optional
            Whether to recursively unroll struct references. Defaults to False.
        includeGlobalReferences : bool, optional
            Whether to include global references in the result. Defaults to False.

        Returns
        -------
        List[str]
            A list of dynamic expressions, including local (and optionally global) references.
        """
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

        # Add in mako expressions that are accessed through pageargs
        # Required for unknown number of data dimensions
        for expr in makoExpressions:
            if expr.startswith("pageargs["):
                # Extract key inside pageargs[]
                key = expr[len("pageargs["):-1]
                assert key.startswith("'") or key.startswith(
                    "\""), f"pageargs key must begin with a string literal, got: {key}"

                # Extract initial string literal (between first 2 " or ' characters)
                quoteChar = key[0]
                endIdx = key.find(quoteChar, 1)
                key = key[1:endIdx]

                assert endIdx != -1, f"pageargs key missing closing quote: {expr}"

                # Search for all expressions that begin with the given key
                for exprKey in operatorRepresentation.keys():
                    if exprKey.startswith(key):
                        representedExpressions.append(operatorRepresentation[exprKey])

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

        if includeGlobalReferences:
            return dynamicLocalReferences + dynamicGlobalReferences
        else:
            return dynamicLocalReferences

# Copyright (C) 2025, ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

import copy
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import onnx_graphsurgeon as gs

from Deeploy.CommonExtensions.CodeTransformationPasses.MemoryAllocation import MemoryManagementGeneration, \
    MemoryPassthroughGeneration
from Deeploy.DeeployTypes import CodeTransformation, NetworkContext, NodeTemplate, ONNXLayer
from Deeploy.Targets.Generic.TileConstraints.UntiledTileConstraint import UntiledTileConstraint

_bypassNodeTemplate = NodeTemplate("""
// BYPASSED (Name: ${nodeName}, Op: ${nodeOp})
SINGLE_CORE ${data_out} = ${data_in};
""")


def bypassNode(ctxt: NetworkContext, layerBinding: Dict[str, ONNXLayer],
               node: gs.Node) -> Tuple[NetworkContext, Dict[str, ONNXLayer]]:

    assert len(node.inputs) == 1 and len(node.outputs) == 1, "Can only bypass nodes with single input and output!"

    # bypassedOutput = ctxt.lookup(node.outputs[0].name)
    # bypassedOutput._deploy = False

    for binding in layerBinding[node.name].mapper.bindings:
        binding.template = copy.deepcopy(_bypassNodeTemplate)
        binding.template.tileConstraint = UntiledTileConstraint()

        passes = []
        for transformationPass in binding.codeTransformer.passes:
            if isinstance(transformationPass, MemoryManagementGeneration):
                passes.append(MemoryPassthroughGeneration(transformationPass.regex))

        binding.codeTransformer = CodeTransformation(passes)

    return ctxt, layerBinding


def editAttribute(layerBinding: Dict[str, ONNXLayer], node: gs.Node, attrName: str, attrValue: Union[List[Any], Any]):
    nodeName = node.name
    operatorRepresentation = layerBinding[nodeName].mapper.parser.operatorRepresentation
    operatorRepresentation[attrName] = attrValue

    if isinstance(attrValue, list):
        node.attrs[attrName] = np.array(attrValue)
    else:
        node.attrs[attrName] = np.array([attrValue])

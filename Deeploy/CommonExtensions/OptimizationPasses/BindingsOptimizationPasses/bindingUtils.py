# ----------------------------------------------------------------------
#
# File: bindingUtils.py
#
# Last edited: 21.11.2023
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

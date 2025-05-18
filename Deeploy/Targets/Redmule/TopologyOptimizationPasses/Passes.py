# ----------------------------------------------------------------------
#
# File: RedMulePasses.py
#
# Last edited: 09.05.2025
#
# Copyright (C) 2025, ETH Zurich and University of Bologna.
#
# Author: Run Wang, ETH Zurich
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

from functools import partial
import numpy as np
import numpy.typing as npt
import onnx_graphsurgeon as gs

from Deeploy.CommonExtensions.OptimizationPasses.Matchers import Match, NonBranchingMatcher
from Deeploy.CommonExtensions.OptimizationPasses.PassClasses import ReplaceSequentialPatternPass, contextagnostic
from Deeploy.CommonExtensions.OptimizationPasses.TopologyOptimizationPasses.LoweringOptimizationPasses import (
    _permuteLastTwoDims,
    _appendTransposeNode,
)



def _redmule_weight_layout_fun(graph: gs.Graph, match: Match, name: str):
    """Convert Conv weights from [cout, h, w, cin] to [h,w,cin, cout] for RedMule accelerator"""
    node = list(match.nodes_map.values())[0]
    
    weightTensor = node.inputs[1]
    if isinstance(weightTensor, gs.Constant):
        weightTensor.values = np.transpose(weightTensor.values, (1, 2, 3, 0))
        
    return graph


@contextagnostic
class RedMuleAdjustWeightMemoryLayoutPass(ReplaceSequentialPatternPass):
    """Pass to convert Conv weights from [cout, h, w, cin] to [hwcin, cout] for RedMule accelerator"""
    
    def __init__(self, redmuleEngineName: str):
        graph = gs.Graph()
        _input = gs.Variable(name='input_1')
        output = graph.layer(inputs=[_input], outputs=['convOut'], op='Conv', name='conv')
        graph.outputs.append(output)
        graph.inputs.append(_input)
        
        super().__init__(
            graph,
            _redmule_weight_layout_fun, 
            "_REDMULE_ADJUST_WEIGHT_MEMORY_LAYOUT_PASS")


def _redmule_gemm_transpose_fun(graph: gs.Graph, match: Match, name: str):
    """
    Handle GEMM transA and transB attributes for RedMule accelerator
    
    Properly handles tensors of any dimensionality, ensuring only the last two
    dimensions are transposed when needed.
    """
    matched_nodes = [m for k, m in match.nodes_map.items()]
    gemm_node = matched_nodes[0]

    if 'transA' not in gemm_node.attrs:
        gemm_node.attrs['transA'] = 0
    if 'transB' not in gemm_node.attrs:
        gemm_node.attrs['transB'] = 0
    if 'alpha' not in gemm_node.attrs:
        gemm_node.attrs['alpha'] = 1.0
    if 'beta' not in gemm_node.attrs:
        gemm_node.attrs['beta'] = 1.0
    
    inputA = gemm_node.inputs[0]
    inputB = gemm_node.inputs[1]
    
   
    if gemm_node.attrs['transA'] != 0:
        if isinstance(inputA, gs.Constant):
            print(f"Physical transpose for constant A: {inputA.name}")
            
            if len(inputA.values.shape) > 2:
                perm = list(range(len(inputA.values.shape)))
                perm[-1], perm[-2] = perm[-2], perm[-1]
                inputA.values = np.transpose(inputA.values, perm)
            else:
                inputA.values = np.transpose(inputA.values)
                
            gemm_node.attrs['transA'] = 0
        else:
        
            perm = list(range(len(inputA.shape)))
            perm[-1], perm[-2] = perm[-2], perm[-1]
            
          
            anchorTransposeNode, anchorTransposeOutput = _appendTransposeNode(
                inputA, 
                name + "_A_Transpose",
                perm  
            )
            gemm_node.inputs[0] = anchorTransposeOutput
            gemm_node.attrs['transA'] = 0
            graph.nodes.append(anchorTransposeNode)
    

    if gemm_node.attrs['transB'] != 0:
        if isinstance(inputB, gs.Constant):
 
            if len(inputB.values.shape) > 2:
            
                perm = list(range(len(inputB.values.shape)))
                perm[-1], perm[-2] = perm[-2], perm[-1]
                
                inputB.values = np.transpose(inputB.values, perm)
            else:
                inputB.values = np.transpose(inputB.values)
                
            gemm_node.attrs['transB'] = 0
        else: 
            print(f"Adding transpose node for variable B: {inputB.name}")
            
            perm = list(range(len(inputB.shape)))
            perm[-1], perm[-2] = perm[-2], perm[-1]
            
            anchorTransposeNode, anchorTransposeOutput = _appendTransposeNode(
                inputB, 
                name + "_B_Transpose",
                perm  
            )
            gemm_node.inputs[1] = anchorTransposeOutput
            gemm_node.attrs['transB'] = 0
            graph.nodes.append(anchorTransposeNode)
    
    return graph


@contextagnostic
class RedMuleGEMMTransposePass(ReplaceSequentialPatternPass):
    """Pass to handle GEMM transA and transB attributes for RedMule accelerator"""
    
    def __init__(self, redmuleEngineName: str):
    
        pattern = gs.Graph()
        
        input_a = gs.Variable(name="input_a")
        input_b = gs.Variable(name="input_b")
        
        gemm_output = pattern.layer(
            op="Gemm", 
            name="gemm_node", 
            inputs=[input_a, input_b], 
            outputs=["gemm_output"]
        )
        
 
        pattern.inputs = [input_a, input_b]
        pattern.outputs = [gemm_output]

        super().__init__(
            pattern=pattern,
            replacement_fn=_redmule_gemm_transpose_fun,
            name="_REDMULE_GEMM_TRANSPOSE_PASS"
        )
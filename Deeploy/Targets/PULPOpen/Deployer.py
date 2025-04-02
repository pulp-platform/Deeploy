# ----------------------------------------------------------------------
#
# File: PULPDeployer.py
#
# Last edited: 08.03.2023
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

from typing import Callable, Dict, List, Type

import numpy as np
import onnx_graphsurgeon as gs

from Deeploy.AbstractDataTypes import Pointer
from Deeploy.CommonExtensions.NetworkDeployers.SignPropDeployer import SignPropDeployer
from Deeploy.CommonExtensions.OptimizationPasses.BindingsOptimizationPasses.AutoTranspose import AutoTransposeMergePass
from Deeploy.CommonExtensions.OptimizationPasses.TopologyOptimizationPasses.LoweringOptimizationPasses import \
    PULPNCHWtoNHWCPass, RemoveGlobalOutputReshapePass, TransposeMatmulInputsPass
from Deeploy.DeeployTypes import ConstantBuffer, DeploymentPlatform, NodeTemplate, TopologyOptimizer, VariableBuffer
from Deeploy.Targets.Generic.TopologyOptimizationPasses.Passes import ReshapeConstOptPass, TransposeConstOptPass, \
    TransposeMergePass, TransposeNoPermOptPass, TransposeSplitPass
from Deeploy.Targets.PULPOpen.TopologyOptimizationPasses.Passes import RQAddTransposeSquashPass

_L3AllocTemplate = NodeTemplate("""
${locPtr} = cl_ram_malloc(${size});
""")

_L3InitTemplate = NodeTemplate("""
load_file_to_ram(${locPtr}, "${extName}.hex");
""")


class PULPDeployer(SignPropDeployer):

    def __init__(self,
                 graph: gs.Graph,
                 deploymentPlatform: DeploymentPlatform,
                 inputTypes: Dict[str, Type[Pointer]],
                 loweringOptimizer: TopologyOptimizer,
                 scheduler: Callable = lambda x: x,
                 name: str = 'DeeployNetwork',
                 default_channels_first = False,
                 deeployStateDir: str = "DeeployStateDir",
                 inputOffsets = {}):
        super().__init__(graph,
                         deploymentPlatform,
                         inputTypes,
                         loweringOptimizer,
                         scheduler,
                         name,
                         default_channels_first = default_channels_first,
                         deeployStateDir = deeployStateDir,
                         inputOffsets = inputOffsets)

        self.loweringOptimizer.passes += [
            TransposeMatmulInputsPass(),
            PULPNCHWtoNHWCPass(self.default_channels_first),
            TransposeSplitPass(),
            RQAddTransposeSquashPass(),
            TransposeSplitPass(),
            TransposeMergePass(),
            TransposeConstOptPass(),
            ReshapeConstOptPass(),
            TransposeNoPermOptPass(),
            RemoveGlobalOutputReshapePass(),
        ]

        self.extNameCount = 0

    def bind(self):
        # SCHEREMO: THIS IS A STOP GAP SOLUTION. DONT REUSE. I MEAN IT. I WILL FIND YOU.
        # SCHEREMO: The BindingOptimizationPass system is fairly fragile;
        # it was designed this way because implementing further topology optimizations after
        # parsing is very involved. If there are further use-cases, we should consider making this effort,
        # but if there is only very few cases, this solution is okay.
        autoTransposePass = AutoTransposeMergePass()
        #self.ctxt, self.layerBinding = autoTransposePass.apply(self.ctxt, self.graph, self.layerBinding)
        # SCHEREMO: THIS IS A STOP GAP SOLUTION. DONT REUSE. I MEAN IT. I WILL FIND YOU.
        ret = super().bind()
        if ret:
            self.ctxt.hoistGlobalDefinition("cluster_dev", "extern struct pi_device cluster_dev;")
        return ret

    def _l3ConstBuffer(self) -> List[VariableBuffer]:
        return [
            buf for buf in self.ctxt.globalObjects.values() if all([
                isinstance(buf, VariableBuffer) and buf._deploy,
                hasattr(buf, "_users") and len(buf._users) > 0,
                hasattr(buf, "_memoryLevel") and buf._memoryLevel == "L3",
            ])
        ]

    def _newExtName(self) -> str:
        name = str(self.extNameCount)
        self.extNameCount += 1
        return name

    def _generateL3BufferAllocationCode(self, buf: VariableBuffer) -> str:
        retStr = ""
        locPtr = str(buf._instance)
        extName = self._newExtName()
        buf.extName = extName
        size = np.prod(buf.shape) * (buf._type.referencedType.typeWidth // 8)
        if isinstance(buf, ConstantBuffer):
            retStr += _L3AllocTemplate.generate({"locPtr": locPtr, "extName": extName, "size": size})
        retStr += _L3InitTemplate.generate({"locPtr": locPtr, "extName": extName, "size": size})
        return retStr

    def generateBufferAllocationCode(self) -> str:
        retStr = super().generateBufferAllocationCode()
        for buf in self._l3ConstBuffer():
            retStr += self._generateL3BufferAllocationCode(buf)
        return retStr

# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0
"""
GAP9-specific deployer that uses cl_dma.h API.

This deployer extends PULPDeployer to use GAP9-specific DMA (ClDma) via
the GAP9Bindings transformers.
"""

from typing import Callable, Dict, Type

import numpy as np
import onnx_graphsurgeon as gs

from Deeploy.AbstractDataTypes import Pointer
from Deeploy.CommonExtensions.NetworkDeployers.SignPropDeployer import SignPropDeployer
from Deeploy.DeeployTypes import ConstantBuffer, DeploymentPlatform, NodeTemplate, TopologyOptimizer, VariableBuffer
from Deeploy.Targets.GAP9.Bindings import GAP9ClusterTransformer, GAP9SimpleTransformer, GAP9Transformer
from Deeploy.Targets.PULPOpen.Deployer import PULPDeployer

# GAP9-specific L3 RAM allocation and loading templates
_GAP9L3AllocTemplate = NodeTemplate("""
${locPtr} = cl_ram_malloc(${size});
""")

_GAP9L3InitTemplate = NodeTemplate("""
load_file_to_ram(${locPtr}, "${extName}.hex");
""")


class GAP9Deployer(PULPDeployer):
    """
    GAP9-specific deployer using cl_dma.h API.

    This deployer uses GAP9-specific transformers that employ ClDma (cl_dma.h)
    instead of the low-level MCHAN API used by PULPDeployer.

    The key differences from PULPDeployer:
    - DMA: Uses ClDma (PMSIS cl_dma.h) instead of MchanDma (MCHAN hardware API)
    - L3 RAM: Uses GAP9 APS256XXN OctaSPI RAM accessed via pi_cl_ram_* APIs
    - File System: Uses ReadFS to load L3 data from flash
    """

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

        # Override transformers to use GAP9-specific ones with ClDma
        self.Transformer = GAP9Transformer
        self.ClusterTransformer = GAP9ClusterTransformer
        self.SimpleTransformer = GAP9SimpleTransformer

    def generateBufferAllocationCode(self) -> str:
        retStr = SignPropDeployer.generateBufferAllocationCode(self)

        L3FileStr = ""
        globalConstBuffers = [
            buf for key, buf in self.ctxt.globalObjects.items() if isinstance(buf, VariableBuffer) and buf._deploy
        ]
        nonArenaBuffers = [buf for buf in globalConstBuffers if buf._users != []]
        outputBuffNames = [outputBuffer.name for outputBuffer in self.graph.outputs]

        # Find all L3 constant buffers
        l3ConstBuffer = []
        for buf in nonArenaBuffers:
            if hasattr(buf, "_memoryLevel") and buf._memoryLevel == "L3" and buf.name not in outputBuffNames:
                l3ConstBuffer.append(buf)

        # Generate allocation and loading code for each L3 buffer
        for idx, buf in enumerate(l3ConstBuffer):
            locPtr = str(buf._instance)
            extName = str(idx)
            buf.extName = extName  # This enables hex dump generation
            size = np.prod(buf.shape) * (buf._type.referencedType.typeWidth // 8)

            # Allocate L3 RAM space (for constant buffers only)
            if isinstance(buf, ConstantBuffer):
                L3FileStr += _GAP9L3AllocTemplate.generate({"locPtr": locPtr, "extName": extName, "size": size})

            # Load data from ReadFS
            L3FileStr += _GAP9L3InitTemplate.generate({"locPtr": locPtr, "extName": extName, "size": size})

        retStr = retStr + L3FileStr

        return retStr

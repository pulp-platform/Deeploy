# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

"""
GAP9-specific deployer that uses cl_dma.h API.

This deployer extends PULPDeployer to use GAP9-specific DMA (ClDma) via
the GAP9Bindings transformers.
"""

from typing import Callable, Dict, List, Type

import numpy as np
import onnx_graphsurgeon as gs

from Deeploy.AbstractDataTypes import Pointer
from Deeploy.DeeployTypes import DeploymentPlatform, TopologyOptimizer
from Deeploy.Targets.PULPOpen.Deployer import PULPDeployer
from Deeploy.Targets.GAP9.Bindings import GAP9Transformer, GAP9ClusterTransformer, GAP9SimpleTransformer


class GAP9Deployer(PULPDeployer):
    """
    GAP9-specific deployer using cl_dma.h API.
    
    This deployer uses GAP9-specific transformers that employ ClDma (cl_dma.h)
    instead of the low-level MCHAN API used by PULPDeployer.
    
    The key difference is in the DMA implementation:
    - PULP: Uses MchanDma (low-level MCHAN hardware API)
    - GAP9: Uses ClDma (PMSIS cl_dma.h high-level API)
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

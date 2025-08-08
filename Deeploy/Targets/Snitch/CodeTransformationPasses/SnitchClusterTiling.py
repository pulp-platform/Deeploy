# ----------------------------------------------------------------------
#
# File: SnitchClusterTiling.py
#
# Last edited: 31.05.2024
#
# Copyright (C) 2024, ETH Zurich and University of Bologna.
#
# Author:
# - Victor Jung, jungvi@iis.ee.ethz.ch, ETH Zurich
#
# ----------------------------------------------------------------------
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the License); you may
# not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an AS IS BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Tuple

from Deeploy.DeeployTypes import CodeGenVerbosity, CodeTransformationPass, ExecutionBlock, NetworkContext, _NoVerbosity
from Deeploy.TilingExtension.AsyncDma import AsyncDma
from Deeploy.TilingExtension.CodeTransformationPasses.DoubleBufferingTilingCodeGeneration import \
    DoubleBufferingTilingCodeGeneration
from Deeploy.TilingExtension.CodeTransformationPasses.SingleBufferingTilingCodeGeneration import \
    SingleBufferingTilingCodeGeneration
from Deeploy.TilingExtension.CodeTransformationPasses.TilingPrototypes import DoubleBufferingTilingMixIn, \
    SingleBufferingTilingMixIn


class SnitchClusterTilingSB(SingleBufferingTilingCodeGeneration, SingleBufferingTilingMixIn):
    pass


class SnitchClusterTilingDB(DoubleBufferingTilingCodeGeneration, DoubleBufferingTilingMixIn):
    pass


class SnitchClusterTiling(CodeTransformationPass):

    def __init__(self, externalMemory: str, localMemory: str, dma: AsyncDma):
        self.SB = SnitchClusterTilingSB(externalMemory, localMemory, dma)
        self.DB = SnitchClusterTilingDB(externalMemory, localMemory, dma)

    def apply(self,
              ctxt: NetworkContext,
              executionBlock: ExecutionBlock,
              name: str,
              verbose: CodeGenVerbosity = _NoVerbosity) -> Tuple[NetworkContext, ExecutionBlock]:
        if verbose.tilingProfiling:
            raise NotImplementedError("Profiling not implemented for L2")

        ctxt, executionBlock = self.SB.apply(ctxt, executionBlock, name)
        ctxt, executionBlock = self.DB.apply(ctxt, executionBlock, name)
        return ctxt, executionBlock

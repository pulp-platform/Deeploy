# Copyright (C) 2025, ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

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

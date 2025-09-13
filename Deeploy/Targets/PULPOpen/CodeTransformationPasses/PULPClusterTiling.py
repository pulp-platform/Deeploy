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
    ProfilingDoubleBufferingTilingMixIn, ProfilingSingleBufferingTilingMixIn, SingleBufferingTilingMixIn


class PULPClusterTilingGenerationSB(SingleBufferingTilingCodeGeneration, SingleBufferingTilingMixIn):
    pass


class ProfilingPULPClusterTilingGenerationSB(SingleBufferingTilingCodeGeneration, ProfilingSingleBufferingTilingMixIn):
    pass


class PULPClusterTilingGenerationDB(DoubleBufferingTilingCodeGeneration, DoubleBufferingTilingMixIn):
    pass


class ProfilingPULPClusterTilingGenerationDB(DoubleBufferingTilingCodeGeneration, ProfilingDoubleBufferingTilingMixIn):
    pass


class PULPClusterTiling(CodeTransformationPass):

    def __init__(self, externalMemory: str, localMemory: str, dma: AsyncDma):
        self.SB = PULPClusterTilingGenerationSB(externalMemory, localMemory, dma)
        self.profilingSB = ProfilingPULPClusterTilingGenerationSB(externalMemory, localMemory, dma)
        self.DB = PULPClusterTilingGenerationDB(externalMemory, localMemory, dma)
        self.profilingDB = ProfilingPULPClusterTilingGenerationDB(externalMemory, localMemory, dma)

    def apply(self,
              ctxt: NetworkContext,
              executionBlock: ExecutionBlock,
              name: str,
              verbose: CodeGenVerbosity = _NoVerbosity) -> Tuple[NetworkContext, ExecutionBlock]:

        if verbose.tilingProfiling:
            ctxt, executionBlock = self.profilingSB.apply(ctxt, executionBlock, name)
            ctxt, executionBlock = self.profilingDB.apply(ctxt, executionBlock, name)
        else:
            ctxt, executionBlock = self.SB.apply(ctxt, executionBlock, name)
            ctxt, executionBlock = self.DB.apply(ctxt, executionBlock, name)

        return ctxt, executionBlock

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


class PULPL3TilingGenerationSB(SingleBufferingTilingCodeGeneration, SingleBufferingTilingMixIn):
    pass


class ProfilingPULPL3TilingGenerationSB(SingleBufferingTilingCodeGeneration, ProfilingSingleBufferingTilingMixIn):
    pass


class PULPL3TilingGenerationDB(DoubleBufferingTilingCodeGeneration, DoubleBufferingTilingMixIn):
    pass


class ProfilingPULPL3TilingGenerationDB(DoubleBufferingTilingCodeGeneration, ProfilingDoubleBufferingTilingMixIn):
    pass


class PULPL3Tiling(CodeTransformationPass):

    def __init__(self, externalMemory: str, localMemory: str, dma: AsyncDma):
        self.SB = PULPL3TilingGenerationSB(externalMemory, localMemory, dma)
        self.DB = PULPL3TilingGenerationDB(externalMemory, localMemory, dma)
        self.profilingSB = ProfilingPULPL3TilingGenerationSB(externalMemory, localMemory, dma)
        self.profilingDB = ProfilingPULPL3TilingGenerationDB(externalMemory, localMemory, dma)

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

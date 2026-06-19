# SPDX-FileCopyrightText: 2024 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from typing import Tuple

from Deeploy.DeeployTypes import CodeGenVerbosity, CodeTransformationPass, ExecutionBlock, NetworkContext, _NoVerbosity
from Deeploy.TilingExtension.AsyncDma import AsyncDma
from Deeploy.TilingExtension.CodeTransformationPasses.DoubleBufferingTilingCodeGeneration import \
    DoubleBufferingTilingCodeGeneration, PerfCounterDoubleBufferingTilingMixIn, ProfilingDoubleBufferingTilingMixIn
from Deeploy.TilingExtension.CodeTransformationPasses.SingleBufferingTilingCodeGeneration import \
    PerfCounterSingleBufferingTilingMixIn, ProfilingSingleBufferingTilingMixIn, SingleBufferingTilingCodeGeneration


class PULPClusterTilingGenerationSB(SingleBufferingTilingCodeGeneration):
    pass


class ProfilingPULPClusterTilingGenerationSB(SingleBufferingTilingCodeGeneration, ProfilingSingleBufferingTilingMixIn):
    pass


class PULPClusterTilingGenerationDB(DoubleBufferingTilingCodeGeneration):
    pass


class ProfilingPULPClusterTilingGenerationDB(DoubleBufferingTilingCodeGeneration, ProfilingDoubleBufferingTilingMixIn):
    pass


class PerfCounterPULPClusterTilingGenerationSB(SingleBufferingTilingCodeGeneration, PerfCounterSingleBufferingTilingMixIn):
    """Single buffering with performance counter profiling"""
    pass


class PerfCounterPULPClusterTilingGenerationDB(DoubleBufferingTilingCodeGeneration, PerfCounterDoubleBufferingTilingMixIn):
    """Double buffering with performance counter profiling"""
    pass


class CombinedProfilingPULPClusterTilingGenerationSB(SingleBufferingTilingCodeGeneration, ProfilingSingleBufferingTilingMixIn, PerfCounterSingleBufferingTilingMixIn):
    """Single buffering with both cycle profiling and performance counter profiling"""
    pass


class CombinedProfilingPULPClusterTilingGenerationDB(DoubleBufferingTilingCodeGeneration, ProfilingDoubleBufferingTilingMixIn, PerfCounterDoubleBufferingTilingMixIn):
    """Double buffering with both cycle profiling and performance counter profiling"""
    pass


class PULPClusterTiling(CodeTransformationPass):

    def __init__(self, externalMemory: str, localMemory: str, dma: AsyncDma, usePerfCounters: bool = False):
        self.usePerfCounters = usePerfCounters
        self.SB = PULPClusterTilingGenerationSB(externalMemory, localMemory, dma)
        self.profilingSB = ProfilingPULPClusterTilingGenerationSB(externalMemory, localMemory, dma)
        self.perfCounterSB = PerfCounterPULPClusterTilingGenerationSB(externalMemory, localMemory, dma)
        self.combinedProfilingSB = CombinedProfilingPULPClusterTilingGenerationSB(externalMemory, localMemory, dma)
        self.DB = PULPClusterTilingGenerationDB(externalMemory, localMemory, dma)
        self.profilingDB = ProfilingPULPClusterTilingGenerationDB(externalMemory, localMemory, dma)
        self.perfCounterDB = PerfCounterPULPClusterTilingGenerationDB(externalMemory, localMemory, dma)
        self.combinedProfilingDB = CombinedProfilingPULPClusterTilingGenerationDB(externalMemory, localMemory, dma)

    def apply(self,
              ctxt: NetworkContext,
              executionBlock: ExecutionBlock,
              name: str,
              verbose: CodeGenVerbosity = _NoVerbosity) -> Tuple[NetworkContext, ExecutionBlock]:

        if self.usePerfCounters and verbose.tilingProfiling:
            # Use combined profiling: cycle measurements + performance counter stats
            ctxt, executionBlock = self.combinedProfilingSB.apply(ctxt, executionBlock, name)
            ctxt, executionBlock = self.combinedProfilingDB.apply(ctxt, executionBlock, name)
        elif verbose.tilingProfiling:
            # Use cycle profiling only (basic cycle measurements)
            ctxt, executionBlock = self.profilingSB.apply(ctxt, executionBlock, name)
            ctxt, executionBlock = self.profilingDB.apply(ctxt, executionBlock, name)
        else:
            # No profiling
            ctxt, executionBlock = self.SB.apply(ctxt, executionBlock, name)
            ctxt, executionBlock = self.DB.apply(ctxt, executionBlock, name)

        return ctxt, executionBlock

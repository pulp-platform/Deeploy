# SPDX-FileCopyrightText: 2024 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from typing import Tuple

from Deeploy.DeeployTypes import CodeGenVerbosity, CodeTransformationPass, ExecutionBlock, NetworkContext, \
    NodeTemplate, _NoVerbosity
from Deeploy.TilingExtension.AsyncDma import AsyncDma
from Deeploy.TilingExtension.CodeTransformationPasses.DoubleBufferingTilingCodeGeneration import \
    DoubleBufferingTilingCodeGeneration, ProfilingDoubleBufferingTilingMixIn
from Deeploy.TilingExtension.CodeTransformationPasses.SingleBufferingTilingCodeGeneration import \
    ProfilingSingleBufferingTilingMixIn, SingleBufferingTilingCodeGeneration


class SnitchClusterTilingSB(SingleBufferingTilingCodeGeneration):
    pass


class SnitchClusterTilingDB(DoubleBufferingTilingCodeGeneration):
    pass


class ProfilingSnitchClusterTilingSB(SingleBufferingTilingCodeGeneration, ProfilingSingleBufferingTilingMixIn):
    _printCycleDifference = NodeTemplate(r"""
    printf("%s%u][Core %d] %s%6u%s", ${prefixStr}, ${profileIdxVar}, snrt_global_core_idx(), "${flavorStr}", \
    ${measurement}, ${suffixStr});
    """)

    _printCycleContribution = NodeTemplate(r"""
    uint32_t total = ${measurementInput} + ${measurementKernel} + ${measurementOutput};
    uint32_t dma = ${measurementInput} + ${measurementOutput};
    float overhead_percentage = (total == 0) ? 0 : dma * 100.0f / total;
    float kernel_percentage = (total == 0) ? 0 : ${measurementKernel} * 100.0f / total;
    printf("%s%u][Core %d] Total      :%6u cycles (%2.1f%% Kernel + %2.1f%% Overhead, %u + %u)\n", ${prefixStr}, ${profileIdxVar}, snrt_global_core_idx(), total, kernel_percentage, overhead_percentage, ${measurementKernel}, dma);
    """)


class ProfilingSnitchClusterTilingDB(DoubleBufferingTilingCodeGeneration, ProfilingDoubleBufferingTilingMixIn):
    _printCycleDifference = NodeTemplate(r"""
    printf("%s%u][Core %d] %s%6u%s", ${prefixStr}, ${profileIdxVar}, snrt_global_core_idx(), "${flavorStr}", \
    ${measurement}, ${suffixStr});
    """)

    _printCycleContribution = NodeTemplate(r"""
    uint32_t total = ${measurementInput} + ${measurementKernel} + ${measurementOutput};
    uint32_t dma = ${measurementInput} + ${measurementOutput};
    float overhead_percentage = (total == 0) ? 0 : dma * 100.0f / total;
    float kernel_percentage = (total == 0) ? 0 : ${measurementKernel} * 100.0f / total;
    printf("%s%u][Core %d] Total      :%6u cycles (%2.1f%% Kernel + %2.1f%% Overhead, %u + %u)\n", ${prefixStr}, ${profileIdxVar}, snrt_global_core_idx(), total, kernel_percentage, overhead_percentage, ${measurementKernel}, dma);
    """)


class SnitchClusterTiling(CodeTransformationPass):

    def __init__(self, externalMemory: str, localMemory: str, dma: AsyncDma):
        self.SB = SnitchClusterTilingSB(externalMemory, localMemory, dma)
        self.profilingSB = ProfilingSnitchClusterTilingSB(externalMemory, localMemory, dma)

        self.DB = SnitchClusterTilingDB(externalMemory, localMemory, dma)
        self.profilingDB = ProfilingSnitchClusterTilingDB(externalMemory, localMemory, dma)

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

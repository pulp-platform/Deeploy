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
    printf("%s%u][Core %d] %s%u%s", ${prefixStr}, ${profileIdxVar}, snrt_global_core_idx(), "${flavorStr}", \
    ${measurementsEnd}[${profileIdxVar}] - ${measurementsStart}[${profileIdxVar}], ${suffixStr});
    """)


class ProfilingSnitchClusterTilingDB(DoubleBufferingTilingCodeGeneration, ProfilingDoubleBufferingTilingMixIn):
    _printCycleDifference = NodeTemplate(r"""
    printf("%s%u][Core %d] %s%u%s", ${prefixStr}, ${profileIdxVar}, snrt_global_core_idx(), "${flavorStr}", \
    ${measurementsEnd}[${profileIdxVar}] - ${measurementsStart}[${profileIdxVar}], ${suffixStr});
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

# Copyright (C) 2025, ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

from typing import Tuple

from Deeploy.CommonExtensions.CodeTransformationPasses.CycleMeasurement import ProfilingCodeGeneration
from Deeploy.DeeployTypes import CodeGenVerbosity, CodeTransformationPass, ExecutionBlock, NetworkContext, _NoVerbosity


class PULPProfileUntiled(CodeTransformationPass):

    def __init__(self):
        self.profileUntiled = ProfilingCodeGeneration()

    def apply(self,
              ctxt: NetworkContext,
              executionBlock: ExecutionBlock,
              name: str,
              verbose: CodeGenVerbosity = _NoVerbosity) -> Tuple[NetworkContext, ExecutionBlock]:

        if verbose.untiledProfiling:
            ctxt, executionBlock = self.profileUntiled.apply(ctxt, executionBlock, name)

        return ctxt, executionBlock

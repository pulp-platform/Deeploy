# ----------------------------------------------------------------------
#
# File: PULPClusterTiling.py
#
# Last edited: 19.04.2024
#
# Copyright (C) 2024, ETH Zurich and University of Bologna.
#
# Author: Moritz Scherer, ETH Zurich
#
# ----------------------------------------------------------------------
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the License); you may
# not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an AS IS BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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

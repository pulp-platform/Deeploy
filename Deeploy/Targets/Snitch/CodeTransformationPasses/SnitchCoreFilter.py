# ----------------------------------------------------------------------
#
# File: SnitchCoreFilter.py
#
# Last edited: 04.06.2024
#
# Copyright (C) 2024, ETH Zurich and University of Bologna.
#
# Author:
# - Luka Macan, luka.macan@unibo.it, University of Bologna
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

from typing import Literal, Tuple

from Deeploy.DeeployTypes import CodeGenVerbosity, CodeTransformationPass, ExecutionBlock, NetworkContext, \
    NodeTemplate, _NoVerbosity


class SnitchCoreFilterPass(CodeTransformationPass):

    def __init__(self, coreType: Literal["dm", "compute"]):
        super().__init__()
        self.coreType = coreType

    def apply(self,
              ctxt: NetworkContext,
              executionBlock: ExecutionBlock,
              name: str,
              verbose: CodeGenVerbosity = _NoVerbosity) -> Tuple[NetworkContext, ExecutionBlock]:
        executionBlock.addLeft(NodeTemplate(f"if (snrt_is_{self.coreType}_core()) {{\n"), {})
        executionBlock.addRight(NodeTemplate("}\n"), {})
        return ctxt, executionBlock

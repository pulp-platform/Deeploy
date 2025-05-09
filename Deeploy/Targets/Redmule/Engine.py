# ----------------------------------------------------------------------
#
# File: Engine.py
#
# Last edited: 26.07.2024
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

from typing import List

import onnx_graphsurgeon as gs

from Deeploy.DeeployTypes import DeploymentEngine, NodeMapper
from Deeploy.Targets.Generic.Layers import MatMulLayer, ConvLayer
from Deeploy.Targets.Generic.Parsers import MatMulParser, ConvParser
from Deeploy.Targets.Redmule.Tiler import RedmuleMatMulTilingReadyBindings, RedmuleConvTilingReadyBindings
from Deeploy.Targets.PULPOpen.Parsers import PULPFPConv2DParser

MatMulRedmuleMapper = NodeMapper(
    MatMulParser(), RedmuleMatMulTilingReadyBindings)
Conv2DRedmuleMapper = NodeMapper(
    PULPFPConv2DParser(), RedmuleConvTilingReadyBindings)

RedmuleMapping = {
    'MatMul': MatMulLayer([MatMulRedmuleMapper]),
    'Conv': ConvLayer([Conv2DRedmuleMapper])
}

_includeList = []

_redmuleInitCode = r"""
// Redmule engine initialization
"""


class RedmuleEngine(DeploymentEngine):

    def __init__(self,
                 name: str,
                 Mapping = RedmuleMapping,
                 initCode: str = _redmuleInitCode,
                 includeList: List[str] = _includeList) -> None:
        super().__init__(name, Mapping, initCode, includeList)



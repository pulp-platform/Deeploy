# ----------------------------------------------------------------------
#
# File: NetworkDeployerWrapper.py
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

from typing import Any, Tuple, Union

import onnx_graphsurgeon as gs

from Deeploy.DeeployTypes import CodeGenVerbosity, NetworkContext, NetworkDeployer, ONNXLayer, _NoVerbosity


class NetworkDeployerWrapper(NetworkDeployer):

    def __init__(self, deployer: NetworkDeployer):
        self.__dict__["_innerObject"] = deployer

    def __getattr__(self, name):
        return getattr(self._innerObject, name)

    def __setattr__(self, name, value):
        if hasattr(self._innerObject, name):
            setattr(self._innerObject, name, value)
        else:
            super().__setattr__(name, value)

    """ Class attributes
    Class attributes don't get caught by __getattr__ method so we have to explicitly override them
    """

    @property
    def parsed(self):
        return self._innerObject.parsed

    @property
    def bound(self):
        return self._innerObject.bound

    @property
    def transformed(self):
        return self._innerObject.transformed

    @property
    def prepared(self):
        return self._innerObject.prepared

    """ Extension augmented methods
    Extensions augment methods and to preserve these augmentations, we have to call the innerObjects method instead of just using the inherited one.
    """

    # SignPropDeployer augment
    def _createIOBindings(self, ctxt: NetworkContext, graph: gs.Graph):
        return self._innerObject._createIOBindings(ctxt, graph)

    # MemoryAwareDeployer, TilerAwareDeployer, and PULPDeployer augments
    def bind(self) -> bool:
        return self._innerObject.bind()

    # MemoryAwareDeployer augment
    def lower(self, graph: gs.Graph) -> gs.Graph:
        return self._innerObject.lower(graph)

    # MemoryAwareDeployer augment
    def codeTransform(self, verbose: CodeGenVerbosity = _NoVerbosity):
        return self._innerObject.codeTransform(verbose)

    # MemoryAwareDeployer augment
    def _parseNode(self, node: ONNXLayer, ctxt: NetworkContext,
                   default_channels_first: bool) -> Tuple[NetworkContext, bool]:
        return self._innerObject._parseNode(node, ctxt, default_channels_first)

    # PULPDeployer augment
    def generateBufferAllocationCode(self) -> str:
        return self._innerObject.generateBufferAllocationCode()

    # MultiEngineDeployer augment
    def _mapNode(self, node: gs.Node) -> Union[ONNXLayer, Any]:
        return self._innerObject._mapNode(node)

# SPDX-FileCopyrightText: 2024 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

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

    def _printMemorySummary(self):
        return self._innerObject._printMemorySummary()

    def _printInputOutputSummary(self):
        return self._innerObject._printInputOutputSummary()

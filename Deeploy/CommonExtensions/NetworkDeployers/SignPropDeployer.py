# SPDX-FileCopyrightText: 2023 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from typing import Callable, Dict, Type

import onnx_graphsurgeon as gs

from Deeploy.AbstractDataTypes import Pointer
from Deeploy.DeeployTypes import CodeGenVerbosity, DeploymentPlatform, NetworkDeployer, TopologyOptimizer, _NoVerbosity
from Deeploy.Logging import DEFAULT_LOGGER as log


class SignPropDeployer(NetworkDeployer):

    def __init__(self,
                 graph: gs.Graph,
                 deploymentPlatform: DeploymentPlatform,
                 inputTypes: Dict[str, Type[Pointer]],
                 loweringOptimizer: TopologyOptimizer,
                 scheduler: Callable = lambda x: x,
                 name: str = 'DeeployNetwork',
                 default_channels_first: bool = True,
                 deeployStateDir: str = "DeeployState",
                 inputOffsets: Dict[str, int] = {}):
        super().__init__(graph, deploymentPlatform, inputTypes, loweringOptimizer, scheduler, name,
                         default_channels_first, deeployStateDir)

        if inputOffsets == {}:
            for key in inputTypes.keys():
                inputOffsets[key] = 0

        self.inputOffsets = inputOffsets

    def _createIOBindings(self, ctxt, graph):
        ctxt = super()._createIOBindings(ctxt, graph)
        for node in graph.inputs:
            data_name = node.name
            nb = ctxt.lookup(data_name)
            data_type = self.inputTypes[data_name]
            nb._signed = (self.inputOffsets[data_name] == 0)
            nb.nLevels = (2**data_type.referencedType.typeWidth)

        return ctxt

    def generateFunction(self, verbose: CodeGenVerbosity = _NoVerbosity) -> str:
        """Helper function to prepare deployment and return generated function code

        """

        if not self.prepared:
            self.prepare(verbose = verbose)

        log.info("=" * 80)
        log.info("Deeploy Code Generation")
        log.info("=" * 80)

        log.info('Input:')
        for name in self.inputTypes.keys():
            buf = self.ctxt.lookup(name)
            log.info(
                f" - '{name}': Type: {buf._type.referencedType.typeName}, nLevels: {buf.nLevels}, Signed: {buf._signed}, Offset: {self.inputOffsets[name]}"
            )

        log.info('Output:')
        for buf in self.outputs():
            log.info(
                f" - '{buf.name}': Type: {buf._type.referencedType.typeName}, nLevels: {buf.nLevels}, Signed: {buf._signed}"
            )

        log.info("-" * 80)
        num_ops = self.numberOfOps(verbose = True)
        log.info("-" * 80)
        log.info(f"Number of Ops.                : {num_ops}")
        log.info(f"Model Parameters              : {self.getParameterSize()}")

        return self.generateInferenceCode()

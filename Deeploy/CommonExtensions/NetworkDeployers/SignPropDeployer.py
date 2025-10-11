# SPDX-FileCopyrightText: 2023 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from typing import Callable, Dict, Type

import onnx_graphsurgeon as gs

from Deeploy.AbstractDataTypes import Pointer
from Deeploy.DeeployTypes import DeploymentPlatform, NetworkDeployer, TopologyOptimizer
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
                 inputOffsets: Dict[str, int] = {},
                 n_cores: int = 8):
        super().__init__(
            graph=graph,
            deploymentPlatform=deploymentPlatform,
            inputTypes=inputTypes,
            loweringOptimizer=loweringOptimizer,
            scheduler=scheduler,
            name=name,
            default_channels_first=default_channels_first,
            deeployStateDir=deeployStateDir,
            n_cores=n_cores,
            )

        if inputOffsets == {}:
            for key in inputTypes.keys():
                inputOffsets[key] = 0

        self.inputOffsets = inputOffsets
        self.n_cores = n_cores

    def _createIOBindings(self, ctxt, graph):
        ctxt = super()._createIOBindings(ctxt, graph)
        for node in graph.inputs:
            data_name = node.name
            nb = ctxt.lookup(data_name)
            data_type = self.inputTypes[data_name]
            nb._signed = (self.inputOffsets[data_name] == 0)
            nb.nLevels = (2**data_type.referencedType.typeWidth)

        return ctxt

    def _printInputOutputSummary(self):
        log.info('Input:')
        for buf in self.inputs():
            log.info(
                f" - '{buf.name}': Type: {buf._type.referencedType.typeName}, nLevels: {buf.nLevels}, Signed: {buf._signed}, Offset: {self.inputOffsets[buf.name]}"
            )

        log.info('Output:')
        for buf in self.outputs():
            log.info(
                f" - '{buf.name}': Type: {buf._type.referencedType.typeName}, nLevels: {buf.nLevels}, Signed: {buf._signed}"
            )

# SPDX-FileCopyrightText: 2023 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from typing import Callable, Dict, Type

import onnx_graphsurgeon as gs

from Deeploy.AbstractDataTypes import IntegerImmediate, Pointer
from Deeploy.CommonExtensions.TypeCheckers.SignPropTypeChecker import SignPropTypeChecker
from Deeploy.DeeployTypes import ConstantBuffer, DeploymentPlatform, NetworkDeployer, OperatorDescriptor, \
    TopologyOptimizer, VariableBuffer
from Deeploy.Logging import DEFAULT_LOGGER as log


class SignPropDeployer(NetworkDeployer):

    def __init__(self,
                 graph: gs.Graph,
                 deploymentPlatform: DeploymentPlatform,
                 inputTypes: Dict[str, Type[Pointer]],
                 loweringOptimizer: TopologyOptimizer,
                 operatorDescriptors: Dict[str, OperatorDescriptor],
                 scheduler: Callable = lambda x: x,
                 name: str = 'DeeployNetwork',
                 default_channels_first: bool = True,
                 deeployStateDir: str = "DeeployState",
                 inputOffsets: Dict[str, int] = {}):
        super().__init__(graph, deploymentPlatform, inputTypes, loweringOptimizer, operatorDescriptors, scheduler, name,
                         default_channels_first, deeployStateDir)

        if inputOffsets == {}:
            for key in inputTypes.keys():
                inputOffsets[key] = 0

        self.inputOffsets = inputOffsets

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

    def parse(self, default_channels_first: bool = True) -> bool:
        parsable = super().parse(default_channels_first)
        if not parsable:
            return False

        # Annotate global buffers
        for obj in self.ctxt.globalObjects.values():
            assert isinstance(obj, VariableBuffer)
            refTy = obj._type.referencedType
            if isinstance(obj, ConstantBuffer):
                assert refTy.checkPromotion(obj.values), f"Can't cast {obj} to {refTy}"
                if issubclass(refTy, IntegerImmediate):
                    obj.nLevels = obj.values.max() - obj.values.min()
                    obj._signed = refTy.typeMin < 0
            elif obj.name in self.inputOffsets:
                obj._signed = (self.inputOffsets[obj.name] == 0)
                obj.nLevels = (2**refTy.typeWidth)

        # Annotate rest
        for layer in self.layerBinding.values():
            node = layer.node
            opRepr = layer.mapper.parser.operatorRepresentation
            typeChecker = layer.mapper.binder.typeChecker
            outTy = self.ctxt.lookup(node.outputs[0].name)._type.referencedType
            if issubclass(outTy, IntegerImmediate) and isinstance(typeChecker, SignPropTypeChecker):
                inputs = [self.ctxt.lookup(t.name) for t in node.inputs]
                outputNLevels = typeChecker._inferNumLevels(inputs, opRepr)
                outputSigned = typeChecker._inferSignedness(inputs, opRepr)

                outputs = [self.ctxt.lookup(t.name) for t in node.outputs]
                for buffer, nLevels, signed in zip(outputs, outputNLevels, outputSigned):
                    buffer.nLevels = nLevels
                    buffer._signed = signed

        return True

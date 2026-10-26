# SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0
from typing import Dict

from Deeploy.DeeployTypes import ConstantBuffer, DeploymentEngine, DeploymentPlatform, NodeMapper, TopologyOptimizer, \
    VariableBuffer
from Deeploy.Targets.Carfield.Templates import AllocateTemplate, FreeTemplate
from Deeploy.Targets.Generic.Layers import AddLayer, ConcatLayer, ConvLayer, MaxPoolLayer, PadLayer, ReluLayer, \
    SingleOperationPerElementLayer
from Deeploy.Targets.Generic.Parsers import AddParser, GenericConv2DParser, GenericMaxPool2DParser, Pad2DParser, \
    ReluParser
from Deeploy.Targets.Generic.Platform import GenericOptimizer, GenericStructBuffer, GenericTransientBuffer

from .Bindings import CarfieldAddBindings, CarfieldArgMaxChannelBindings, CarfieldCastInt32Bindings, \
    CarfieldConcat2Bindings, CarfieldConv2DBindings, CarfieldDivBindings, CarfieldMaxPool2DBindings, \
    CarfieldPad2DBindings, CarfieldReluBindings, CarfieldResizeNearest2DBindings, CarfieldSubBindings
from .Layers import CarfieldChannelBroadcastLayer
from .Parsers import CarfieldArgMaxChannelParser, CarfieldCastInt32Parser, CarfieldChannelBroadcastParser, \
    CarfieldConcat2Parser, CarfieldResizeNearestParser
from .TopologyOptimizationPasses import CarfieldResizeNearestPass

AddMapper = NodeMapper(AddParser(), CarfieldAddBindings)
PadMapper = NodeMapper(Pad2DParser(), CarfieldPad2DBindings)
ConvMapper = NodeMapper(GenericConv2DParser(), CarfieldConv2DBindings)
ReluMapper = NodeMapper(ReluParser(), CarfieldReluBindings)
SubMapper = NodeMapper(CarfieldChannelBroadcastParser(), CarfieldSubBindings)
DivMapper = NodeMapper(CarfieldChannelBroadcastParser(), CarfieldDivBindings)
MaxPoolMapper = NodeMapper(GenericMaxPool2DParser(), CarfieldMaxPool2DBindings)
ResizeMapper = NodeMapper(CarfieldResizeNearestParser(), CarfieldResizeNearest2DBindings)
ConcatMapper = NodeMapper(CarfieldConcat2Parser(), CarfieldConcat2Bindings)
ArgMaxMapper = NodeMapper(CarfieldArgMaxChannelParser(), CarfieldArgMaxChannelBindings)
CastMapper = NodeMapper(CarfieldCastInt32Parser(), CarfieldCastInt32Bindings)

CarfieldMapping = {
    'Add': AddLayer([AddMapper]),
    'Pad': PadLayer([PadMapper]),
    'Conv': ConvLayer([ConvMapper]),
    'Relu': ReluLayer([ReluMapper]),
    'Sub': CarfieldChannelBroadcastLayer([SubMapper]),
    'Div': CarfieldChannelBroadcastLayer([DivMapper]),
    'MaxPool': MaxPoolLayer([MaxPoolMapper]),
    'Resize': SingleOperationPerElementLayer([ResizeMapper]),
    'Concat': ConcatLayer([ConcatMapper]),
    'ArgMax': SingleOperationPerElementLayer([ArgMaxMapper]),
    'Cast': SingleOperationPerElementLayer([CastMapper]),
}

CarfieldOptimizer = TopologyOptimizer([CarfieldResizeNearestPass()] + list(GenericOptimizer.passes),
                                      name = "CarfieldOptimizer")

includeList = [
    "Add.h", "Conv.h", "Conv2D_tiled.h", "Utils.h", "pulp.h", "string.h", "Relu.h", "Sub.h", "Div.h", "MaxPool.h",
    "Resize.h", "Concat.h", "ArgMax.h", "Cast.h"
]


class CarfieldEngine(DeploymentEngine):

    def __init__(self, name = "CarfieldEngine", Mapping = CarfieldMapping, initCode = "", includeList = includeList):
        super().__init__(name, Mapping, initCode, includeList)


class CarfieldConstantBuffer(ConstantBuffer):
    """Constant tensor living at a fixed DDR address.

    The data is not embedded in the binary: it is restored into DDR once per
    session, and the generated C only declares a pointer to it. Allocation and
    deallocation therefore emit nothing -- the pointer is initialised by the
    linker, so no core needs to run any code for it and no `core_id` guard is
    required.
    """

    initTemplate = AllocateTemplate.CarfieldGlobalInitTemplate
    allocTemplate = AllocateTemplate.CarfieldGlobalAllocTemplate
    freeTemplate = FreeTemplate.CarfieldGlobalFreeTemplate

    def _bufferRepresentation(self) -> Dict:

        if not hasattr(self, "address"):
            raise RuntimeError(f"ConstantBuffer '{self.name}' has no DDR address. Addresses are assigned by "
                               "CarfieldDeployer.bind(); this buffer was most likely generated through a "
                               "deployer that is not a CarfieldDeployer.")

        return {
            "type": self._type,
            "name": self.name,
            "size": self.values.size,
            "address": self.address,
        }


class CarfieldVariableBuffer(VariableBuffer):

    initTemplate = AllocateTemplate.CarfieldVariableInitTemplate
    allocTemplate = AllocateTemplate.CarfieldVariableAllocateTemplate
    deallocTemplate = FreeTemplate.CarfieldVariableFreeTemplate


class CarfieldPlatform(DeploymentPlatform):

    def __init__(self,
                 engines = [CarfieldEngine()],
                 variableBuffer = CarfieldVariableBuffer,
                 constantBuffer = CarfieldConstantBuffer,
                 structBuffer = GenericStructBuffer,
                 transientBuffer = GenericTransientBuffer):
        super().__init__(engines, variableBuffer, constantBuffer, structBuffer, transientBuffer)

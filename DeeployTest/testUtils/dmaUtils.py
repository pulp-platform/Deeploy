# ----------------------------------------------------------------------
#
# File: dmaUtils.py
#
# Last edited: 11.09.2025
#
# Copyright (C) 2025, ETH Zurich and University of Bologna.
#
# Author:
# - Luka Macan, University of Bologna
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

import math
from typing import Dict, List, Optional, Tuple, Type

import numpy.typing as npt
import onnx_graphsurgeon as gs

from Deeploy.AbstractDataTypes import BaseType, Pointer, PointerClass
from Deeploy.CommonExtensions.DataTypes import minimalIntegerType
from Deeploy.DeeployTypes import NetworkContext, NetworkDeployer, NodeParser, NodeTemplate, NodeTypeChecker, \
    ONNXLayer, OperatorRepresentation, VariableBuffer
from Deeploy.MemoryLevelExtension.MemoryLevels import MemoryHierarchy, MemoryLevel
from Deeploy.MemoryLevelExtension.NetworkDeployers.MemoryLevelDeployer import MemoryDeployerWrapper, \
    MemoryPlatformWrapper
from Deeploy.MemoryLevelExtension.OptimizationPasses.MemoryLevelAnnotationPasses import AnnotateDefaultMemoryLevel, \
    AnnotateIOMemoryLevel
from Deeploy.Targets.PULPOpen.Deployer import PULPDeployer
from Deeploy.Targets.PULPOpen.Platform import MemoryPULPPlatform, PULPOptimizer
from Deeploy.Targets.Snitch.Deployer import SnitchDeployer
from Deeploy.Targets.Snitch.Platform import SnitchOptimizer, SnitchPlatform
from Deeploy.TilingExtension.MemoryConstraints import MemoryConstraint, NodeMemoryConstraint, \
    PatternMemoryConstraints, TensorMemoryConstraint
from Deeploy.TilingExtension.MemoryScheduler import MemoryBlock
from Deeploy.TilingExtension.TileConstraint import TileConstraint
from Deeploy.TilingExtension.TilerExtension import MemoryMap, TilerDeployerWrapper, TilingSolution
from Deeploy.TilingExtension.TilingCodegen import AbsoluteHyperRectangle, TilingSchedule, VariableReplacementScheme

from .tilingUtils import DBOnlyL3Tiler, DBTiler, SBTiler

memcpyTemplate = NodeTemplate("""
memcpy((void *)${dest}, (void *)${src}, ${size});
""")


# Same interface as NodeTypeChecker but allow any input type and the
# output type matches the input type.
class MemcpyTypeChecker(NodeTypeChecker):

    def __init__(self):
        super().__init__([], [])

    def typeInferOutput(self, ctxt: NetworkContext, node: gs.Node,
                        operatorRepresentation: OperatorRepresentation) -> NetworkContext:
        assert len(node.inputs) == 1 and len(node.outputs) == 1
        buffer_in = ctxt.lookup(node.inputs[0].name)
        ctxt.annotateType(node.outputs[0].name, buffer_in._type)
        return ctxt

    def typeCheckNodeInputs(self, ctxt: NetworkContext, node: gs.Node) -> bool:
        return True

    def typeInferGlobalCtxt(self, ctxt: NetworkContext, node: gs.Node) -> NetworkContext:
        # Whatever it has already annotated, it's good
        return ctxt


class MemcpyTileConstraint(TileConstraint):

    @classmethod
    def serializeTilingSolution(
            cls, tilingSolution: NodeMemoryConstraint, absoluteOutputCubes: List[AbsoluteHyperRectangle],
            targetMemLevel: str, ctxt: NetworkContext,
            operatorRepresentation: OperatorRepresentation) -> Tuple[VariableReplacementScheme, TilingSchedule]:
        inputLoadSchedule = [{"src": absCube.rectangle} for absCube in absoluteOutputCubes]
        outputLoadSchedule = [{"dest": absCube.rectangle} for absCube in absoluteOutputCubes]
        inputOffsets, outputOffsets = cls.extractBaseAddr(tilingSolution, targetMemLevel, operatorRepresentation,
                                                          ["src", "dest"])

        def size(abs: AbsoluteHyperRectangle, buffer: VariableBuffer) -> int:
            return math.prod(abs.rectangle.dims) * (buffer._type.referencedType.typeWidth // 8)

        buffer_src = ctxt.lookup(operatorRepresentation['src'])
        assert isinstance(buffer_src, VariableBuffer)

        replacements: Dict[str, List[int]] = {"size": [size(abs, buffer_src) for abs in absoluteOutputCubes]}
        replacement_types = {key: PointerClass(minimalIntegerType(values)) for key, values in replacements.items()}

        return VariableReplacementScheme(replacements,
                                         replacement_types), TilingSchedule(inputOffsets, outputOffsets,
                                                                            inputLoadSchedule, outputLoadSchedule)


class MemcpyParser(NodeParser):

    def parseNode(self, node: gs.Node) -> bool:
        return len(node.inputs) == 1 and len(node.outputs) == 1

    def parseNodeCtxt(self,
                      ctxt: NetworkContext,
                      node: gs.Node,
                      channels_first: bool = True) -> Tuple[NetworkContext, bool]:
        assert len(node.inputs) == 1 and len(node.outputs) == 1
        src = ctxt.lookup(node.inputs[0].name)
        self.operatorRepresentation['src'] = src.name
        self.operatorRepresentation['dest'] = ctxt.lookup(node.outputs[0].name).name
        self.operatorRepresentation['size'] = math.prod(src.shape) * (src._type.referencedType.typeWidth // 8)
        return ctxt, True


class MemcpyLayer(ONNXLayer):
    pass


def generate_graph(nodeCount: int, shape: Tuple[int, ...], dtype: npt.DTypeLike) -> gs.Graph:
    assert nodeCount > 0

    tensor_in = gs.Variable(name = "input_0", dtype = dtype, shape = shape)

    nodes = []
    for i in range(nodeCount):
        tensor_out = gs.Variable(name = f"out_{i}", dtype = dtype, shape = shape)
        nodes.append(gs.Node("Memcpy", f"memcpy_{i}", {}, [tensor_in], [tensor_out]))
        tensor_in = tensor_out

    return gs.Graph(nodes, [nodes[0].inputs[0]], [nodes[-1].outputs[0]], "dma_test_graph")


def generate_tiling(ctxt: NetworkContext, memoryStart: str, memoryOrder: List[str], memoryHierarchy: MemoryHierarchy,
                    inputShape: Tuple[int, ...], tileShape: Tuple[int, ...], graph: gs.Graph, _type: BaseType,
                    doublebuffer: bool) -> Tuple[TilingSolution, MemoryMap]:
    assert memoryStart in memoryOrder
    memoryStartIndex = memoryOrder.index(memoryStart)

    if memoryStartIndex + 1 < len(memoryOrder):
        memoryMultibuffer = memoryOrder[memoryOrder.index(memoryStart) + 1]
    else:
        memoryMultibuffer = None

    if memoryStartIndex + 2 < len(memoryOrder):
        singleTileMemories = memoryOrder[memoryStartIndex + 2:]
    else:
        singleTileMemories = []

    inputSize = math.prod(inputShape)
    tileSize = math.prod(tileShape)

    def assertFitsInMemory(size: int, memory: str) -> None:
        memorySize = memoryHierarchy.memoryLevels[memory].size
        assert size <= memorySize, f"The required tensor space is too big for the {memory} memory. Required space: {size}, memory space: {memorySize}"

    inputSizeInBytes = inputSize * (_type.typeWidth // 8)
    assertFitsInMemory(2 * inputSizeInBytes, memoryStart)

    tileSizeInBytes = tileSize * (_type.typeWidth // 8)
    for memory in singleTileMemories:
        assertFitsInMemory(2 * tileSizeInBytes, memory)

    if doublebuffer:
        multiBufferCoefficient = 2
    else:
        multiBufferCoefficient = 1

    multibufferSizeInBytes = tileSizeInBytes * multiBufferCoefficient
    if memoryMultibuffer is not None:
        assertFitsInMemory(multibufferSizeInBytes + tileSizeInBytes, memoryMultibuffer)

    inputMultibufferAddrSpace = (0, multibufferSizeInBytes)
    outputMultibufferAddrSpace = (multibufferSizeInBytes, 2 * multibufferSizeInBytes)

    inputTileAddrSpace = (0, tileSizeInBytes)
    outputTileAddrSpace = (tileSizeInBytes, 2 * tileSizeInBytes)

    # Tiling Solution

    tilingSolution = []

    def generateMemoryConstraint(memory: str, shape: Tuple[int, ...], multiBufferCoefficient: int,
                                 addrSpace: Optional[Tuple[int, int]]) -> MemoryConstraint:
        size = math.prod(shape)
        mc = MemoryConstraint(memory, size)
        mc.shape = shape
        mc.multiBufferCoefficient = multiBufferCoefficient
        if addrSpace is not None:
            mc.addrSpace = addrSpace
        return mc

    for node in graph.nodes:
        inputMemoryConstraints = {}
        outputMemoryConstraints = {}
        for i, memory in enumerate(memoryOrder[memoryOrder.index(memoryStart):]):
            if i == 0:
                inputMc = generateMemoryConstraint(memory = memory,
                                                   shape = inputShape,
                                                   multiBufferCoefficient = 1,
                                                   addrSpace = None)
                outputMc = generateMemoryConstraint(memory = memory,
                                                    shape = inputShape,
                                                    multiBufferCoefficient = 1,
                                                    addrSpace = None)
            elif i == 1:
                inputMc = generateMemoryConstraint(memory = memory,
                                                   shape = tileShape,
                                                   multiBufferCoefficient = multiBufferCoefficient,
                                                   addrSpace = inputMultibufferAddrSpace)
                outputMc = generateMemoryConstraint(memory = memory,
                                                    shape = tileShape,
                                                    multiBufferCoefficient = multiBufferCoefficient,
                                                    addrSpace = outputMultibufferAddrSpace)
            else:
                inputMc = generateMemoryConstraint(memory = memory,
                                                   shape = tileShape,
                                                   multiBufferCoefficient = 1,
                                                   addrSpace = inputTileAddrSpace)
                outputMc = generateMemoryConstraint(memory = memory,
                                                    shape = tileShape,
                                                    multiBufferCoefficient = 1,
                                                    addrSpace = outputTileAddrSpace)
            inputMemoryConstraints[memory] = inputMc
            outputMemoryConstraints[memory] = outputMc

        inputTensorMemoryConstraint = TensorMemoryConstraint(tensorName = node.inputs[0].name,
                                                             constraints = inputMemoryConstraints,
                                                             ctxt = ctxt)

        outputTensorMemoryConstraint = TensorMemoryConstraint(tensorName = node.outputs[0].name,
                                                              constraints = outputMemoryConstraints,
                                                              ctxt = ctxt)

        nodeMemoryConstraint = NodeMemoryConstraint()
        nodeMemoryConstraint.addTensorConstraint(inputTensorMemoryConstraint, 'input')
        nodeMemoryConstraint.addTensorConstraint(outputTensorMemoryConstraint, 'output')

        patternMemoryConstraints = PatternMemoryConstraints()
        patternMemoryConstraints.addConstraint(nodeMemoryConstraint)

        tilingSolution.append(patternMemoryConstraints)

    # Memory Map

    # Initialize an empty memory map
    memoryMap = {memory: [[] for _ in range(len(graph.nodes) + 1)] for memory in memoryOrder}

    # Set memoryStart memory

    def appendMemoryMapStart(tensorName: str, lifetime: Tuple[int, int], addrSpace: Tuple[int, int]) -> None:
        memoryMap[memoryStart][-1].append(MemoryBlock(tensorName, memoryStart, lifetime, addrSpace))

    addrSpacePing = (0, inputSizeInBytes)
    addrSpacePong = (inputSizeInBytes, 2 * inputSizeInBytes)

    ## First input tensor has a special lifetime (0, 0)
    appendMemoryMapStart(graph.nodes[0].inputs[0].name, (0, 0), addrSpacePing)

    for i, node in enumerate(graph.nodes):
        # Start with addrSpacePong because we used "Ping" for the first input tensor
        appendMemoryMapStart(node.outputs[0].name, (i, i + 1), addrSpacePong if i % 2 == 0 else addrSpacePing)

    ## Set the rest

    def setMemoryMapRest(memory: str, inputAddrSpace: Tuple[int, int], outputAddrSpace: Tuple[int, int]) -> None:
        for i, node in enumerate(graph.nodes):
            # Empirically concluded from looking at produced memory maps
            if i + 1 == len(graph.nodes):
                endLifetime = i + 1
            else:
                endLifetime = i

            memoryMap[memory][i].extend([
                MemoryBlock(name = node.inputs[0].name, level = memory, lifetime = (i, i), addrSpace = inputAddrSpace),
                MemoryBlock(name = node.outputs[0].name,
                            level = memory,
                            lifetime = (i, endLifetime),
                            addrSpace = outputAddrSpace),
            ])

    if memoryMultibuffer is not None:
        setMemoryMapRest(memoryMultibuffer, inputMultibufferAddrSpace, outputMultibufferAddrSpace)

    for memory in singleTileMemories:
        setMemoryMapRest(memory, inputTileAddrSpace, outputTileAddrSpace)

    return tilingSolution, memoryMap


def defaultScheduler(graph: gs.Graph) -> List[List[gs.Node]]:
    return [[node] for node in graph.nodes]


def setup_pulp_deployer(defaultMemory: str, targetMemory: str, graph: gs.Graph, inputTypes: Dict[str, Type[Pointer]],
                        doublebuffer: bool, deeployStateDir: str) -> NetworkDeployer:
    L3 = MemoryLevel(name = "L3", neighbourNames = ["L2"], size = 64000000)
    L2 = MemoryLevel(name = "L2", neighbourNames = ["L3", "L1"], size = 1024000)
    L1 = MemoryLevel(name = "L1", neighbourNames = ["L2"], size = 64000)
    memoryLevels = [L3, L2, L1]
    memoryLevelMap = {mem.name: mem for mem in memoryLevels}

    assert defaultMemory in memoryLevelMap, f"defaultMemory {defaultMemory} is not part of PULP's memory hierarchy {list(memoryLevelMap.keys())}"
    assert targetMemory in memoryLevelMap, f"targetMemory {targetMemory} is not part of PULP's memory hierarchy {list(memoryLevelMap.keys())}"

    memoryHierarchy = MemoryHierarchy(memoryLevels)
    memoryHierarchy.setDefaultMemoryLevel(defaultMemory)

    platform = MemoryPULPPlatform(memoryHierarchy, memoryLevelMap[targetMemory])

    deployer = PULPDeployer(graph,
                            platform,
                            inputTypes,
                            PULPOptimizer,
                            defaultScheduler,
                            default_channels_first = True,
                            deeployStateDir = deeployStateDir)

    memoryLevelAnnotationPasses = [AnnotateIOMemoryLevel(defaultMemory), AnnotateDefaultMemoryLevel(memoryHierarchy)]
    # Make the deployer memory-level aware
    deployer = MemoryDeployerWrapper(deployer, memoryLevelAnnotationPasses)

    if doublebuffer:
        assert defaultMemory in ["L3", "L2"]
        if defaultMemory == "L3":
            deployer = TilerDeployerWrapper(deployer, DBOnlyL3Tiler)
        else:
            deployer = TilerDeployerWrapper(deployer, DBTiler)
    else:
        deployer = TilerDeployerWrapper(deployer, SBTiler)

    return deployer


def setup_snitch_deployer(defaultMemory: str, targetMemory: str, graph: gs.Graph, inputTypes: Dict[str, Type[Pointer]],
                          doublebuffer: bool, deeployStateDir: str) -> NetworkDeployer:
    L3 = MemoryLevel(name = "L3", neighbourNames = ["L2"], size = 64000000)
    L2 = MemoryLevel(name = "L2", neighbourNames = ["L3", "L1"], size = 1024000)
    L1 = MemoryLevel(name = "L1", neighbourNames = ["L2"], size = 64000)
    memoryLevels = [L3, L2, L1]
    memoryLevelMap = {mem.name: mem for mem in memoryLevels}

    assert defaultMemory in memoryLevelMap, f"defaultMemory {defaultMemory} is not part of PULP's memory hierarchy {list(memoryLevelMap.keys())}"
    assert targetMemory in memoryLevelMap, f"targetMemory {targetMemory} is not part of PULP's memory hierarchy {list(memoryLevelMap.keys())}"

    memoryHierarchy = MemoryHierarchy(memoryLevels)
    memoryHierarchy.setDefaultMemoryLevel(defaultMemory)

    platform = SnitchPlatform()
    platform = MemoryPlatformWrapper(platform, memoryHierarchy, memoryLevelMap[targetMemory])

    deployer = SnitchDeployer(graph,
                              platform,
                              inputTypes,
                              SnitchOptimizer,
                              defaultScheduler,
                              deeployStateDir = deeployStateDir)
    memoryLevelAnnotationPasses = [AnnotateIOMemoryLevel(defaultMemory), AnnotateDefaultMemoryLevel(memoryHierarchy)]
    # Make the deployer memory-level aware
    deployer = MemoryDeployerWrapper(deployer, memoryLevelAnnotationPasses)

    assert defaultMemory == "L2"
    if doublebuffer:
        deployer = TilerDeployerWrapper(deployer, DBTiler)
    else:
        deployer = TilerDeployerWrapper(deployer, SBTiler)

    return deployer


def prepare_deployer_with_custom_tiling(deployer: NetworkDeployer, defaultMemory: str, targetMemory: str,
                                        tileShape: Tuple[int, ...], doublebuffer: bool) -> None:
    # Decomposed deployer.prepare() to enter a custom tiling solution
    deployer.frontEnd()
    super(TilerDeployerWrapper, deployer).bind()

    tilingSolution, memoryMap = generate_tiling(
        ctxt = deployer.ctxt,
        memoryStart = defaultMemory,
        memoryOrder = [defaultMemory, targetMemory],
        memoryHierarchy = deployer.Platform.memoryHierarchy,
        inputShape = deployer.graph.inputs[0].shape,
        tileShape = tileShape,
        graph = deployer.graph,
        _type = deployer.inputTypes['input_0'].referencedType,
        doublebuffer = doublebuffer,
    )
    deployer.tile(tilingSolution, memoryMap)
    deployer.backEnd()
    deployer.prepared = True

# ----------------------------------------------------------------------
#
# File: TilerExtension.py
#
# Last edited: 09.05.2023
#
# Copyright (C) 2023, ETH Zurich and University of Bologna.
#
# Author:
# - Victor Jung, jungvi@iis.ee.ethz.ch, ETH Zurich
# - Moritz Scherer, scheremo@iis.ee.ethz.ch, ETH Zurich
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

# Create Monad that take a Deployer and make it TilerAware
# Define Tiler Obj centralize all tilling related functionalities for a given deployer.
# Like Template-T-Obj mapping, propagate cst, graph edition, etc

import copy
import csv
import os
import subprocess
from typing import Dict, List, Literal, Optional, OrderedDict, Tuple, Type, Union

import numpy as np
import onnx_graphsurgeon as gs
import plotly.graph_objects as go
import plotly.io as pio
from ortools.constraint_solver.pywrapcp import IntVar, SolutionCollector

import Deeploy.CommonExtensions.DataTypes as BasicDataTypes
from Deeploy.AbstractDataTypes import PointerClass
from Deeploy.CommonExtensions.NetworkDeployers.NetworkDeployerWrapper import NetworkDeployerWrapper
from Deeploy.DeeployTypes import ConstantBuffer, NetworkContext, NodeBinding, NodeTemplate, ONNXLayer, Schedule, \
    SubGraph, TransientBuffer, VariableBuffer
from Deeploy.MemoryLevelExtension.MemoryLevels import MemoryHierarchy, MemoryLevel
from Deeploy.MemoryLevelExtension.NetworkDeployers.MemoryLevelDeployer import MemoryDeployerWrapper, \
    MemoryLevelAwareDeployer, MemoryPlatform, MemoryPlatformWrapper, TargetMemoryLevelMapping
from Deeploy.TilingExtension.GenericFlow import GenericFlowState
from Deeploy.TilingExtension.MemoryConstraintFlows import GraphMemoryConstraintFlow, TensorMemLevelTuple, \
    convertFlowState2NodeMemoryConstraint
from Deeploy.TilingExtension.MemoryConstraints import MemoryConstraint, NodeMemoryConstraint, \
    PatternMemoryConstraints, TensorMemoryConstraint
from Deeploy.TilingExtension.MemoryScheduler import MemoryBlock, MemoryScheduler
from Deeploy.TilingExtension.TileConstraint import TileConstraint
from Deeploy.TilingExtension.TilerModel import TilerModel

TilingSolution = List[PatternMemoryConstraints]

_deallocTemplate = NodeTemplate("")


class Tiler():

    arenaName = "MEMORYARENA"
    memorySchedulerClass: Type[MemoryScheduler] = MemoryScheduler

    _MINIMALLOC_INPUT_FILENAME = "input_minimalloc"
    _MINIMALLOC_OUTPUT_FILENAME = "output_minimalloc"

    # Initialize with the list of TemplateTCFbinding
    def __init__(self, memoryHierarchy: MemoryHierarchy):

        self.memoryHierarchy = memoryHierarchy
        self.tilerModel: Optional[TilerModel] = None
        self.innerMemoryScheduler = self.memorySchedulerClass("_inner", tileScheduler = True)
        self.outerMemoryScheduler = self.memorySchedulerClass("_outer", tileScheduler = False)
        self.symbolicMemoryConstraints: Optional[List[PatternMemoryConstraints]] = None

        self._worstCaseBufferSize: Dict[str, int] = {}

        self.visualizeMemoryAlloc: bool = False
        self.memoryAllocStrategy: Literal["TetrisRandom", "TetrisCo-Opt", "MiniMalloc"] = "TetrisRandom"
        self.searchStrategy: Literal["min", "max", "random-max"] = "random-max"

    @property
    def worstCaseBufferSize(self):
        return self._worstCaseBufferSize

    def plotMemoryAlloc(self,
                        memoryMap: Dict[str, List[List[MemoryBlock]]],
                        ctxt: NetworkContext,
                        deeployStateDir: str,
                        defaultMemoryLevel: MemoryLevel,
                        targetMemLevelName: str = 'L1'):

        innerMemoryAllocDir = os.path.join(deeployStateDir, f"MemoryAlloc{targetMemLevelName}")
        os.makedirs(os.path.abspath(deeployStateDir), exist_ok = True)
        os.makedirs(os.path.abspath(innerMemoryAllocDir), exist_ok = True)
        defaultMemLevelPlotPath = os.path.abspath(
            os.path.join(deeployStateDir, f"memory_alloc_{defaultMemoryLevel.name}.html"))

        addTraceConfig = {"fill": "toself", "hoverinfo": "text", "mode": "lines", "line": dict(width = 2)}

        updateLayoutConfig = {
            "xaxis_title": "Lifetime",
            "yaxis_title": "Address Space (Bytes)",
            "xaxis": dict(tickformat = "d", showgrid = True),
            "yaxis": dict(tickformat = "d", showgrid = True),
            "hovermode": "closest",
            "showlegend": False,
        }

        fig = go.Figure()

        # JUNGVI: Currently I/O have infinite lifetime, will change that soon...
        infiniteLifetimeBuffers = [
            buffer for buffer in ctxt.globalObjects.values()
            if not self.arenaName in buffer.name and isinstance(buffer, ConstantBuffer)
        ]

        constantBuffersOffset = 0
        for ioBuffers in infiniteLifetimeBuffers:
            _ioSize = np.prod(ioBuffers.shape) * ioBuffers._type.referencedType.typeWidth // 8
            _maxLifetime = len(memoryMap[defaultMemoryLevel.name])
            fig.add_trace(
                go.Scatter(x = [-0.5, -0.5, _maxLifetime + 0.5, _maxLifetime + 0.5],
                           y = [
                               constantBuffersOffset, constantBuffersOffset + _ioSize, constantBuffersOffset + _ioSize,
                               constantBuffersOffset
                           ],
                           name = ioBuffers.name,
                           text = ioBuffers.name,
                           **addTraceConfig))
            constantBuffersOffset += _ioSize

        for buffer in memoryMap[defaultMemoryLevel.name][-1]:
            if hasattr(ctxt.lookup(buffer.name), "_alias"):
                continue
            fig.add_trace(
                go.Scatter(x = [
                    buffer._lifetime[0] - 0.5, buffer._lifetime[0] - 0.5, buffer._lifetime[1] + 0.5,
                    buffer._lifetime[1] + 0.5
                ],
                           y = [
                               constantBuffersOffset + buffer._addrSpace[0],
                               constantBuffersOffset + buffer._addrSpace[1],
                               constantBuffersOffset + buffer._addrSpace[1],
                               constantBuffersOffset + buffer._addrSpace[0]
                           ],
                           name = buffer.name,
                           text = buffer.name,
                           **addTraceConfig))

        fig.add_trace(
            go.Scatter(
                x = [-0.5, len(memoryMap[defaultMemoryLevel.name]) - 1.5],
                y = [defaultMemoryLevel.size, defaultMemoryLevel.size],
                name = f"{defaultMemoryLevel.name} Memory Size",
                text = f"{defaultMemoryLevel.name} Memory Size",
                line = dict(color = "red", width = 2, dash = "dash"),
                fill = "toself",
                hoverinfo = "text",
                mode = "lines",
            ))
        fig.update_layout(title = f"Deeploy Memory Allocation {defaultMemoryLevel.name}", **updateLayoutConfig)
        pio.write_html(fig, defaultMemLevelPlotPath)

        for step_idx, innerMemoryAlloc in enumerate(memoryMap[targetMemLevelName]):
            targetMemLevelPlotPath = os.path.abspath(
                os.path.join(innerMemoryAllocDir, f"memory_alloc_{targetMemLevelName}_step{step_idx}.html"))
            fig = go.Figure()
            for buffer in innerMemoryAlloc:
                fig.add_trace(
                    go.Scatter(
                        x = [
                            buffer._lifetime[0] - 0.5, buffer._lifetime[0] - 0.5, buffer._lifetime[1] + 0.5,
                            buffer._lifetime[1] + 0.5
                        ],
                        y = [buffer._addrSpace[0], buffer._addrSpace[1], buffer._addrSpace[1], buffer._addrSpace[0]],
                        name = buffer.name,
                        text = buffer.name,
                        **addTraceConfig))
            fig.update_layout(title = f"Deeploy Memory Allocation {targetMemLevelName} Step {step_idx}",
                              **updateLayoutConfig)
            pio.write_html(fig, targetMemLevelPlotPath)

    def _convertCtxtToStaticSchedule(self, ctxt: NetworkContext,
                                     memoryMap: Dict[str, List[List[MemoryBlock]]]) -> NetworkContext:

        maxAddr: Dict[str, int] = {}

        for memoryLevel, patternList in memoryMap.items():
            currentMax = 0
            for nodeList in patternList:
                blockNames = [block.name for block in nodeList]
                for node in nodeList:

                    _buffer = ctxt.lookup(node.name)
                    # SCHEREMO: If alias buffers have zero cost, they don't contribute to the currentMax and their addrSpace is None
                    if hasattr(_buffer, "_alias") and (ctxt.is_global(_buffer._alias) or _buffer._alias in blockNames):
                        continue

                    currentMax = max(currentMax, node._addrSpace[1])

            maxAddr[memoryLevel] = currentMax
            self._worstCaseBufferSize[memoryLevel] = currentMax

        for level, addrSpace in maxAddr.items():
            if addrSpace == 0:
                continue

            arenaName = f"{self.arenaName}_{level}"

            scratchBuffer = ctxt.VariableBuffer(arenaName, [addrSpace])
            scratchBuffer._type = PointerClass(BasicDataTypes.int8_t)
            ctxt.add(scratchBuffer, "global")
            scratchBuffer._instance = scratchBuffer._type(arenaName, ctxt)
            scratchBuffer._memoryLevel = level

        # SCHEREMO: Adapt homelevel tensors to their respective arena
        for memoryLevel, patternList in memoryMap.items():
            if not ctxt.is_global(f"{self.arenaName}_{memoryLevel}"):
                continue
            staticBuf = ctxt.lookup(f"{self.arenaName}_{memoryLevel}")
            for nodeList in patternList:
                blockNames = [block.name for block in nodeList]
                for node in nodeList:
                    tensorName = node.name
                    _buffer = ctxt.lookup(tensorName)

                    if _buffer._memoryLevel != memoryLevel:
                        continue

                    if hasattr(_buffer, "_alias") and ctxt.is_global(_buffer._alias):
                        continue

                    if hasattr(_buffer, "_alias") and _buffer._alias in blockNames:

                        alias = ctxt.dealiasBuffer(tensorName)
                        aliasNodes = [node for node in nodeList if node.name == alias]

                        assert len(aliasNodes) == 1, f"alias {alias} references more than one node!"

                        aliasNode = aliasNodes[0]

                        _buffer.allocTemplate = NodeTemplate(
                            " \
                        ${name} = (${type.typeName}) " +
                            f"((char*){str(staticBuf._instance)} + {aliasNode.addrSpace[0]});")
                        _buffer.deallocTemplate = _deallocTemplate

                        continue

                    offset = node.addrSpace[0]

                    _buffer.allocTemplate = NodeTemplate(" \
                    ${name} = (${type.typeName}) " + f"((char*){str(staticBuf._instance)} + {offset});")
                    _buffer.deallocTemplate = _deallocTemplate

        return ctxt

    def minimalloc(self, memoryMap, ctxt, nodeMemoryConstraint, capacity: int, memoryLevel: str):

        with open(f"{self._MINIMALLOC_INPUT_FILENAME}.csv", mode = "w", newline = "") as file:
            writer = csv.writer(file, lineterminator = "\n")
            writer.writerow(["id", "lower", "upper", "size"])
            for memoryBlock in memoryMap:

                _buffer = ctxt.lookup(memoryBlock.name)
                if nodeMemoryConstraint is None:
                    _bufferSize = _buffer.size if isinstance(
                        _buffer,
                        TransientBuffer) else np.prod(_buffer.shape) * (_buffer._type.referencedType.typeWidth / 8)
                else:
                    if isinstance(_buffer, TransientBuffer):
                        _bufferSize = nodeMemoryConstraint.tensorMemoryConstraints[
                            memoryBlock.name].memoryConstraints[memoryLevel].size
                    else:
                        _bufferSize = nodeMemoryConstraint.tensorMemoryConstraints[
                            memoryBlock.name].memoryConstraints[memoryLevel].size * (
                                _buffer._type.referencedType.typeWidth /
                                8) * nodeMemoryConstraint.tensorMemoryConstraints[
                                    memoryBlock.name].memoryConstraints[memoryLevel].multiBufferCoefficient

                writer.writerow([
                    memoryBlock.name,
                    str(memoryBlock.lifetime[0]),
                    str(memoryBlock.lifetime[1] + 1),
                    str(int(_bufferSize))
                ])

        try:
            minimallocInstallDir = os.environ["MINIMALLOC_INSTALL_DIR"]
        except KeyError:
            raise KeyError("MINIMALLOC_INSTALL_DIR symbol not found!")

        minimallocOutput = subprocess.run([
            f"{minimallocInstallDir}/minimalloc", f"--capacity={capacity}",
            f"--input={self._MINIMALLOC_INPUT_FILENAME}.csv", f"--output={self._MINIMALLOC_OUTPUT_FILENAME}.csv"
        ],
                                          capture_output = True,
                                          text = True)

        if minimallocOutput.returncode != 0:
            print(
                f"\033[91mError: Memory allocator failed with return code {minimallocOutput.returncode} at memory level {memoryLevel} with capacity of {capacity} bytes \033[0m"
            )
            raise subprocess.CalledProcessError(minimallocOutput.returncode, " ".join(minimallocOutput.args))

        with open(f"{self._MINIMALLOC_OUTPUT_FILENAME}.csv", mode = "r", newline = "") as file:
            reader = csv.reader(file)
            header = next(reader)
            for row in reader:
                for memoryBlock in memoryMap:
                    if memoryBlock.name == row[0]:
                        memoryBlock._addrSpace = (int(row[-1]), int(row[-1]) + int(row[-2]))

        return memoryMap

    def computeTilingSchedule(self, ctxt: NetworkContext) -> Tuple[TilingSolution, Dict[str, List[List[MemoryBlock]]]]:

        assert self.tilerModel is not None and self.symbolicMemoryConstraints is not None, "Set up the model before trying to compute a schedule!"

        collector = self.tilerModel.trySolveModel()
        tilingSchedule = self._getTilingSolution(self.tilerModel, ctxt, collector, self.symbolicMemoryConstraints)

        if not self.memoryAllocStrategy == "MiniMalloc":
            self.innerMemoryScheduler.annotateSolution(ctxt, self.tilerModel)
            self.outerMemoryScheduler.annotateSolution(ctxt, self.tilerModel)

        memoryMap = {}

        for key in self.innerMemoryScheduler.memoryMap.keys():
            memoryMap[key] = [*self.innerMemoryScheduler.memoryMap[key], *self.outerMemoryScheduler.memoryMap[key]]

        if self.memoryAllocStrategy == "MiniMalloc":
            for memoryLevel in memoryMap.keys():
                constantTensorOffset = self.outerMemoryScheduler.getConstantTensorOffset(ctxt, memoryLevel)
                if memoryLevel == self.memoryHierarchy._defaultMemoryLevel.name:
                    memoryMap[memoryLevel][-1] = self.minimalloc(
                        memoryMap[memoryLevel][-1], ctxt, None,
                        self.memoryHierarchy.memoryLevels[memoryLevel].size - constantTensorOffset, memoryLevel)
                else:
                    for idx, memMap in enumerate(memoryMap[memoryLevel]):
                        if len(memoryMap[memoryLevel][idx]) != 0:
                            memoryMap[memoryLevel][idx] = self.minimalloc(
                                memMap, ctxt, tilingSchedule[idx].nodeConstraints[0],
                                self.memoryHierarchy.memoryLevels[memoryLevel].size - constantTensorOffset, memoryLevel)
            print(f"\033[92mMemory allocation sucessful!\033[0m")

        for idx, pattern in enumerate(tilingSchedule):
            for nodeIdx, nodeConstraint in enumerate(pattern.nodeConstraints):
                for tensorConstraint in nodeConstraint.tensorMemoryConstraints.values():
                    for memoryConstraint in tensorConstraint.memoryConstraints.values():
                        patternList = memoryMap[memoryConstraint.memoryLevel]
                        blockPattern = patternList[idx]

                        # SCHEREMO: Don't try to annotate home base of tensor
                        if ctxt.lookup(tensorConstraint.tensorName
                                      )._memoryLevel == memoryConstraint.memoryLevel and not isinstance(
                                          ctxt.lookup(tensorConstraint.tensorName), TransientBuffer):
                            continue

                        _block = [memBlock for memBlock in blockPattern if memBlock.name == tensorConstraint.tensorName]

                        assert len(
                            _block
                        ) == 1, f"Missing or superfluous memory block {tensorConstraint.tensorName} allocation found in {_block}!"

                        block = _block[0]
                        memoryConstraint.addrSpace = block.addrSpace

        self._convertCtxtToStaticSchedule(ctxt, memoryMap)

        return tilingSchedule, memoryMap

    def setupModel(self, ctxt: NetworkContext, schedule: Schedule, layerBinding: 'OrderedDict[str, ONNXLayer]',
                   targetMemoryLevelMapping: TargetMemoryLevelMapping,
                   outputBufferList: List[VariableBuffer]) -> NetworkContext:

        wrapSchedule: List[SubGraph] = []
        for entry in schedule:
            if isinstance(entry, gs.Node):
                wrapSchedule.append([entry])
            else:
                wrapSchedule.append(entry)

        tilerModel = TilerModel(searchStrategy = self.searchStrategy)
        tilerModel = self._setupGeometricConstraints(tilerModel, ctxt, wrapSchedule, layerBinding)
        tilerModel = self._setupTensorDimensionProducts(tilerModel, ctxt, wrapSchedule)
        tilerModel = self._setupHeuristics(tilerModel, ctxt, wrapSchedule)
        tilerModel, allSymbolicMemoryConstraints = self._setupMemoryConstraints(tilerModel, ctxt, wrapSchedule,
                                                                                layerBinding, targetMemoryLevelMapping,
                                                                                outputBufferList)

        self.tilerModel = tilerModel
        self.symbolicMemoryConstraints = allSymbolicMemoryConstraints

        return ctxt

    # SCHEREMO: Return a integer factor or IntVar variable for the multi Buffer coefficient given the tiling path, hop and tensorName.
    def multiBufferStrategy(self, tilerModel: TilerModel, ctxt: NetworkContext, pattern: SubGraph, path: List[str],
                            hop: str, tensorName: str) -> Union[int, IntVar]:

        varBuffer = ctxt.lookup(tensorName)

        generalCoeff = 2

        if isinstance(varBuffer, TransientBuffer):
            coefficient = 1
        elif isinstance(varBuffer, ConstantBuffer):
            coefficient = generalCoeff
        else:
            coefficient = generalCoeff

        # if tensorName == pattern[-1].outputs[0].name:
        #     maxVal = (np.prod(varBuffer.shape) // (coefficient)).item()
        #     numElt = tilerModel.getTensorNumberOfEltVar(tensorName)
        #     constr = numElt <= maxVal

        #     if (constr != True):
        #         tilerModel.addConstraint(constr)

        return coefficient

    # SCHEREMO: Given a PatternMemoryConstraints object, propagate the IOBuffer freeing strategy.
    # Input: Single-buffered Liveness analysis of the input/output IO buffers that should be tiled
    # Output: Buffering-strategy aware liveness analysis of the input/output IO buffers

    # This version implements "static n-ple buffering"

    def propagateIOBufferStrategy(self, tileConstraintPattern: PatternMemoryConstraints, pattern: SubGraph,
                                  ctxt: NetworkContext) -> PatternMemoryConstraints:

        borderTensorStep = NodeMemoryConstraint()
        for patternStep in tileConstraintPattern.nodeConstraints:
            borderTensorStep += patternStep

        for idx in range(len(tileConstraintPattern.nodeConstraints)):
            tileConstraintPattern.nodeConstraints[idx] += borderTensorStep

        return tileConstraintPattern

    def _resolveTensorMemoryConstraint(self, tilerModel: TilerModel, ctxt: NetworkContext, collector: SolutionCollector,
                                       tensorConstraint: TensorMemoryConstraint) -> TensorMemoryConstraint:
        assert self.tilerModel is not None, "Can't resolve tensor memory constraints, tilerModel is None!"

        tensorName = tensorConstraint.tensorName
        solvedTensorConstraint = TensorMemoryConstraint(tensorName, {}, ctxt)

        for memoryLevel, memoryConstraint in tensorConstraint.memoryConstraints.items():
            size = self.tilerModel._resolveVariable(memoryConstraint.size)

            newMemoryConstraint: MemoryConstraint = MemoryConstraint(memoryLevel, size)
            multiBufferCoefficient = self.tilerModel._resolveVariable(memoryConstraint.multiBufferCoefficient)
            newMemoryConstraint.multiBufferCoefficient = multiBufferCoefficient

            if not isinstance(ctxt.lookup(tensorName), TransientBuffer):

                tensorShapeLen = len(ctxt.lookup(tensorName).shape)
                newShape: List[int] = []

                if isinstance(memoryConstraint.size, int):
                    newShape = ctxt.lookup(tensorName).shape
                else:
                    _, copyIdx = tilerModel.getNameCopyIdx(memoryConstraint.size.Name())
                    for i in range(tensorShapeLen):
                        newShape.append(
                            self.tilerModel._resolveVariable(tilerModel.getTensorDimVar(tensorName, i, copyIdx)))

                newMemoryConstraint.shape = tuple(newShape)

            solvedTensorConstraint.addMemoryConstraint(newMemoryConstraint)

        return solvedTensorConstraint

    def _getTilingSolution(self, tilerModel: TilerModel, ctxt: NetworkContext, collector: SolutionCollector,
                           allConstraints: List[PatternMemoryConstraints]) -> List[PatternMemoryConstraints]:

        retList = []

        def _checkResolve(ctxt, tensorName, tensorConstraint):

            if ctxt.is_global(tensorName) and len(tensorConstraint.memoryConstraints.values()) <= 1:
                return False
            if len(tensorConstraint.memoryConstraints.values()) <= 1 and not isinstance(
                    ctxt.lookup(tensorName), TransientBuffer):
                return False
            return True

        for patternConstraints in allConstraints:
            newMemoryConstraint = PatternMemoryConstraints()
            for stepConstraints in patternConstraints.nodeConstraints:
                newStepMemoryConstraint = NodeMemoryConstraint()
                for tensorName, tensorConstraint in stepConstraints.tensorMemoryConstraints.items():
                    if _checkResolve(ctxt, tensorName, tensorConstraint):
                        solvedTensorConstraint = self._resolveTensorMemoryConstraint(
                            tilerModel, ctxt, collector, tensorConstraint)
                        ioDir = stepConstraints.getIO(tensorName)
                        newStepMemoryConstraint.addTensorConstraint(solvedTensorConstraint, ioDir)

                newMemoryConstraint.addConstraint(newStepMemoryConstraint)
            retList.append(newMemoryConstraint)

        return retList

    def _setupTensorDimensionProducts(self, tilerModel: TilerModel, ctxt: NetworkContext,
                                      schedule: List[SubGraph]) -> TilerModel:

        for idx, pattern in enumerate(schedule):
            subGraph = gs.Graph(nodes = pattern)
            subgraphTensors: OrderedDict[str, gs.Tensor] = subGraph.tensors(check_duplicates = True)

            for _, tensor in subgraphTensors.items():
                if not ctxt.lookup(tensor.name)._deploy:
                    continue

                tilerModel.addTensorNumOfEltToModel(ctxt, tensor.name, idx)

        return tilerModel

    def _setupGeometricConstraints(self, tilerModel: TilerModel, ctxt: NetworkContext, schedule: List[SubGraph],
                                   layerBinding: OrderedDict[str, ONNXLayer]) -> TilerModel:

        # SCHEREMO: Each pattern is a decoupled sub-problem w.r.t the geometric constraints.
        # We need to regenerate dimension variables for each tensor
        # This is done by setting the copyIdx in the tilerModel

        for idx, pattern in enumerate(schedule):
            tilerModel.copyIdx = idx

            for node in pattern:

                if node.name not in layerBinding.keys():
                    continue

                parseDict = layerBinding[node.name].mapper.parser.operatorRepresentation
                template = layerBinding[node.name].mapper.binder.template

                tilerModel = template.tileConstraint.addGeometricalConstraint(tilerModel,
                                                                              parseDict = parseDict,
                                                                              ctxt = ctxt)

                tilerModel = template.tileConstraint.addPolicyConstraint(tilerModel, parseDict = parseDict, ctxt = ctxt)

        return tilerModel

    def _setupHeuristics(self, tilerModel: TilerModel, ctxt: NetworkContext, schedule: List[SubGraph]) -> TilerModel:

        for idx, pattern in enumerate(schedule):

            patternTensorList = []
            seenTensorNameList = []
            for node in pattern:
                for gsTensor in node.inputs + node.outputs:
                    ctxtTensor = ctxt.lookup(gsTensor.name)
                    if ctxtTensor.name not in seenTensorNameList:
                        seenTensorNameList.append(ctxtTensor.name)
                        patternTensorList.append(ctxtTensor)

            patternMemSizeExpr: IntVar = 0
            for tensor in patternTensorList:
                if not ctxt.lookup(tensor.name)._deploy:
                    continue

                patternMemSizeExpr += tilerModel.getTensorNumberOfEltVar(
                    tensorName = tensor.name, copyIdx = idx) * (tensor._type.referencedType.typeWidth // 8)

            if isinstance(patternMemSizeExpr, int):
                _max = patternMemSizeExpr
            else:
                _max = patternMemSizeExpr.Max()

            patternVariable = tilerModel.addVariable(name = "DEEPLOY_PATTERN_MEM",
                                                     lowerBound = 1,
                                                     upperBound = _max,
                                                     copyIdx = idx)
            tilerModel.addConstraint(patternVariable == patternMemSizeExpr)

            tilerModel.addObjective(patternVariable, 'maximize')

        return tilerModel

    def _setupMemoryConstraints(
            self, tilerModel: TilerModel, ctxt: NetworkContext, schedule: List[SubGraph],
            layerBinding: 'OrderedDict[str, ONNXLayer]', targetMemoryLevelMapping: TargetMemoryLevelMapping,
            outputBufferList: List[VariableBuffer]) -> Tuple[TilerModel, List[PatternMemoryConstraints]]:

        allMemoryConstraints = self._generateAllMemoryConstraints(tilerModel, ctxt, schedule, layerBinding,
                                                                  targetMemoryLevelMapping)

        outerMemoryConstraints = PatternMemoryConstraints()
        for constraint in allMemoryConstraints:
            for nodeConstraint in constraint.nodeConstraints:
                outerMemoryConstraints.addConstraint(nodeConstraint)

        if self.memoryAllocStrategy == "MiniMalloc":
            # JUNGVI: This method adds the memory constraints in case of decoupled tiling and memory allocation.
            self.outerMemoryScheduler.constraintTileBuffersWithOverlappingLifetime(tilerModel, ctxt,
                                                                                   outerMemoryConstraints,
                                                                                   self.memoryHierarchy)

        for level in self.memoryHierarchy.memoryLevels.keys():
            self.outerMemoryScheduler.scheduleMemoryConstraints(tilerModel, ctxt, [outerMemoryConstraints],
                                                                self.memoryHierarchy, self.memoryAllocStrategy,
                                                                outputBufferList, level)

        # Update inner memoryHierarchy with outer constraints
        innerMemoryHierarchy = MemoryHierarchy([])
        for level, memLevel in self.memoryHierarchy.memoryLevels.items():
            newMemLevel = copy.copy(memLevel)

            if not self.memoryAllocStrategy == "MiniMalloc":
                outerConstraint = tilerModel.getVariable(self.outerMemoryScheduler.getSymbolicCostName(0, level), 0)
                newMemLevel.size = newMemLevel.size - outerConstraint

            innerMemoryHierarchy._add(newMemLevel)

        for level in innerMemoryHierarchy.memoryLevels.keys():
            self.innerMemoryScheduler.scheduleMemoryConstraints(tilerModel, ctxt, allMemoryConstraints,
                                                                innerMemoryHierarchy, self.memoryAllocStrategy,
                                                                outputBufferList, level)

        return tilerModel, allMemoryConstraints

    def _generateAllMemoryConstraints(
            self, tilerModel: TilerModel, ctxt: NetworkContext, schedule: List[SubGraph],
            layerBinding: OrderedDict[str, ONNXLayer],
            targetMemoryLevelMapping: TargetMemoryLevelMapping) -> List[PatternMemoryConstraints]:

        dynamicTensorConstraints, constantTensorConstraints = self._generateMemoryConstraints(
            tilerModel, ctxt, schedule, layerBinding, targetMemoryLevelMapping)

        allConstraints: List[PatternMemoryConstraints] = []
        # Initialize structures

        for pattern in dynamicTensorConstraints:
            allPattern = PatternMemoryConstraints()
            for step in pattern.nodeConstraints:
                allStep = step + constantTensorConstraints
                allPattern.addConstraint(allStep)
            allConstraints.append(allPattern)

        return allConstraints

    def _generateMemoryConstraints(
        self, tilerModel: TilerModel, ctxt: NetworkContext, schedule: List[SubGraph],
        layerBinding: OrderedDict[str, ONNXLayer], targetMemoryLevelMapping: TargetMemoryLevelMapping
    ) -> Tuple[List[PatternMemoryConstraints], NodeMemoryConstraint]:

        # SCHEREMO: Construct non-double-buffered constraints of local variable buffers

        outerVariableConstraints, innerVariableConstraints = self._generateVariableBufferConstraints(
            tilerModel, ctxt, schedule, layerBinding, targetMemoryLevelMapping)

        # SCHEREMO: Construct global buffer constraints

        constantBufferConstraint = self._generateBufferConstraints(ctxt)

        # SCHEREMO: Construct first-level constraint set (all global buffers + tensors stored in higher level)

        firstLevelConstraints: List[PatternMemoryConstraints] = copy.copy(outerVariableConstraints)
        for patternConstraint in firstLevelConstraints:
            for idx in range(len(patternConstraint.nodeConstraints)):
                patternConstraint.nodeConstraints[idx] += constantBufferConstraint

        # SCHEREMO: Construct constraint set for tiled tensors (including double buffering, excluding static global constraints)
        tiledTensorConstraints: List[PatternMemoryConstraints] = self._generateTilePathConstraints(
            tilerModel, ctxt, firstLevelConstraints, innerVariableConstraints, schedule)

        # SCHEREMO: Construct constraint set for tiled tensors + local-only tensors (dynamic tensor set)
        dynamicTensorConstraints: List[PatternMemoryConstraints] = []
        for tilingConstraints, innerConstraints in zip(tiledTensorConstraints, innerVariableConstraints):

            dynamicTensorPattern = PatternMemoryConstraints()
            for tilingPatternStep, innerPatternStep in zip(tilingConstraints.nodeConstraints,
                                                           innerConstraints.nodeConstraints):
                dynamicTensorPatternStep = copy.copy(tilingPatternStep)

                # Pick all constraints that are purely internal
                for innerTensorName, innerTensor in innerPatternStep.tensorMemoryConstraints.items():
                    if not any([
                            innerTensorName == dynamicTensorName
                            for dynamicTensorName, tensor in dynamicTensorPatternStep.tensorMemoryConstraints.items()
                    ]):
                        ioDir = tilingPatternStep.getIO(innerTensorName)
                        dynamicTensorPatternStep.addTensorConstraint(innerTensor, ioDir)
                dynamicTensorPattern.addConstraint(dynamicTensorPatternStep)
            dynamicTensorConstraints.append(dynamicTensorPattern)

        # SCHEREMO: Construct unkilled tensor set
        inplaceTensorConstraints: List[PatternMemoryConstraints] = []
        for tilingConstraints, outerConstraints in zip(dynamicTensorConstraints, firstLevelConstraints):
            dynamicTensorPattern = PatternMemoryConstraints()
            for tilingPatternStep, outerPatternStep in zip(tilingConstraints.nodeConstraints,
                                                           outerConstraints.nodeConstraints):
                dynamicTensorPatternStep = copy.copy(tilingPatternStep)

                # Pick all constraints that are purely internal
                for outerTensorName, outerTensor in outerPatternStep.tensorMemoryConstraints.items():
                    if not any(
                        [(outerTensorName == dynamicTensorName) or (ctxt.is_global(outerTensorName))
                         for dynamicTensorName, tensor in dynamicTensorPatternStep.tensorMemoryConstraints.items()]):
                        dynamicTensorPatternStep.addTensorConstraint(outerTensor, "intermediate")
                dynamicTensorPattern.addConstraint(dynamicTensorPatternStep)
            inplaceTensorConstraints.append(dynamicTensorPattern)

        return inplaceTensorConstraints, constantBufferConstraint

    def _generateTilePath(self, tilerModel: TilerModel, ctxt: NetworkContext,
                          tensorMemoryConstraint: TensorMemoryConstraint, pattern: SubGraph) -> TensorMemoryConstraint:

        assert len(tensorMemoryConstraint.memoryConstraints.keys()
                  ) == 2, "Can't generate a tile path for more than one hierarchy level!"

        tensorName = tensorMemoryConstraint.tensorName

        valList = list(tensorMemoryConstraint.memoryConstraints.values())
        constraintA = valList[0]
        constraintB = valList[1]

        # SCHEREMO : Base is whichever constraint is constant
        base = constraintA if isinstance(constraintA.size, int) else constraintB
        end = constraintA if base == constraintB else constraintB

        path = self.memoryHierarchy.bfs(base.memoryLevel, end.memoryLevel)
        requiredHops = path[1:]

        returnTensorConstraint = TensorMemoryConstraint(tensorName, {}, ctxt)
        returnTensorConstraint.addMemoryConstraint(base)

        for hop in requiredHops:
            factor = self.multiBufferStrategy(tilerModel, ctxt, pattern, path, hop, tensorName)
            assert factor >= 1 and isinstance(factor, int), "Invalid factor!"

            memConstraint = MemoryConstraint(hop, end.size)
            memConstraint.multiBufferCoefficient = factor
            returnTensorConstraint.addMemoryConstraint(memConstraint)

        return returnTensorConstraint

    def _generateIntermediateTilingSteps(self, tilerModel: TilerModel, ctxt: NetworkContext,
                                         sourceStep: NodeMemoryConstraint, destinationStep: NodeMemoryConstraint,
                                         pattern: SubGraph) -> NodeMemoryConstraint:
        tileConstraintStep = NodeMemoryConstraint()

        mergedStep = sourceStep + destinationStep
        tileTensorConstraints = [
            tensor for tensor in mergedStep.tensorMemoryConstraints.values()
            if len(tensor.memoryConstraints.values()) > 1
        ]

        for tileTensor in tileTensorConstraints:
            tiledTensor = self._generateTilePath(tilerModel, ctxt, tileTensor, pattern)
            ioDir = mergedStep.getIO(tileTensor.tensorName)
            tileConstraintStep.addTensorConstraint(tiledTensor, ioDir)

        return tileConstraintStep

    def _generateTilePathConstraints(self, tilerModel: TilerModel, ctxt: NetworkContext,
                                     sourceConstraints: List[PatternMemoryConstraints],
                                     destinationConstraints: List[PatternMemoryConstraints],
                                     schedule: List[SubGraph]) -> List[PatternMemoryConstraints]:

        tileConstraints = []

        for idx, (sourceConstraint, destinationConstraint) in enumerate(zip(sourceConstraints, destinationConstraints)):

            tilerModel.copyIdx = idx

            assert (len(sourceConstraint.nodeConstraints) == 1
                   ), "source pattern must be constant and single step, since it's live throughout the pattern!"
            sourcePatternStep = sourceConstraint.nodeConstraints[0]

            tileConstraint = PatternMemoryConstraints()

            for destinationConstraintStep in destinationConstraint.nodeConstraints:
                tileConstraintStep = self._generateIntermediateTilingSteps(tilerModel, ctxt, sourcePatternStep,
                                                                           destinationConstraintStep, schedule[idx])
                tileConstraint.addConstraint(tileConstraintStep)

            propagatedTileConstraint = self.propagateIOBufferStrategy(tileConstraint, schedule[idx], ctxt)
            assert len(propagatedTileConstraint.nodeConstraints) == len(tileConstraint.nodeConstraints)

            tileConstraints.append(propagatedTileConstraint)

        return tileConstraints

    def _generateBufferConstraints(self, ctxt: NetworkContext) -> NodeMemoryConstraint:

        constantGlobalConstraint: NodeMemoryConstraint = NodeMemoryConstraint()
        constantGlobalBuffers = [
            node for node in ctxt.globalObjects.values() if isinstance(node, ConstantBuffer) and node._deploy == True
        ]

        for constantBuffer in constantGlobalBuffers:

            tensorName = constantBuffer.name

            memorySize = int(np.prod(ctxt.lookup(tensorName).shape))

            elementMemorySize = memorySize
            memoryConstraint = MemoryConstraint(constantBuffer._memoryLevel, elementMemorySize)
            tensorConstraint = TensorMemoryConstraint(constantBuffer.name,
                                                      {memoryConstraint.memoryLevel: memoryConstraint}, ctxt)
            constantGlobalConstraint.addTensorConstraint(tensorConstraint, "input")

        return constantGlobalConstraint

    def _generateVariableBufferConstraints(
        self, tilerModel: TilerModel, ctxt: NetworkContext, schedule: List[SubGraph],
        layerBinding: OrderedDict[str, ONNXLayer], targetMemoryLevelMapping: TargetMemoryLevelMapping
    ) -> Tuple[List[PatternMemoryConstraints], List[PatternMemoryConstraints]]:

        def deltaFlow(
                patternFlow: List[GenericFlowState[TensorMemLevelTuple]]) -> GenericFlowState[TensorMemLevelTuple]:

            initialFlow = patternFlow[0]
            endFlow = patternFlow[1]

            # SCHEREMO: The genset and killset of the innerflow are correct; however, since we now pass the initialliveset of the pattern to the constraint flow. we need to remove bypassed tensors
            mergedLiveSet = initialFlow.liveSet - endFlow.liveSet
            mergedGenSet = initialFlow.genSet
            mergedKillSet = initialFlow.killSet

            mergedFlow = GenericFlowState[TensorMemLevelTuple](mergedLiveSet, mergedKillSet, mergedGenSet)

            return mergedFlow

        initialLiveBuffers = {
            value.name
            for value in ctxt.globalObjects.values()
            if (isinstance(value, ctxt.VariableBuffer) and value._users != [])
        }

        producedBuffers = {layer.node.outputs[0].name for layer in layerBinding.values()}
        inputBufferNames = initialLiveBuffers - producedBuffers
        inputBuffers = [ctxt.lookup(name) for name in inputBufferNames]

        initialLiveTensors = {TensorMemLevelTuple(buf.name, buf._memoryLevel) for buf in inputBuffers}

        constraintFlow = GraphMemoryConstraintFlow(ctxt, targetMemoryLevelMapping)
        graphFlowStates = constraintFlow.flow(schedule, initialLiveTensors)

        innerMemConstraints: List[PatternMemoryConstraints] = []
        outerMemConstraints: List[PatternMemoryConstraints] = []

        for idx, pattern in enumerate(schedule):

            tilerModel.copyIdx = idx

            innerPatternMemoryConstraints = PatternMemoryConstraints()
            outerPatternMemoryConstraints = PatternMemoryConstraints()

            outerFlowState = graphFlowStates[idx]
            patternFlow = constraintFlow._patternFlowStates[idx]

            dynamicOuterBufferConstraints = convertFlowState2NodeMemoryConstraint(tilerModel,
                                                                                  ctxt,
                                                                                  outerFlowState,
                                                                                  useMax = True)

            outerPatternMemoryConstraints.addConstraint(dynamicOuterBufferConstraints)
            outerMemConstraints.append(outerPatternMemoryConstraints)

            mergedFlow = [deltaFlow(patternFlow)]

            for step, innerFlowState in zip(pattern, mergedFlow):
                transientBufferConstraints = self._generatePatternStepTransientBufferConstraints(
                    tilerModel, ctxt, layerBinding, step, targetMemoryLevelMapping)

                dynamicInnerBufferConstraints = convertFlowState2NodeMemoryConstraint(tilerModel,
                                                                                      ctxt,
                                                                                      innerFlowState,
                                                                                      useMax = False)

                innerPatternMemoryConstraints.addConstraint(transientBufferConstraints + dynamicInnerBufferConstraints)

            innerMemConstraints.append(innerPatternMemoryConstraints)

        return outerMemConstraints, innerMemConstraints

    def _generatePatternStepTransientBufferConstraints(
            self, tilerModel: TilerModel, ctxt: NetworkContext, layerBinding: OrderedDict[str, ONNXLayer],
            step: gs.Node, targetMemoryLevelMapping: TargetMemoryLevelMapping) -> NodeMemoryConstraint:

        patternStepTransientBufferSizes = NodeMemoryConstraint()

        template = layerBinding[step.name].mapper.binder.template

        symbolicNodeRep = template.tileConstraint.constructSymbolicNodeRep(
            tilerModel, parseDict = layerBinding[step.name].mapper.parser.operatorRepresentation, ctxt = ctxt)

        transientBufferList: List[Tuple[str,
                                        Union[int,
                                              IntVar]]] = template.computeTransientBuffersSize(ctxt, symbolicNodeRep)

        for tensorName, memorySize in transientBufferList:

            # SCHEREMO: Assume transientbuffers end up in the same level as their user's main input
            memoryLevelName = targetMemoryLevelMapping.lookup(step.name, step.inputs[0].name)
            ctxt.lookup(tensorName)._memoryLevel = memoryLevelName

            transientSize = tilerModel.addTransientBufferSizeToModel(tensorName, memorySize)

            #memoryLevelName = self.memoryHierarchy.getDefaultMemoryLevel().name

            transientMemoryConstraint = MemoryConstraint(memoryLevelName, transientSize)
            transientBufferConstraint = TensorMemoryConstraint(tensorName, {memoryLevelName: transientMemoryConstraint},
                                                               ctxt)
            patternStepTransientBufferSizes.addTensorConstraint(transientBufferConstraint, "intermediate")

        return patternStepTransientBufferSizes

    def assertLayerWiseTiling(self, schedule: List[List[gs.Node]]) -> bool:
        for pattern in schedule:
            if len(pattern) > 1:
                return False

        return True

    def assertUniformMemoryLevelAllocation(self, ctxt: NetworkContext, defaultMemoryLevel: str) -> bool:
        for buffer in ctxt.localObjects.values():
            if buffer._memoryLevel != defaultMemoryLevel:
                return False
        return True


class TilerDeployerWrapper(NetworkDeployerWrapper):

    def __init__(self, deployer: Union[MemoryLevelAwareDeployer, MemoryDeployerWrapper], tilerCls: Type[Tiler] = Tiler):
        super().__init__(deployer)
        assert isinstance(self.Platform, (MemoryPlatform, MemoryPlatformWrapper)), \
            f"Platform should be a MemoryPlatform or MemoryPlatformWrapper! Got {type(self.Platform).__name__}"
        self.tiler = tilerCls(self.Platform.memoryHierarchy)

    @property
    def worstCaseBufferSize(self):
        maxAddr: Dict[str, int] = self.tiler.worstCaseBufferSize

        # WIESEP: Memory map form tiler does not include inputs and outputs
        for node in (self.inputs() + self.outputs()):
            maxAddr[node._memoryLevel] += np.prod(node.shape) * node._type.referencedType.typeWidth // 8

        return maxAddr

    def tile(self, tilingSolution: Optional[TilingSolution] = None):
        if tilingSolution is None:
            schedule = self.scheduler(self.graph)

            # PR-TODO: Rename this!
            outputBufferList = {
                "inputs": [node.name for node in self.graph.inputs],
                "outputs": [node.name for node in self.graph.outputs],
            }

            # JUNGVI: Currently using MiniMalloc is only supported for layer-wise execution and all tensors in the default memory level.
            if self.tiler.memoryAllocStrategy == "MiniMalloc":
                assert self.tiler.assertLayerWiseTiling(schedule), "Using MiniMalloc and DFT is not supported!"
                assert self.tiler.assertUniformMemoryLevelAllocation(
                    self.ctxt, self.Platform.memoryHierarchy._defaultMemoryLevel.name
                ), "All tensors have to be in the default memory level when using MiniMalloc!"

            self.tiler.setupModel(ctxt = self.ctxt,
                                  schedule = schedule,
                                  layerBinding = self.layerBinding,
                                  targetMemoryLevelMapping = self.getTargetMemoryLevelMapping(),
                                  outputBufferList = outputBufferList)
            tilingSolution, memoryMap = self.tiler.computeTilingSchedule(self.ctxt)
            if self.tiler.visualizeMemoryAlloc:
                self.tiler.plotMemoryAlloc(memoryMap, self.ctxt, self.deeployStateDir,
                                           self.Platform.memoryHierarchy._defaultMemoryLevel)

        # SCHEREMO: Annotate execution block with solution
        for layer, pattern in zip(self.layerBinding.values(), tilingSolution):
            layer.mapper.binder.executionBlock.patternMemoryConstraint = pattern

        # SCHEREMO: Code generation STUB

    def bind(self):
        if not super().bind():
            return False

        self.tile()
        return True


def TilingReadyNodeBindings(nodeBindings: List[NodeBinding], tileConstraint: TileConstraint) -> List[NodeBinding]:
    '''
    Apply the TillingReadyNodeTemplate to the template of each NodeBinding.
    '''
    nodeBindingsCopy = copy.deepcopy(nodeBindings)  #.copy()
    for binding in nodeBindingsCopy:
        binding.template.tileConstraint = tileConstraint

    return nodeBindingsCopy

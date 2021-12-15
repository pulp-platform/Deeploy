# ----------------------------------------------------------------------
#
# File: MemoryScheduler.py
#
# Last edited: 06.10.2023
#
# Copyright (C) 2023, ETH Zurich and University of Bologna.
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

from __future__ import annotations

import random
from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from ortools.constraint_solver.pywrapcp import IntVar

from Deeploy.CommonExtensions.OptimizationPasses.TopologyOptimizationPasses.LoweringOptimizationPasses import \
    _permuteList
from Deeploy.DeeployTypes import ConstantBuffer, NetworkContext, TransientBuffer
from Deeploy.MemoryLevelExtension.MemoryLevels import MemoryHierarchy
from Deeploy.TilingExtension.MemoryConstraints import PatternMemoryConstraints, TensorMemoryConstraint
from Deeploy.TilingExtension.TilerModel import TilerModel


@dataclass
class MemoryBlock:
    name: str
    level: str
    _lifetime: Tuple[int, int]
    _addrSpace: Optional[Tuple[int, int]] = None

    @property
    def addrSpace(self) -> Optional[Tuple[int, int]]:
        return self._addrSpace

    @addrSpace.setter
    def addrSpace(self, addrSpace: Optional[Tuple[int, int]]):
        if addrSpace is None:
            self._addrSpace = None
            return

        assert addrSpace[0] < addrSpace[1], "Address space range needs to be ordered from lesser to greater!"
        self._addrSpace = addrSpace

    @property
    def lifetime(self) -> Tuple[int, int]:
        return self._lifetime

    @lifetime.setter
    def lifetime(self, lifetime: Tuple[int, int]):
        assert lifetime[0] <= lifetime[1], "Lifetime range needs to be ordered from lesser to greater!"
        self._lifetime = lifetime

    def __init__(self, name: str, level: str, lifetime: Tuple[int, int], addrSpace: Optional[Tuple[int, int]]):
        self.name = name
        self.level = level
        self.lifetime = lifetime

        if addrSpace is not None:
            self.addrSpace = addrSpace

    def collides(self, other: MemoryBlock) -> bool:
        assert (isinstance(other, MemoryBlock)), f"{other} is not a MemoryBlock!"

        if self.addrSpace is None or other.addrSpace is None:
            return False

        xCollision: bool = False
        yCollision: bool = False

        if self.lifetime[0] <= other.lifetime[1] and self.lifetime[1] >= other.lifetime[0]:
            xCollision = True

        if self.addrSpace[0] < other.addrSpace[1] and self.addrSpace[1] > other.addrSpace[0]:
            yCollision = True

        return (xCollision and yCollision)


class MemoryScheduler():
    _ROWSUMNAME = "rowSum"
    _COLSUMNAME = "colSum"
    _PERMUTATIONIDXNAME = "permutationIdx"
    _INTERMEDIATEADJPRODUCTNAME = "intermediateAdjProduct"
    _FINALADJPRODUCTNAME = "AdjProduct"
    _COSTVARIABLENAME = "H"
    _COSTPRODUCTNAME = "costProduct"

    byteAlignment = 4

    @staticmethod
    def overlap(lifetimeA: Tuple[int, int], lifetimeB: Tuple[int, int]) -> bool:
        overlap: bool = False
        overlap |= (lifetimeA[0] >= lifetimeB[0] and lifetimeA[0] <= lifetimeB[1])
        overlap |= (lifetimeB[0] >= lifetimeA[0] and lifetimeB[0] <= lifetimeA[1])

        return overlap

    def __init__(self, stringSuffix: str, tileScheduler: bool, seed: int = 19960801):
        self._stringSuffix = stringSuffix
        self.stringSuffix = ""
        self.tileScheduler = tileScheduler

        self.seed = seed
        self.memoryMap: Dict[str, List[List[MemoryBlock]]] = {}

        self._permutationState: Dict[str, Union[List[List[Union[IntVar]]], np.ndarray]] = {}

    def _addPermutationMatrix(self, tilerModel: TilerModel, numVars: int,
                              patternIdx: int) -> List[List[Union[IntVar, int]]]:

        permMat: List[List[Union[IntVar, int]]] = []

        for i in range(numVars):
            rowSumName = f"{self._ROWSUMNAME}_{i}" + self.stringSuffix
            jSum = tilerModel.addVariable(rowSumName, 0, 1, patternIdx)
            permMat.append([])
            for j in range(numVars):
                name = f"{self._PERMUTATIONIDXNAME}_{i}_{j}" + self.stringSuffix
                jVar = tilerModel.addVariable(name, 0, 1, patternIdx)
                permMat[i].append(jVar)
            tilerModel.addConstraint(tilerModel._model.SumEquality(permMat[i], jSum))
            tilerModel.addConstraint(jSum == 1)

        for i in range(numVars):
            colSumName = f"{self._COLSUMNAME}_{i}" + self.stringSuffix
            jSum = tilerModel.addVariable(colSumName, 0, 1, patternIdx)
            constraintVec = []
            for j in range(numVars):
                name = f"{self._PERMUTATIONIDXNAME}_{j}_{i}" + self.stringSuffix
                jVar = tilerModel.getVariable(name, patternIdx)
                constraintVec.append(jVar)
            tilerModel.addConstraint(tilerModel._model.SumEquality(constraintVec, jSum))
            tilerModel.addConstraint(jSum == 1)

        return permMat

    def _permuteMatrices(self, tilerModel: TilerModel, permutationMatrix: List[List[Union[IntVar, int]]],
                         adjacencyMatrix: List[List[int]], costVector: List[Union[int, IntVar]], patternIdx: int):

        def boolMatMulSingle(A, B, row, col, transposeB = False):

            constr = 0
            numVars = len(A)

            for j in range(numVars):
                if not transposeB:
                    constr += A[row][j] * B[j][col]
                else:
                    constr += A[row][j] * B[col][j]

            return constr

        def boolMatVecMulSingle(A, B, row):

            constr = 0
            numVars = len(B)

            for j in range(numVars):
                constr += A[row][j] * B[j]

            return constr

        permAdj_intermediate: List[List[Union[IntVar, int]]] = []
        permAdj: List[List[Union[IntVar, int]]] = []
        permCost: List[Union[IntVar, int]] = []

        numVars = len(costVector)

        for i in range(numVars):
            permAdj_intermediate.append([])
            for j in range(numVars):
                name = f"{self._INTERMEDIATEADJPRODUCTNAME}_{i}_{j}" + self.stringSuffix
                jVar = tilerModel.addVariable(name, 0, 1, patternIdx)
                constr = boolMatMulSingle(permutationMatrix, adjacencyMatrix, i, j, False)
                tilerModel.addConstraint(jVar == constr)
                permAdj_intermediate[i].append(jVar)

        for i in range(numVars):
            permAdj.append([])
            for j in range(numVars):
                name = f"{self._FINALADJPRODUCTNAME}_{i}_{j}" + self.stringSuffix
                jVar = tilerModel.addVariable(name, 0, 1, patternIdx)
                constr = boolMatMulSingle(permAdj_intermediate, permutationMatrix, i, j, True)
                tilerModel.addConstraint(jVar == constr)
                permAdj[i].append(jVar)

        costMax = 0
        for cost in costVector:
            if isinstance(cost, int):
                newCost = cost
            else:
                newCost = cost.Max()
            costMax = max(costMax, newCost)

        for j in range(numVars):
            name = f"{self._COSTPRODUCTNAME}_{j}" + self.stringSuffix
            jVar = tilerModel.addVariable(name, 0, costMax, patternIdx)
            constr = boolMatVecMulSingle(permutationMatrix, costVector, j)
            tilerModel.addConstraint(jVar == constr)
            permCost.append(jVar)

        return permAdj, permCost

    def _generateCost(self, tilerModel: TilerModel, adjMatrix: List[List[Union[int, IntVar]]],
                      costVector: List[Union[int, IntVar]], patternIdx: int):

        def maxVal(val) -> int:
            if isinstance(val, int):
                return val
            else:
                return val.Max()

        hVector = []
        numVars = len(costVector)

        name = f"{self._COSTVARIABLENAME}_0" + self.stringSuffix

        hVar = tilerModel.addVariable(name, 0, maxVal(costVector[0]), patternIdx)
        constr = hVar == costVector[0]
        tilerModel.addConstraint(constr)
        hVector.append(hVar)

        for i in range(1, numVars):
            name = f"{self._COSTVARIABLENAME}_{i}" + self.stringSuffix
            # SCHEREMO: Check for overlap here!
            hVar = tilerModel.addVariable(name, 0, maxVal(hVector[i - 1]) + maxVal(costVector[i]), patternIdx)
            prod = []
            for j in range(i):
                name = f"{self._COSTVARIABLENAME}_{i}_maxEntry_{j}" + self.stringSuffix
                pVar = tilerModel.addVariable(name, 0, maxVal(hVector[j]) + maxVal(costVector[i]), patternIdx)
                constr = (pVar == (adjMatrix[i][j] * hVector[j] + costVector[i]))
                tilerModel.addConstraint(constr)
                prod.append(pVar)
            tilerModel.addConstraint(tilerModel._model.MaxEquality(prod, hVar))
            hVector.append(hVar)

        name = "cost" + self.stringSuffix
        costMax = max([maxVal(entry) for entry in hVector])
        cost = tilerModel.addVariable(name, 0, costMax, patternIdx)
        tilerModel.addConstraint(tilerModel._model.MaxEquality(hVector, cost))

        return cost

    def _buildInterferenceGraph(self, lifetimeMap):

        interferenceGraph: Dict[str, List[str]] = {}
        for name, lifetime in lifetimeMap.items():
            neighbors: List[str] = []
            for neighborName, neighborLifetime in lifetimeMap.items():
                if neighborName == name:
                    continue

                if self.overlap(lifetime, neighborLifetime):
                    neighbors.append(neighborName)

            interferenceGraph[name] = neighbors

        return interferenceGraph

    def _calculateLifetimes(self, ctxt: NetworkContext, patternMemoryConstraint: PatternMemoryConstraints,
                            memoryLevel: str):

        def filterTensorMemoryConstraint(ctxt: NetworkContext, tensorMemoryConstraint: TensorMemoryConstraint) -> bool:

            if ctxt.lookup(tensorMemoryConstraint.tensorName)._deploy == False:
                return False

            for level in tensorMemoryConstraint.memoryConstraints.values():

                homeLevel = ctxt.lookup(tensorName)._memoryLevel

                if not level.memoryLevel == memoryLevel:
                    continue

                # SCHEREMO: Transient buffers are only considered by last-level schedulers
                if isinstance(ctxt.lookup(tensorMemoryConstraint.tensorName), TransientBuffer) and self.tileScheduler:
                    return True

                elif isinstance(ctxt.lookup(tensorMemoryConstraint.tensorName), TransientBuffer):
                    return False

                # SCHEREMO: The original level is only considered by "home-level" schedulers
                if level.memoryLevel == homeLevel and not self.tileScheduler:

                    # SCHEREMO: ConstantBuffers are assigned and allocated at compile time, Global Var Buffers are assigned at init time
                    if isinstance(ctxt.lookup(tensorMemoryConstraint.tensorName), ConstantBuffer) or ctxt.is_global(
                            tensorMemoryConstraint.tensorName):
                        return False
                    return True

                if level.memoryLevel != homeLevel and self.tileScheduler:
                    return True

            return False

        tensorMap = OrderedDict()
        tensorLifetimeMap: Dict[str, Tuple[int, int]] = dict()

        for stepIdx, nodeConstraint in enumerate(patternMemoryConstraint.nodeConstraints):
            for tensorName, tensorMemoryConstraint in nodeConstraint.tensorMemoryConstraints.items():

                if not filterTensorMemoryConstraint(ctxt, tensorMemoryConstraint):
                    continue

                if tensorName in tensorLifetimeMap.keys():
                    prevLifetime = tensorLifetimeMap[tensorName]
                    tensorLifetimeMap[tensorName] = tuple((prevLifetime[0], stepIdx))
                else:
                    tensorLifetimeMap[tensorName] = tuple((stepIdx, stepIdx))
                    tensorMap[tensorName] = tensorMemoryConstraint

        return tensorLifetimeMap, tensorMap

    def _buildAdjacencyMatrix(self, graph, tensorMap):
        numVars = len(graph)

        adjacencyMatrix = np.zeros((numVars, numVars), dtype = int)

        for node, neighbors in graph.items():
            nodeIdx = list(tensorMap.keys()).index(node)
            for neighbor in neighbors:
                adjacencyIdx = list(tensorMap.keys()).index(neighbor)
                adjacencyMatrix[nodeIdx, adjacencyIdx] = 1

        return adjacencyMatrix

    def _buildCostVector(self, ctxt, graph, tensorMap, memoryLevel):
        costVector: List[Union[int, IntVar]] = []
        numVars = len(graph)

        if numVars == 0:
            costVector.append(0)
            return costVector

        for node, neighbors in graph.items():

            constraints = tensorMap[node].memoryConstraints
            cost = 0

            for c in constraints.values():
                if c.memoryLevel == memoryLevel:

                    if not isinstance(ctxt.lookup(node), TransientBuffer):
                        typeWidth = max(1, ctxt.lookup(node)._type.referencedType.typeWidth // 8)
                    else:
                        typeWidth = 1

                    # SCHEREMO: Make sure each tile is word-aligned for better access performance
                    # and to comply with implicit PULP L3 tiling bugs
                    wordCost = (((c.size * typeWidth) + type(self).byteAlignment - 1) //
                                type(self).byteAlignment) * type(self).byteAlignment
                    cost = wordCost * c.multiBufferCoefficient

                    # SCHEREMO: In-place operator outputs are "costless" whenever their input is in the same pattern
                    if hasattr(ctxt.lookup(node), "_alias") and ctxt.lookup(node)._alias in neighbors:
                        cost = 0

            costVector.append(cost)

        return costVector

    def heuristicPermutation(self, adjacencyMatrix, costVector) -> List[int]:
        permutationList = list(range(len(costVector)))
        random.seed(self.seed)
        random.shuffle(permutationList)

        return permutationList

    def _stablePermutation(self, adjacencyMatrix, costVector, permutationList):

        if len(costVector) == 1:
            return adjacencyMatrix, costVector, np.ones_like(adjacencyMatrix)

        permutationMatrix = np.zeros_like(adjacencyMatrix)
        newCostVector = []

        for i in permutationList:
            newCostVector.append(costVector[i])

        for idx, i in enumerate(permutationList):
            permutationMatrix[idx, i] = 1

        newAdjacencyMatrix = permutationMatrix @ adjacencyMatrix @ np.transpose(permutationMatrix)

        return newAdjacencyMatrix, newCostVector, permutationMatrix

    # SCHEREMO: Set the end of the lifetime of in-place operator inputs to the lifetime of their outputs
    def _dealiasLifetimeMap(self, ctxt: NetworkContext,
                            tensorLifetimeMap: Dict[str, Tuple[int, int]]) -> Dict[str, Tuple[int, int]]:

        tensorLifetimeMap = tensorLifetimeMap.copy()

        if not self.tileScheduler:
            for key, lifetime in tensorLifetimeMap.items():
                alias = ctxt.dealiasBuffer(key)

                if alias == key:
                    continue

                if ctxt.is_global(alias):
                    tensorLifetime = (0, lifetime[1])
                    tensorLifetimeMap[key] = tensorLifetime
                    continue

                aliasLifetime = tensorLifetimeMap[alias]
                tensorLifetime = (aliasLifetime[0], max(aliasLifetime[1], lifetime[1]))
                tensorLifetimeMap[alias] = tensorLifetime

        return tensorLifetimeMap

    def _scheduleMemoryConstraints(self,
                                   tilerModel: TilerModel,
                                   ctxt: NetworkContext,
                                   allMemoryConstraints: List[PatternMemoryConstraints],
                                   memoryHierarchy: MemoryHierarchy,
                                   memoryLevel: str = "L1",
                                   optimizeSchedule: bool = False):

        if memoryLevel not in self.memoryMap:
            self.memoryMap[memoryLevel] = []

        for patternIdx, patternMemoryConstraint in enumerate(allMemoryConstraints):

            # SCHEREMO: Calculate lifetimes
            tensorLifetimeMap, tensorMap = self._calculateLifetimes(ctxt, patternMemoryConstraint, memoryLevel)

            tensorLifetimeMap = self._dealiasLifetimeMap(ctxt, tensorLifetimeMap)

            # SCHEREMO: Build interference graph
            graph = self._buildInterferenceGraph(tensorLifetimeMap)

            numVars = len(graph)

            # SCHEREMO: Build adjacency matrices for memoryLevel
            adjacencyMatrix = self._buildAdjacencyMatrix(graph, tensorMap)
            costVector = self._buildCostVector(ctxt, graph, tensorMap, memoryLevel)
            nameVector: List[str] = []

            blockList = []

            for node, neighbors in graph.items():
                nameVector.append(node)
                relativeLifeTime = tensorLifetimeMap[node]
                absoluteLifetime = (relativeLifeTime[0] + patternIdx, relativeLifeTime[1] + patternIdx)

                memBlock = MemoryBlock(node, memoryLevel, absoluteLifetime, None)
                blockList.append(memBlock)

            self.memoryMap[memoryLevel].append(blockList)

            # SCHEREMO: Build permutation matrix
            if optimizeSchedule:
                if numVars > 1:

                    permutationMatrix = self._addPermutationMatrix(tilerModel, numVars, patternIdx)
                    permAdj, permCost = self._permuteMatrices(tilerModel, permutationMatrix, adjacencyMatrix,
                                                              costVector, patternIdx)

                else:
                    permutationMatrix = np.ones((1,))
                    permAdj, permCost = adjacencyMatrix, costVector

            else:
                permutationList = self.heuristicPermutation(adjacencyMatrix, costVector)
                permAdj, permCost, permutationMatrix = self._stablePermutation(adjacencyMatrix, costVector,
                                                                               permutationList)

            self._permutationState[memoryLevel + f"_{patternIdx}"] = permutationMatrix

            cost = self._generateCost(tilerModel, permAdj, permCost, patternIdx)
            constr = cost < memoryHierarchy.memoryLevels[memoryLevel].size
            tilerModel.addConstraint(constr)

        return

    def scheduleMemoryConstraints(self,
                                  tilerModel: TilerModel,
                                  ctxt: NetworkContext,
                                  allMemoryConstraints: List[PatternMemoryConstraints],
                                  memoryHierarchy: MemoryHierarchy,
                                  memoryLevel: str = "L1",
                                  optimizeSchedule: bool = False):

        self.stringSuffix = self._stringSuffix + f"_{memoryLevel}"
        return self._scheduleMemoryConstraints(tilerModel, ctxt, allMemoryConstraints, memoryHierarchy, memoryLevel,
                                               optimizeSchedule)

    def getSymbolicCostName(self, patternIdx: int, memoryLevel: str) -> str:
        stringSuffix = self._stringSuffix + f"_{memoryLevel}"

        name = f"cost{stringSuffix}"
        return name

    def getCost(self, tilerModel, patternIdx: int, memoryLevel: str) -> int:

        stringSuffix = self._stringSuffix + f"_{memoryLevel}"

        collector = tilerModel._solveModel("max")
        numVars = len(self.memoryMap[memoryLevel][patternIdx])

        name = f"cost{stringSuffix}_copyIdx_{patternIdx}"
        symVar = tilerModel._variables[name]
        var = tilerModel._resolveVariable(symVar)
        cost = var

        return cost

    def getHVector(self, tilerModel, patternIdx: int, memoryLevel: str) -> np.ndarray:

        stringSuffix = self._stringSuffix + f"_{memoryLevel}"

        collector = tilerModel._solveModel("max")
        numVars = len(self.memoryMap[memoryLevel][patternIdx])

        hVec = np.zeros((numVars))

        for i in range(numVars):
            name = f"{self._COSTVARIABLENAME}_{i}{stringSuffix}_copyIdx_{patternIdx}"
            symVar = tilerModel._variables[name]
            var = tilerModel._resolveVariable(symVar)
            hVec[i] = var

        return hVec

    def getBlockVector(self, patternIdx: int, memoryLevel: str) -> List[MemoryBlock]:

        return self.memoryMap[memoryLevel][patternIdx]

    def getPMatrix(self, tilerModel, patternIdx: int, memoryLevel: str) -> np.ndarray:

        stringSuffix = self._stringSuffix + f"_{memoryLevel}"

        collector = tilerModel._solveModel("max")
        numVars = len(self.memoryMap[memoryLevel][patternIdx])

        permMat = np.zeros((numVars, numVars))

        for i in range(numVars):
            for j in range(numVars):
                name = f"{self._PERMUTATIONIDXNAME}_{i}_{j}{stringSuffix}_copyIdx_{patternIdx}"
                symVar = tilerModel._variables[name]
                var = tilerModel._resolveVariable(symVar)
                permMat[i, j] = var

        return permMat

    def annotateSolution(self, ctxt: NetworkContext, tilerModel: TilerModel):

        def permMatrix2permList(permMatrix: np.ndarray) -> List[int]:

            _permMatrix = []

            if len(permMatrix) == 0:
                return []

            if len(permMatrix) == 1:
                return [0]

            for i in range(permMatrix.shape[0]):
                rowVec = list(permMatrix[i])
                _permMatrix.append(rowVec)

            return [row.index(1) for row in _permMatrix]

        for memoryLevel, patternList in self.memoryMap.items():
            for patternIdx, pattern in enumerate(patternList):

                permutationMatrix = self._permutationState[memoryLevel + f"_{patternIdx}"]

                if not isinstance(permutationMatrix, np.ndarray):
                    _permutationMatrix = self.getPMatrix(tilerModel, patternIdx, memoryLevel)
                else:
                    _permutationMatrix = permutationMatrix

                permList = permMatrix2permList(_permutationMatrix)

                if pattern != [] and len(pattern) > 1:
                    permPattern = _permuteList(pattern, permList)
                else:
                    permPattern = pattern

                aliasedBlocks = []

                for blockIdx, memoryBlock in enumerate(permPattern):

                    blockNames = [block.name for block in permPattern]
                    _buffer = ctxt.lookup(memoryBlock.name)

                    alias = ctxt.dealiasBuffer(memoryBlock.name)

                    # SCHEREMO: If we're handling an active alias to a global buffer in their home memory level, we don't need to resolve addresses
                    if all([alias != memoryBlock.name, ctxt.is_global(alias), _buffer._memoryLevel == memoryLevel]):
                        continue

                    # SCHEREMO: Don't fully unroll aliases here - this is pattern-sensitive!
                    if hasattr(_buffer, "_alias") and _buffer._alias in blockNames:
                        _alias = ctxt.lookup(memoryBlock.name)._alias
                        aliasedBlocks.append((memoryBlock, _alias))
                        continue

                    upperIdx = blockIdx

                    upperEndVar = tilerModel.getVariable(
                        f"{self._COSTVARIABLENAME}_{upperIdx}{self._stringSuffix}_{memoryLevel}", patternIdx)
                    upperEnd = tilerModel._resolveVariable(upperEndVar)

                    maxAddr = 0
                    for idx, oldBlock in enumerate(permPattern):
                        if self.overlap(oldBlock.lifetime, memoryBlock.lifetime):
                            if oldBlock.addrSpace is not None:
                                maxAddr = max(maxAddr, oldBlock.addrSpace[1])

                    lowerEnd = maxAddr
                    memoryBlock.addrSpace = (lowerEnd, upperEnd)

                for block, alias in aliasedBlocks:
                    for refBlock in sorted(permPattern, key = lambda x: x.lifetime[0]):
                        if refBlock.name == alias:
                            block.addrSpace = refBlock.addrSpace
                            break

# ----------------------------------------------------------------------
#
# File: TilerModel.py
#
# Last edited: 25.06.2023
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

from dataclasses import dataclass
from pprint import pformat
from typing import Dict, List, Literal, Optional, Tuple, Union

import numpy as np
from ortools.constraint_solver.pywrapcp import IntExpr, IntVar, SolutionCollector, Solver

from Deeploy.DeeployTypes import NetworkContext, OperatorRepresentation
from Deeploy.MemoryLevelExtension.MemoryLevels import MemoryLevel

_COPYIDXSUFFIX = "_copyIdx_"
_SOLVERTIMEOUT = 60000


@dataclass
class AddConstraintStrategy:
    """Base class for strategies of adding constraints"""


@dataclass
class PerformanceHint(AddConstraintStrategy):
    """Constraints marked with PerformanceHint will be added only if the TilerModel is still valid after adding them.
       Constraints with higher priority will be tried first.
    """
    priority: int = 0


class TilerModel():

    def __init__(self, copyIdxSuffix: Optional[str] = None):

        self._model: Solver = Solver('CPSimple')
        self._objectives: List[Tuple[IntVar, bool]] = []
        self._constraints: List[IntExpr] = []
        self._memoryConstraints: List[Tuple[MemoryLevel, IntExpr]] = []
        self._performanceConstraints: List[Tuple[int, IntExpr]] = []
        self._performanceMemoryConstraints: List[Tuple[int, Tuple[MemoryLevel, IntExpr]]] = []
        self._variables: Dict[str, IntVar] = {}

        self.copyIdx: int = 0
        self._copyIdxSuffix: str = copyIdxSuffix if copyIdxSuffix is not None else _COPYIDXSUFFIX
        self._collector: Optional[SolutionCollector] = None

    def _resolveVariable(self, var) -> int:
        if isinstance(var, int):
            return var

        if self._collector is None:
            return 0

        return self._collector.Value(self._collector.SolutionCount() - 1, var)

    def _addVariable(self, name: str, lowerBound: int, upperBound: int) -> IntVar:

        assert name not in self._variables.keys(), \
            f"Error while adding {name} variable in {self}, variable already exists."

        self._variables[name] = self._model.IntVar(int(lowerBound), int(upperBound), name)
        return self._variables[name]

    def _getSuffix(self, copyIdx: Optional[int]) -> str:
        if copyIdx is not None:
            varName = f"{self._copyIdxSuffix}{copyIdx}"
        else:
            varName = f"{self._copyIdxSuffix}{self.copyIdx}"
        return varName

    def getNameCopyIdx(self, variableName: str) -> Tuple[str, int]:
        splitList = variableName.split(self._copyIdxSuffix)
        varName = splitList[0]
        copyIdx = splitList[1]

        return (varName, int(copyIdx))

    def existsCopyIdx(self, name: str, copyIdx: Optional[int] = None) -> bool:

        if copyIdx is None:
            _copyIdx = self.copyIdx
        else:
            _copyIdx = copyIdx

        varName1 = name + "_num_elements" + self._getSuffix(_copyIdx)
        varName2 = name + "_dim_0" + self._getSuffix(_copyIdx)
        ret = (varName1 in self._variables) or (varName2 in self._variables)
        return ret

    def addObjective(self, objective: IntVar, objectiveType: Union[Literal['maximize'], Literal['minimize']]):
        if objectiveType == 'maximize':
            self._objectives.append((objective, False))
        else:
            self._objectives.append((objective, True))

    def addConstraint(self,
                      constraintExpression: IntExpr,
                      memoryLevel: Optional[MemoryLevel] = None,
                      strategy: Optional[AddConstraintStrategy] = None):
        if isinstance(strategy, PerformanceHint):
            if memoryLevel is None:
                self._performanceConstraints.append((strategy.priority, constraintExpression))
            else:
                self._performanceMemoryConstraints.append((strategy.priority, (memoryLevel, constraintExpression)))
        else:
            if memoryLevel is None:
                self._constraints.append(constraintExpression)
            else:
                self._memoryConstraints.append((memoryLevel, constraintExpression))

    def addVariable(self, name: str, lowerBound: int, upperBound: int, copyIdx: Optional[int] = None) -> IntVar:

        varName = name + self._getSuffix(copyIdx)
        return self._addVariable(varName, lowerBound, upperBound)

    def getVariable(self, name: str, copyIdx: Optional[int] = None) -> IntVar:
        varName = name + self._getSuffix(copyIdx)
        return self._variables[varName]

    def getTensorDimVar(self, tensorName: str, dimIdx: int, copyIdx: Optional[int] = None):

        varName = f"{tensorName}_dim_{dimIdx}" + self._getSuffix(copyIdx)

        return self._variables[varName]

    def getTensorNumberOfEltVar(self, tensorName: str, copyIdx: Optional[int] = None):

        varName = f"{tensorName}_num_elements" + self._getSuffix(copyIdx)

        return self._variables[varName]

    def addTensorDimToModel(self, ctxt: NetworkContext, tensorName: str, copyIdx: Optional[int] = None):
        '''
        Add every dimensions of an unseen tensors in the given list as Integer Variable of the Model and the context.
        Namespace of added variables is: f"{tensor.name}_dim_{idx}".
        '''
        tensor = ctxt.lookup(tensorName)

        for idx, dim in enumerate(tensor.shape):

            varName = f"{tensor.name}_dim_{idx}" + self._getSuffix(copyIdx)

            if varName in self._variables:
                continue

            self._addVariable(name = varName, lowerBound = 1, upperBound = dim)

    def addTensorNumOfEltToModel(self, ctxt: NetworkContext, tensorName: str, copyIdx: Optional[int] = None):
        '''
        For each tensor in the given list, add a variable equal to the product of dimension variables of this tensor.
        Namespace of those new variables are f"{tensor.name}_num_elements".
        '''

        varNameNumElt = f"{tensorName}_num_elements" + self._getSuffix(copyIdx)
        if varNameNumElt in self._variables:
            return

        tensor = ctxt.lookup(tensorName)

        tensorDimProductExpr = 1

        for idx, _ in enumerate(tensor.shape):

            varNameIdx = f"{tensor.name}_dim_{idx}" + self._getSuffix(copyIdx)
            tensorDimProductExpr *= self._variables[varNameIdx]

        tensorDimProductVar = self._addVariable(name = varNameNumElt,
                                                lowerBound = 1,
                                                upperBound = np.prod(tensor.shape))

        self._model.Add(tensorDimProductVar == tensorDimProductExpr)

    def addTransientBufferSizeToModel(self, tensorName: str, memorySizeExpr: Union[IntExpr, IntVar, int]) -> IntVar:

        transientName = tensorName

        if isinstance(memorySizeExpr, int):
            lowerBound = memorySizeExpr
            upperBound = memorySizeExpr
        else:
            lowerBound = memorySizeExpr.Min()
            upperBound = memorySizeExpr.Max()

        transientSize = self._addVariable(name = transientName, lowerBound = lowerBound, upperBound = upperBound)
        self._model.Add(transientSize == memorySizeExpr)

        return transientSize

    def addMinTileSizeConstraint(self,
                                 operatorRepresentation: OperatorRepresentation,
                                 variableName: str,
                                 intvar: IntVar,
                                 modulo: int,
                                 prefix: str = "",
                                 strategy: Optional[AddConstraintStrategy] = None) -> IntVar:

        tileSizeVar = self.addVariable(prefix + operatorRepresentation["nodeName"] + f"_{variableName}" + "_tileSize",
                                       1, operatorRepresentation[variableName])
        mulVar = self.addVariable(prefix + operatorRepresentation["nodeName"] + f"_{variableName}" + "_mul", 1,
                                  operatorRepresentation[variableName])
        addVar = self.addVariable(prefix + operatorRepresentation["nodeName"] + f"_{variableName}" + "_add", 0,
                                  operatorRepresentation[variableName])
        self.addConstraint(addVar <= tileSizeVar, strategy = strategy)
        self.addConstraint(addVar >= (modulo * (tileSizeVar < intvar.Max())), strategy = strategy)
        self.addConstraint(operatorRepresentation[variableName] == mulVar * tileSizeVar + addVar, strategy = strategy)
        self.addConstraint(intvar == tileSizeVar, strategy = strategy)

        return addVar

    def addTileSizeDivisibleConstraint(self,
                                       operatorRepresentation: OperatorRepresentation,
                                       variableName: str,
                                       intvar: IntVar,
                                       modulo: int,
                                       prefix: str = "",
                                       strategy: Optional[AddConstraintStrategy] = None) -> IntVar:

        tileSizeVar = self.addVariable(prefix + operatorRepresentation["nodeName"] + f"_{variableName}" + "_tileSize",
                                       1, operatorRepresentation[variableName])
        mulVar = self.addVariable(prefix + operatorRepresentation["nodeName"] + f"_{variableName}" + "_mul", 1,
                                  operatorRepresentation[variableName])

        mulMulVar = self.addVariable(prefix + operatorRepresentation["nodeName"] + f"_{variableName}" + "_mulmul", 1,
                                     operatorRepresentation[variableName])

        addVar = self.addVariable(prefix + operatorRepresentation["nodeName"] + f"_{variableName}" + "_add", 0,
                                  operatorRepresentation[variableName])

        self.addConstraint(addVar <= tileSizeVar,
                           strategy = strategy)  # Reminder tile has to be smaller than the regular tile
        self.addConstraint(tileSizeVar == mulMulVar * modulo, strategy = strategy)
        self.addConstraint(operatorRepresentation[variableName] == mulVar * tileSizeVar + addVar, strategy = strategy)
        self.addConstraint(intvar == tileSizeVar, strategy = strategy)

        return addVar

    def debugConstraints(self) -> bool:

        offendingGeometricalConstraints: List[IntExpr] = []
        offendingMemoryConstraints: List[Tuple[MemoryLevel, IntVar, IntExpr]] = []

        for constraint in self._constraints:
            if self._model.CheckConstraint(constraint):
                self._model.Add(constraint)
                continue

            offendingGeometricalConstraints.append(constraint)

        if offendingGeometricalConstraints != []:

            errorMsg = [""]
            errorMsg += ["ERROR: Some geometrical constraints are infeasible. A minimal set is this one:"]
            errorMsg += [pformat(offendingGeometricalConstraints, indent = 2)]
            raise RuntimeError(("\n").join(errorMsg))

        self.copyIdx = 0

        for idx, (memoryLevel, constraint) in enumerate(self._memoryConstraints):
            constrExpr = constraint <= memoryLevel.size

            if self._model.CheckConstraint(constrExpr):
                self._model.Add(constrExpr)
                continue

            self.copyIdx += 1

            offendingConstraint = self.addVariable(name = f"constraint", lowerBound = 0, upperBound = 2**63 - 1)
            self._model.Add(offendingConstraint == constraint)

            offendingMemoryConstraints.append((memoryLevel, offendingConstraint, constraint))

        collector = self._solveModel('min')

        minimumRequirement: Dict[str, int] = {}
        for memLevel, memRequirement, _ in offendingMemoryConstraints:
            value = collector.Value(collector.SolutionCount() - 1, memRequirement)

            if memLevel.name in minimumRequirement.keys():
                minimumRequirement[memLevel.name] = max(minimumRequirement[memLevel.name], value)
            else:
                minimumRequirement[memLevel.name] = value

        errorMsg = [""]

        for key, val in minimumRequirement.items():
            levelError = ""
            levelError += f"ERROR: minimal memory requirement violated, please increase {key} to at least {val} or change constraints"
            errorMsg.append(levelError)

        errorMsg.append(f"Offending constraints were")
        for memLevel, _, constr in offendingMemoryConstraints:
            errorMsg.append(f"{memLevel.size} >= {str(constr)}")

        if len(errorMsg) > 1:
            raise RuntimeError(("\n").join(errorMsg))

        return True

    def _trySetupConstraints(self,) -> bool:
        for constraint in self._constraints:
            self._model.Add(constraint)

        for memLevel, constraint in self._memoryConstraints:
            constrExpr = constraint <= memLevel.size
            self._model.Add(constrExpr)

        for _, performanceConstraint in sorted(self._performanceConstraints, reverse = True):
            if self._model.CheckConstraint(performanceConstraint):
                self._model.Add(performanceConstraint)

        for _, (memLevel, performanceConstraint) in sorted(self._performanceMemoryConstraints, reverse = True):
            constrExpr = performanceConstraint <= memLevel.size
            if self._model.CheckConstraint(constrExpr):
                self._model.Add(constrExpr)

        return self._model.CheckConstraint(self._model.TrueConstraint())

    def _setupObjective(self, patternIdx: Optional[int] = None):

        _patternIdx: int

        if patternIdx is None:
            _patternIdx = 0
        else:
            _patternIdx = patternIdx

        assert _patternIdx <= len(
            self._objectives), f"patternIdx {_patternIdx} is larger than list of _objectives, {len(self._objectives)}"

        _objective = self._objectives[_patternIdx]

        if _objective[1] == False:
            objective = self._model.Maximize(_objective[0], step = 1)
        else:
            objective = self._model.Minimize(_objective[0], step = 1)

        return objective

    def trySolveModel(self):

        solvable: bool = self._trySetupConstraints()
        if not solvable:
            self.debugConstraints()

        return self._solveModel()

    def _solveModel(self, solType: Union[Literal['min'], Literal['max']] = 'max') -> SolutionCollector:
        variablesList = [var for varName, var in self._variables.items()]

        if solType == 'max':
            decision_builder = self._model.Phase(variablesList, self._model.CHOOSE_FIRST_UNBOUND,
                                                 self._model.ASSIGN_MAX_VALUE)
        else:
            decision_builder = self._model.Phase(variablesList, self._model.CHOOSE_FIRST_UNBOUND,
                                                 self._model.ASSIGN_MIN_VALUE)

        collector = self._model.LastSolutionCollector()

        for var in variablesList:
            collector.Add(var)

        objective = self._setupObjective()

        timelimit = self._model.TimeLimit(_SOLVERTIMEOUT)

        log = self._model.SearchLog(1000000)

        _ = self._model.Solve(decision_builder, [objective, collector, log, timelimit])

        assert collector.SolutionCount() > 0, "Error in Tiler: No solution found"

        self._collector = collector
        return self._collector

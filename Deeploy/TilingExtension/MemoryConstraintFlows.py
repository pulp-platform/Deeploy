# ----------------------------------------------------------------------
#
# File: MemoryConstraintFlows.py
#
# Last edited: 01.08.2023
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

import copy
from collections import namedtuple
from typing import Iterable, List, Optional, Set, Tuple

import numpy as np
import onnx_graphsurgeon as gs

from Deeploy.DeeployTypes import ConstantBuffer, NetworkContext
from Deeploy.MemoryLevelExtension.NetworkDeployers.MemoryLevelDeployer import TargetMemoryLevelMapping
from Deeploy.TilingExtension.GenericFlow import GenericFlow, GenericFlowState
from Deeploy.TilingExtension.MemoryConstraints import MemoryConstraint, NodeMemoryConstraint, TensorMemoryConstraint
from Deeploy.TilingExtension.TilerModel import TilerModel

TensorMemLevelTuple = namedtuple("TensorMemLevelTuple", "tensorName targetMemoryLevel")


class PatternMemoryConstraintFlow(GenericFlow[TensorMemLevelTuple, gs.Node]):

    def __init__(self, ctxt: NetworkContext, pattern: List[gs.Node],
                 targetMemoryLevelMapping: TargetMemoryLevelMapping):
        super().__init__()
        self.ctxt = ctxt
        self.pattern = pattern
        self.patternNodeNames: Set[str] = {node.name for node in pattern}
        self.targetMemoryLevelMapping = targetMemoryLevelMapping

    def flow(self,
             iterator: Iterable[gs.Node],
             initialLiveSet: Optional[Set[TensorMemLevelTuple]] = None) -> List[GenericFlowState[TensorMemLevelTuple]]:

        if initialLiveSet is None:
            _initialLiveSet: Set[TensorMemLevelTuple] = self._getIOConstraints()[0]
        else:
            _initialLiveSet = initialLiveSet | self._getIOConstraints()[0]

        flowStates = super().flow(iterator, _initialLiveSet)
        return flowStates

    def _getInputTensorTuples(self, inputTensorNames: List[str]) -> Set[TensorMemLevelTuple]:

        inputTensorTuples: Set[TensorMemLevelTuple] = set()

        for tensorName in inputTensorNames:
            tensorUserSet = set(copy.deepcopy(self.ctxt.lookup(tensorName)._users))
            patternTensorUserSet = [node for node in self.pattern if node.name in tensorUserSet]

            for requiredUser in patternTensorUserSet:
                targetMemoryLevel = self.targetMemoryLevelMapping.lookup(requiredUser.name, tensorName)
                inputTensorTuples |= {TensorMemLevelTuple(tensorName, targetMemoryLevel)}

        return inputTensorTuples

    def _getOutputTensorTuples(self, outputTensorNames: List[str]) -> Set[TensorMemLevelTuple]:

        outputTensorTuples: Set[TensorMemLevelTuple] = set()

        for node in self.pattern:
            outputNames = [tensor.name for tensor in node.outputs]

            for tensorName in outputNames:
                if tensorName in copy.deepcopy(outputTensorNames):
                    targetMemoryLevel = self.targetMemoryLevelMapping.lookup(node.name, tensorName)
                    outputTensorTuples |= {TensorMemLevelTuple(tensorName, targetMemoryLevel)}

        return outputTensorTuples

    def _getIONames(self) -> Tuple[List[str], List[str]]:

        def _containsAll(listA: List, listB: List) -> bool:
            if not len(listB) < len(listA):
                return False

            return all([entry in listA for entry in listB])

        producedTensors = []
        inputTensors = []
        outputTensors = []

        patternNodeNames = [node.name for node in self.pattern]

        for node in self.pattern:
            inTensorNames = [node.name for node in node.inputs]
            outTensorNames = [node.name for node in node.outputs]

            for tensor in inTensorNames:
                if tensor not in producedTensors:
                    inputTensors.append(tensor)

            for tensor in outTensorNames:
                producedTensors.append(tensor)
                if not _containsAll(patternNodeNames, self.ctxt.lookup(tensor)._users):
                    outputTensors.append(tensor)

        return inputTensors, outputTensors

    def _getIOConstraints(self) -> Tuple[Set[TensorMemLevelTuple], Set[TensorMemLevelTuple]]:

        inputTensorNames, outputTensorNames = self._getIONames()

        patternInputConstraints = self._getInputTensorTuples(inputTensorNames)
        patternOutputConstraints = self._getOutputTensorTuples(outputTensorNames)

        return patternInputConstraints, patternOutputConstraints

    def computeGenSet(self, step: gs.Node) -> Set[TensorMemLevelTuple]:

        returnSet: Set[TensorMemLevelTuple] = set()
        for tensor in step.outputs:
            targetMemoryLevel = self.targetMemoryLevelMapping.lookup(step.name, tensor.name)
            returnSet.add(TensorMemLevelTuple(tensor.name, targetMemoryLevel))

        return returnSet

    def computeKillSet(self, step: gs.Node) -> Set[TensorMemLevelTuple]:

        returnSet: Set[TensorMemLevelTuple] = set()
        killTensorNames: List[str] = []

        _, outputTensorNames = self._getIONames()

        intermediateTensorNames = [tensor.name for tensor in step.inputs if tensor.name not in outputTensorNames]
        for tensorName in intermediateTensorNames:
            patternUsers = [node for node in self.ctxt.lookup(tensorName)._users if node in self.patternNodeNames]
            assert patternUsers != [], f"Tensor {tensorName} has no users in this pattern and is not an output!"
            if step.name == patternUsers[-1]:
                killTensorNames.append(tensorName)

        for tensorName in killTensorNames:
            targetMemoryLevel = self.targetMemoryLevelMapping.lookup(step.name, tensorName)
            returnSet.add(TensorMemLevelTuple(tensorName, targetMemoryLevel))

        return returnSet


class GraphMemoryConstraintFlow(GenericFlow[TensorMemLevelTuple, List[gs.Node]]):

    def __init__(self, ctxt: NetworkContext, targetMemoryLevelMapping: TargetMemoryLevelMapping):
        self.ctxt = ctxt
        self._patternFlowStates: List[List[GenericFlowState[TensorMemLevelTuple]]] = []
        self.targetMemoryLevelMapping = targetMemoryLevelMapping

    @property
    def patternFlowState(self):
        if not len(self._patternFlowStates) > 0:
            return None

        return self._patternFlowStates[-1]

    def preComputeStep(self, step: List[gs.Node]) -> None:

        constraintFlow = PatternMemoryConstraintFlow(self.ctxt, step, self.targetMemoryLevelMapping)
        if self.patternFlowState is not None:
            flowStates = constraintFlow.flow(step, self.patternFlowState[-1].liveSet)
        else:
            flowStates = constraintFlow.flow(step, None)

        self._patternFlowStates.append(flowStates)

    def computeGenSet(self, step: List[gs.Node]) -> Set[TensorMemLevelTuple]:

        genSet = set()
        outputConstraints = self.patternFlowState[-1].liveSet - self.patternFlowState[0].liveSet
        for constraint in outputConstraints:
            genSet.add(TensorMemLevelTuple(constraint.tensorName, self.ctxt.lookup(constraint.tensorName)._memoryLevel))

        return genSet

    def computeKillSet(self, step: List[gs.Node]) -> Set[TensorMemLevelTuple]:

        killSet = set()

        # SCHEREMO: pretty straightforward, just use the current patternFlow liveSet
        liveSet = [
            tensorTuple for tensorTuple in self.patternFlowState[0].liveSet
            if not isinstance(self.ctxt.lookup(tensorTuple.tensorName), ConstantBuffer)
        ]

        inputConstraints = liveSet

        patternNodeNames = [node.name for node in step]

        for constraint in inputConstraints:
            refBuffer = self.ctxt.lookup(constraint.tensorName)

            userList = refBuffer._users

            if len(userList) == 0:
                continue

            if userList[-1] in patternNodeNames and not isinstance(refBuffer, ConstantBuffer):

                killConstraint = TensorMemLevelTuple(constraint.tensorName,
                                                     self.ctxt.lookup(constraint.tensorName)._memoryLevel)
                killSet.add(killConstraint)

        return killSet


def convertFlowState2NodeMemoryConstraint(tilerModel: TilerModel,
                                          ctxt: NetworkContext,
                                          flowState: GenericFlowState[TensorMemLevelTuple],
                                          useMax: bool = False) -> NodeMemoryConstraint:

    nodeMemoryConstraint = NodeMemoryConstraint()

    memoryOccupyingSet = flowState.liveSet | flowState.genSet
    _inputs = [item.tensorName for item in flowState.liveSet]
    _outputs = [item.tensorName for item in flowState.genSet]

    for tensorName, memoryLevel in memoryOccupyingSet:

        if tilerModel.existsCopyIdx(tensorName):
            tilerModel.addTensorNumOfEltToModel(ctxt, tensorName)
            memorySize = tilerModel.getTensorNumberOfEltVar(tensorName)

            if useMax:
                _memorySize = memorySize.Max()
            else:
                _memorySize = memorySize

        else:
            # SCHEREMO: This means the tensor is passed through, we don't tile it
            _memorySize = int(np.prod(ctxt.lookup(tensorName).shape))

        elementMemorySize = _memorySize
        memLevelConstraint = MemoryConstraint(memoryLevel, elementMemorySize)
        tensorConstraint = TensorMemoryConstraint(tensorName, {memoryLevel: memLevelConstraint}, ctxt)

        if tensorName in _inputs:
            ioDir = "input"
        elif tensorName in _outputs:
            ioDir = "output"
        else:
            ioDir = "intermediate"

        nodeMemoryConstraint.addTensorConstraint(tensorConstraint, ioDir)

    return nodeMemoryConstraint

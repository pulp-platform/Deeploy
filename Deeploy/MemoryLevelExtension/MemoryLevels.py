# ----------------------------------------------------------------------
#
# File: MemoryLevel.py
#
# Last edited: 04.05.2023
#
# Copyright (C) 2023, ETH Zurich and University of Bologna.
#
# Author: Victor Jung, ETH Zurich
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

from typing import Dict, List, Optional, Sequence, Tuple

import onnx_graphsurgeon as gs

from Deeploy.DeeployTypes import CodeTransformation, NetworkContext, NodeBinding, NodeTemplate, NodeTypeChecker, \
    OperatorRepresentation


class MemoryLevel():

    def __init__(self, name: str, neighbourNames: List[str], size: int = 0):
        self.name = name
        self.neighbourNames = neighbourNames
        self.size = size  # By convention the size is in Bytes

        if self.size < 0:
            raise ValueError(
                f'Error while assigning a Memory Size to {self.name} Memory Level: Memory Size cannot be negative')

        if self.name in self.neighbourNames:
            raise ValueError(f'Node {self.name} cannot be a neighbour of itself')

    def __eq__(self, other):

        ret = [neighbour_name in other.neighbourNames for neighbour_name in self.neighbourNames]
        ret += [neighbour_name in self.neighbourNames for neighbour_name in other.neighbourNames]
        ret += [self.name == other.name, self.size == other.size]
        return all(ret)


class MemoryHierarchy():

    def __init__(self, node_list: List[MemoryLevel]):
        '''Effectively build the MemoryHierarchy from a list of MemoryLevels and check the validity of the hierarchy'''
        self.memoryLevels: Dict[str, MemoryLevel] = {}
        self._defaultMemoryLevel: Optional[MemoryLevel] = None

        for node in node_list:
            self._add(node)

        self._check()

    def __eq__(self, other):
        if not isinstance(other, MemoryHierarchy):
            return False

        if not other.memoryLevels.keys() == self.memoryLevels.keys():
            return False

        for memory_level_name in self.memoryLevels.keys():
            if not self.memoryLevels[memory_level_name] == other.memoryLevels[memory_level_name]:
                return False

        return True

    def _add(self, new_node: MemoryLevel):
        '''Add a new node to the hierarchy and propagate the neighbour relationships'''
        if new_node.name in self.memoryLevels.keys():
            raise ValueError(f'Node {new_node.name} already exists in MemoryHierarchy')

        self.memoryLevels[new_node.name] = new_node

    def _check(self):
        '''Check if the memory hierarchy is a valid undirected graph (i.e. every node point at a valid neighbour)'''
        for node_name, node in self.memoryLevels.items():
            violatingNodes = [
                neighbourName for neighbourName in node.neighbourNames if neighbourName not in self.memoryLevels.keys()
            ]
            assert len(violatingNodes) == 0, \
                f'Invalid Memory Hierarchy graph, node {node.name} point to non-existing neighbour(s) {violatingNodes}'

    def bfs(self, start: str, target: str) -> List[str]:

        visited = [start]

        queue = [[start]]
        queueIdx = 0

        if start == target:
            return queue[0]

        while queueIdx < len(queue):
            currentPath = queue[queueIdx]
            neighbours = self.memoryLevels[currentPath[-1]].neighbourNames

            if target in neighbours:
                currentPath.append(target)
                return currentPath

            for nextNode in neighbours:
                if nextNode not in visited:
                    newPath = currentPath[:]
                    newPath.append(nextNode)
                    queue.append(newPath)
                    visited.append(nextNode)
            queueIdx += 1

        return []

    def setDefaultMemoryLevel(self, name: str):
        assert (name in self.memoryLevels), f"Node {name} not in MemoryHierarchy"
        self._defaultMemoryLevel = self.memoryLevels[name]

    def getDefaultMemoryLevel(self):
        if self._defaultMemoryLevel is None:
            raise ValueError('defaultMemoryLevel level not set!')
        return self._defaultMemoryLevel


class NodeMemoryLevelChecker():

    def __init__(self, inputMemoryLevels: Sequence[Optional[str]], outputMemoryLevels: Sequence[Optional[str]]):
        self.inputMemoryLevels = inputMemoryLevels
        self.outputMemoryLevels = outputMemoryLevels

    def _memEq(self, memoryLevel: str, annotatedMemoryLevel: str) -> bool:
        if memoryLevel is None:
            return True
        else:
            return memoryLevel == annotatedMemoryLevel

    def _checkMemoryLevels(self, ctxt: NetworkContext, memoryLevels: Sequence[str],
                           tensors: Sequence[gs.Tensor]) -> bool:
        buffers = [ctxt.lookup(tensor.name) for tensor in tensors]
        if not all(hasattr(buffer, "_memoryLevel") for buffer in buffers):
            return False

        annotatedMemoryLevels = [buffer._memoryLevel for buffer in buffers]
        if all(
                self._memEq(memoryLevel, annotatedMemoryLevel)
                for memoryLevel, annotatedMemoryLevel in zip(memoryLevels, annotatedMemoryLevels)):
            return True
        else:
            return False

    def check(self, ctxt: NetworkContext, node: gs.Node, operatorRepresentation) -> Tuple[NetworkContext, bool]:
        if self._checkMemoryLevels(ctxt, self.inputMemoryLevels, node.inputs) and self._checkMemoryLevels(
                ctxt, self.outputMemoryLevels, node.outputs):
            return ctxt, True
        else:
            return ctxt, False


class MemoryAwareNodeBinding(NodeBinding):

    def __init__(self, typeChecker: NodeTypeChecker, memoryLevelChecker: NodeMemoryLevelChecker, template: NodeTemplate,
                 codeTransformer: CodeTransformation):
        super().__init__(typeChecker, template, codeTransformer)
        self.memoryLevelChecker = memoryLevelChecker

    def typeCheck(self, ctxt: NetworkContext, node: gs.Node,
                  operatorRepresentation: OperatorRepresentation) -> Tuple[NetworkContext, bool]:
        newCtxt, ret = self.memoryLevelChecker.check(ctxt, node, operatorRepresentation)
        if ret:
            return super().typeCheck(newCtxt, node, operatorRepresentation)

        return ctxt, False


def memoryAwareNodeBindingExtension(binding: NodeBinding,
                                    memoryLevelChecker: NodeMemoryLevelChecker) -> MemoryAwareNodeBinding:
    return MemoryAwareNodeBinding(binding.typeChecker, memoryLevelChecker, binding.template, binding.codeTransformer)

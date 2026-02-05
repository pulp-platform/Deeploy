# SPDX-FileCopyrightText: 2023 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, List, Optional


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

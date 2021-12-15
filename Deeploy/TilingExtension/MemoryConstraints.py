# ----------------------------------------------------------------------
#
# File: MemoryConstraints.py
#
# Last edited: 27.07.2023
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

import copy
from collections import OrderedDict
from typing import Dict, List, Literal, Optional, Tuple, Union

from ortools.constraint_solver.pywrapcp import IntVar

from Deeploy.DeeployTypes import NetworkContext


class MemoryConstraint():
    __slots__ = ["memoryLevel", "size", "multiBufferCoefficient", "shape", "addrSpace"]

    def __init__(self, memoryLevel: str, size: Union[IntVar, int]):
        self.memoryLevel: str = memoryLevel
        self.size: Union[int, IntVar] = size
        self.multiBufferCoefficient: Union[int, IntVar] = 1

        self.shape: Optional[Tuple[int]] = None
        self.addrSpace: Optional[Tuple[int, int]] = None

    def __repr__(self) -> str:
        retStr = f"MemoryLevel: {self.memoryLevel}, Size: {self.size}, MultiBuffer: {self.multiBufferCoefficient}"

        if self.shape is not None:
            retStr += f" Shape: {self.shape}"

        if self.addrSpace is not None:
            retStr += f" Address Space: {self.addrSpace}"

        return retStr

    def __deepcopy__(self, memo = {}):
        new = self.__class__(self.memoryLevel, self.size)
        new.multiBufferCoefficient = self.multiBufferCoefficient
        new.shape = self.shape
        new.addrSpace = self.addrSpace
        memo[id(self)] = new
        return new


class TensorMemoryConstraint():
    __slots__ = ["tensorName", "memoryConstraints"]

    def __init__(self, tensorName: str, constraints: Dict[str, MemoryConstraint], ctxt: NetworkContext):
        # SCHEREMO: Asserts the tensor is registered in the context
        _ = ctxt.lookup(tensorName)
        self.tensorName: str = tensorName
        self.memoryConstraints: OrderedDict[str, MemoryConstraint] = copy.deepcopy(
            constraints)  # Lists are mutable, so copy for persistence

    def _amendMemoryConstraints(self, memoryConstraints: Dict[str, MemoryConstraint]):

        _cleanConstraints = []
        for key, new in memoryConstraints.items():

            if not key in self.memoryConstraints.keys():
                _cleanConstraints.append(new)
                continue

            old = self.memoryConstraints[key]

            if old.memoryLevel == new.memoryLevel:
                assert old.size == new.size, "Tried to override existing constraints!"
                continue

            _cleanConstraints.append(new)

        for constraint in _cleanConstraints:
            self.addMemoryConstraint(constraint)

    def addMemoryConstraint(self, memoryConstraint: MemoryConstraint):
        name = memoryConstraint.memoryLevel
        self.memoryConstraints[name] = memoryConstraint

    def __repr__(self) -> str:
        retStr = f"{self.tensorName}: "
        retStr += "{\n"
        for i in self.memoryConstraints.values():
            line = str(i)
            retLines = line.split("\n")
            retLine = ""
            for line in retLines:
                retLine += ("\t" + line + "\n")
            retStr += retLine
        retStr += "}"
        return retStr


class NodeMemoryConstraint():
    __slots__ = ["inputTensorMemoryConstraints", "intermediateTensorMemoryConstraints", "outputTensorMemoryConstraints"]

    def __init__(self):
        self.inputTensorMemoryConstraints: Dict[str, TensorMemoryConstraint] = {}
        self.intermediateTensorMemoryConstraints: Dict[str, TensorMemoryConstraint] = {}
        self.outputTensorMemoryConstraints: Dict[str, TensorMemoryConstraint] = {}

    @property
    def tensorMemoryConstraints(self):
        return {
            **self.inputTensorMemoryConstraints,
            **self.intermediateTensorMemoryConstraints,
            **self.outputTensorMemoryConstraints
        }

    def _amendTensorConstraint(self, tensorMemoryConstraint: TensorMemoryConstraint):
        name = tensorMemoryConstraint.tensorName
        if name in self.tensorMemoryConstraints.keys():
            self.tensorMemoryConstraints[name]._amendMemoryConstraints(tensorMemoryConstraint.memoryConstraints)

    def getIO(self, tensorName: str) -> Optional[Literal["input", "intermediate", "output"]]:
        if tensorName in self.inputTensorMemoryConstraints.keys():
            return "input"
        elif tensorName in self.outputTensorMemoryConstraints.keys():
            return "output"
        elif tensorName in self.intermediateTensorMemoryConstraints.keys():
            return "intermediate"
        else:
            return None

    def addTensorConstraint(self, tensorMemoryConstraint: TensorMemoryConstraint, io: Literal["input", "output",
                                                                                              "intermediate"]):
        name = tensorMemoryConstraint.tensorName
        if name in self.tensorMemoryConstraints.keys():
            self._amendTensorConstraint(tensorMemoryConstraint)
            return

        if io == "input":
            _dict = self.inputTensorMemoryConstraints
        elif io == "output":
            _dict = self.outputTensorMemoryConstraints
        else:
            _dict = self.intermediateTensorMemoryConstraints

        _dict[name] = tensorMemoryConstraint

    def __add__(self, other):
        assert isinstance(other, NodeMemoryConstraint), f"Can't add {other} to {self}, expected NodeMemoryConstraint!"

        new = NodeMemoryConstraint()
        new.inputTensorMemoryConstraints = copy.deepcopy(self.inputTensorMemoryConstraints)
        new.intermediateTensorMemoryConstraints = copy.deepcopy(self.intermediateTensorMemoryConstraints)
        new.outputTensorMemoryConstraints = copy.deepcopy(self.outputTensorMemoryConstraints)

        for key, constraint in other.tensorMemoryConstraints.items():
            ioDir = other.getIO(key)
            new.addTensorConstraint(constraint, ioDir)

        return new

    def __repr__(self) -> str:
        retStr = ""
        retStr += "{\n"
        for i in self.tensorMemoryConstraints.values():
            line = str(i)
            retLines = line.split("\n")
            retLine = ""
            for line in retLines:
                retLine += ("\t" + line + "\n")
            retStr += retLine
        retStr += "}"
        return retStr


class PatternMemoryConstraints():
    __slots__ = ["nodeConstraints"]

    def __init__(self):
        self.nodeConstraints: List[NodeMemoryConstraint] = []

    def addConstraint(self, nodeConstraint: NodeMemoryConstraint):
        self.nodeConstraints.append(nodeConstraint)

    def __add__(self, other):

        assert isinstance(other,
                          PatternMemoryConstraints), f"Can't add {other} to {self}, expected PatternMemoryConstraints!"

        newConst = PatternMemoryConstraints()
        for old, new in zip(self.nodeConstraints, other.nodeConstraints):
            newConst.addConstraint(old + new)
        return newConst

    def __repr__(self) -> str:
        retStr = ""
        retStr += "{\n"
        for i in self.nodeConstraints:
            line = str(i)
            retLines = line.split("\n")
            retLine = ""
            for line in retLines:
                retLine += ("\t" + line + "\n")
            retStr += retLine
        retStr += "}"
        return retStr

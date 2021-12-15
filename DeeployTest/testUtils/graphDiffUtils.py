# ----------------------------------------------------------------------
#
# File: graphDiffUtils.py
#
# Last edited: 26.07.2024
#
# Copyright (C) 2024, ETH Zurich and University of Bologna.
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

from typing import Callable, Generator, Generic, List, Optional, Sequence, TypeVar

import numpy as np
import onnx_graphsurgeon as gs

T = TypeVar("T")


class DiffTreeNode(Generic[T]):

    _MESSAGE_INDENTATION_INCREMENT = 2

    def __init__(self, _object: T, other: T, name: str, children: List[DiffTreeNode]):
        self._object = _object
        self.other = other
        self.name = name
        self.children = children

    def messages(self, indentation: int = 0) -> List[str]:
        msg = " " * indentation + self.name
        if len(self.children) == 0:
            msg += f": {self._object} vs {self.other}"

        ret = [msg]
        for child in self.children:
            ret += child.messages(indentation + self._MESSAGE_INDENTATION_INCREMENT)
        return ret

    def __repr__(self):
        return self.name


class DiffTree():

    def __init__(self, root: Optional[DiffTreeNode]):
        self.root = root

    @property
    def message(self) -> str:
        if self.root is not None:
            return ("\n").join(self.root.messages())
        return ""

    def __iter__(self):

        def dfs(node: Optional[DiffTreeNode[T]]) -> Generator[DiffTreeNode[T], None, None]:
            if node is None:
                return
            yield node
            for child in node.children:
                yield from dfs(child)

        return dfs(self.root)


def createParentDiffNode(_object: T, other: T, name: str,
                         children: Sequence[Optional[DiffTreeNode]]) -> Optional[DiffTreeNode[T]]:
    """Return a node if children list has nodes, otherwise None"""
    filteredChildren = [child for child in children if child is not None]
    if len(filteredChildren) > 0:
        return DiffTreeNode(_object, other, name, filteredChildren)
    return None


def _attrDiff(instance: T,
              other: T,
              attr: str,
              eqFun: Callable[[T, T], bool] = lambda x, y: x == y) -> Optional[DiffTreeNode]:
    if not eqFun(getattr(instance, attr), getattr(other, attr)):
        return DiffTreeNode(getattr(instance, attr), getattr(other, attr), attr, [])
    return None


def _variableDiff(variable: gs.Variable, other: gs.Variable) -> Optional[DiffTreeNode]:
    diffs = []
    for attr in ["dtype", "shape"]:
        diffs.append(_attrDiff(variable, other, attr))
    return createParentDiffNode(variable, other, variable.name, diffs)


def _constantDiff(constant: gs.Constant, other: gs.Constant) -> Optional[DiffTreeNode]:
    diffs = []
    diffs.append(_attrDiff(constant, other, "data_location"))
    diffs.append(_attrDiff(constant, other, "values", np.array_equal))
    return createParentDiffNode(constant, other, constant.name, diffs)


def tensorDiff(tensor: gs.Tensor, other: gs.Tensor) -> Optional[DiffTreeNode]:
    if isinstance(tensor, gs.Variable) and isinstance(other, gs.Variable):
        return _variableDiff(tensor, other)

    if isinstance(tensor, gs.Constant) and isinstance(other, gs.Constant):
        return _constantDiff(tensor, other)

    assert isinstance(tensor, gs.Variable) or isinstance(tensor, gs.Constant)

    return DiffTreeNode(tensor, other, tensor.name, [DiffTreeNode(type(tensor), type(other), "type", [])])


def listDiff(_list: Sequence[T], other: Sequence[T], name: str,
             diffFun: Callable[[T, T], Optional[DiffTreeNode]]) -> Optional[DiffTreeNode]:
    if len(_list) != len(other):
        diffs = [DiffTreeNode(len(_list), len(other), "length", [])]
    else:
        diffs = [diffFun(item, item_other) for item, item_other in zip(_list, other)]
    return createParentDiffNode(_list, other, name, diffs)


def nodeDiff(node: gs.Node, other: gs.Node) -> Optional[DiffTreeNode]:
    diffs = []
    attrs = set(list(node.attrs.keys()) + list(other.attrs.keys()))
    for attr in attrs:
        if attr not in node.attrs:
            diffs.append(DiffTreeNode(None, other.attrs[attr], attr, []))
        elif attr not in other.attrs:
            diffs.append(DiffTreeNode(node.attrs[attr], None, attr, []))
        elif node.attrs[attr] != other.attrs[attr]:
            diffs.append(DiffTreeNode(node.attrs[attr], other.attrs[attr], attr, []))
    diffs.append(listDiff(node.inputs, other.inputs, "inputs", tensorDiff))
    diffs.append(listDiff(node.outputs, other.outputs, "outputs", tensorDiff))
    return createParentDiffNode(node, other, node.name, diffs)

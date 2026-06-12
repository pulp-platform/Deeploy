# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, List, Tuple, Union

from ortools.constraint_solver.pywrapcp import IntVar

from Deeploy.DeeployTypes import NetworkContext, SubGraph, TransientBuffer
from Deeploy.TilingExtension.MemoryConstraints import PatternMemoryConstraints
from Deeploy.TilingExtension.MemoryScheduler import MemoryScheduler
from Deeploy.TilingExtension.TilerExtension import Tiler
from Deeploy.TilingExtension.TilerModel import TilerModel


class DBOnlyL3Tiler(Tiler):

    def multiBufferStrategy(self, tilerModel: TilerModel, ctxt: NetworkContext, pattern: SubGraph, path: List[str],
                            hop: str, tensorName: str) -> Union[int, IntVar]:
        buffer = ctxt.lookup(tensorName)

        if isinstance(buffer, TransientBuffer):
            return 1

        if hop == 'L1':
            return 1

        return 2


class DBTiler(Tiler):

    def multiBufferStrategy(self, tilerModel: TilerModel, ctxt: NetworkContext, pattern: SubGraph, path: List[str],
                            hop: str, tensorName: str) -> Union[int, IntVar]:
        buffer = ctxt.lookup(tensorName)

        if isinstance(buffer, TransientBuffer):
            return 1

        return 2


class SBTiler(Tiler):

    def multiBufferStrategy(self, tilerModel: TilerModel, ctxt: NetworkContext, pattern: SubGraph, path: List[str],
                            hop: str, tensorName: str) -> Union[int, IntVar]:
        return 1


class TrainingMemoryScheduler(MemoryScheduler):
    """MemoryScheduler variant for training networks.

    Extends input tensor lifetimes to the end of the full tiling schedule so
    that forward-pass inputs remain live during the backward pass.
    """

    def _calculateLifetimes(self, ctxt: NetworkContext, patternMemoryConstraint: PatternMemoryConstraints,
                            memoryLevel: str) -> Tuple[Dict[str, Tuple[int, int]], Dict]:
        tensorLifetimeMap, tensorMap = super()._calculateLifetimes(ctxt, patternMemoryConstraint, memoryLevel)

        maxStepIdx = len(patternMemoryConstraint.nodeConstraints)
        for tensorName, lifetime in tensorLifetimeMap.items():
            buffer = ctxt.lookup(tensorName)
            if buffer.is_input:
                tensorLifetimeMap[tensorName] = (0, maxStepIdx)

        return tensorLifetimeMap, tensorMap


class TrainingSBTiler(SBTiler):
    memorySchedulerClass = TrainingMemoryScheduler

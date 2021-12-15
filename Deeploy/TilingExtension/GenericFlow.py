# ----------------------------------------------------------------------
#
# File: Flow.py
#
# Last edited: 28.07.2023
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

from abc import abstractmethod
from dataclasses import dataclass
from typing import Generic, Iterable, List, Optional, Set, TypeVar

flowType = TypeVar("flowType")
iteratorType = TypeVar("iteratorType")


@dataclass
class GenericFlowState(Generic[flowType]):
    liveSet: Set[flowType]
    killSet: Set[flowType]
    genSet: Set[flowType]

    def __repr__(self) -> str:
        retStr = ""
        retStr += "\nliveSet:\n"
        retStr += str(self.liveSet)
        retStr += "\nkillSet:\n"
        retStr += str(self.killSet)
        retStr += "\ngenSet:\n"
        retStr += str(self.genSet)
        return retStr


# SCHEREMO: Checkout data flow analysis (https://en.wikipedia.org/wiki/Data-flow_analysis)
class GenericFlow(Generic[flowType, iteratorType]):

    def flowStep(self, liveSet: Set[flowType], killSet: Set[flowType], genSet: Set[flowType]) -> Set[flowType]:

        # SCHEREMO: Assert general flow invariants
        assert (genSet & killSet) == set(
        ), f"ERROR: Spawning and killing {flowType} instance in same step: \ngenSet = {genSet}\n killSet = {killSet}"
        assert (genSet & liveSet) == set(
        ), f"ERROR: Spawning an already live {flowType} instance : \ngenSet = {genSet}\n liveSet = {liveSet}"
        assert (killSet - liveSet) == set(
        ), f"ERROR: Killing a non-live {flowType} instance,  \nkillSet = {killSet}, \nliveSet = {liveSet}"

        liveSet = (liveSet | genSet) - killSet

        return liveSet

    def flow(self,
             iterator: Iterable[iteratorType],
             initialLiveSet: Optional[Set[flowType]] = None) -> List[GenericFlowState[flowType]]:

        flowStates: List[GenericFlowState[flowType]] = []

        liveSet: Set[flowType] = set()
        if initialLiveSet is not None:
            liveSet = initialLiveSet

        killSet: Set[flowType]
        genSet: Set[flowType]

        for step in iterator:

            self.preComputeStep(step)

            genSet = self.computeGenSet(step)
            killSet = self.computeKillSet(step)

            flowStates.append(GenericFlowState[flowType](liveSet, killSet, genSet))

            liveSet = self.flowStep(liveSet, killSet, genSet)

        flowStates.append(GenericFlowState[flowType](liveSet, set(), set()))

        return flowStates

    def preComputeStep(self, step: iteratorType) -> None:
        pass

    @abstractmethod
    def computeGenSet(self, step: iteratorType) -> Set[flowType]:
        pass

    @abstractmethod
    def computeKillSet(self, step: iteratorType) -> Set[flowType]:
        pass

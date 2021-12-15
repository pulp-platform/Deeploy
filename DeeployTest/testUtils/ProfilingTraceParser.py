# ----------------------------------------------------------------------
#
# File: ProfilingTraceParser.py
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

import re
from dataclasses import dataclass
from typing import Dict, List, Literal, get_args

BufferingMode = Literal["SB", "DB"]


@dataclass
class LayerProfiling:
    bufferingMode: BufferingMode
    ops: int
    kernelCycles: List[int]
    inputDmaCycles: List[int]
    outputDmaCycles: List[int]


class ProfilingTraceParser:

    lineRegex = re.compile(
        r"\[(\w+)\]\[(SB|DB)\]\[(\d+) ops\]\[Tile \d+\] (Input DMA|Output DMA|Kernel) took (\d+) cycles\n")

    def parse(self, trace: str) -> Dict[str, LayerProfiling]:
        layerProfilings: Dict[str, LayerProfiling] = {}
        for match in ProfilingTraceParser.lineRegex.finditer(trace):
            layerName, bufferingMode, ops, measurementName, cycles = match.groups()

            if layerName not in layerProfilings:
                assert bufferingMode in get_args(BufferingMode), f"Unsupported bufferingMode {bufferingMode}"
                layerProfilings[layerName] = LayerProfiling(
                    bufferingMode = bufferingMode,  # type: ignore
                    ops = int(ops),
                    kernelCycles = [],
                    inputDmaCycles = [],
                    outputDmaCycles = [])

            if measurementName == "Kernel":
                layerProfilings[layerName].kernelCycles.append(int(cycles))
            elif measurementName == "Input DMA":
                layerProfilings[layerName].inputDmaCycles.append(int(cycles))
            elif measurementName == "Output DMA":
                layerProfilings[layerName].outputDmaCycles.append(int(cycles))
            else:
                raise RuntimeError(f"Unsupported measurement name: {measurementName}")

        return layerProfilings

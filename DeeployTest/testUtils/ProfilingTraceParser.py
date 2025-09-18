# SPDX-FileCopyrightText: 2024 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

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

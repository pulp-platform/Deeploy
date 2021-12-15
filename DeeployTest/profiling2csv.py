# ----------------------------------------------------------------------
#
# File: profiling2csv.py
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

import argparse
import csv
import dataclasses

from prettytable import PrettyTable
from testUtils.ProfilingTraceParser import LayerProfiling, ProfilingTraceParser


@dataclasses.dataclass
class LayerInfo:
    name: str
    bufferingMode: str
    ops: int
    totalKernelCycles: int
    totalInputDmaCycles: int
    totalOutputDmaCycles: int


def layerInfoFromProfiling(name: str, profiling: LayerProfiling) -> LayerInfo:
    return LayerInfo(name = name,
                     bufferingMode = profiling.bufferingMode,
                     ops = profiling.ops,
                     totalKernelCycles = sum(profiling.kernelCycles),
                     totalInputDmaCycles = sum(profiling.inputDmaCycles),
                     totalOutputDmaCycles = sum(profiling.outputDmaCycles))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = 'Parse and visualize profiling results')
    parser.add_argument('trace_path', type = str, help = 'Path to the profiling trace file')
    parser.add_argument('-o',
                        '--output_path',
                        type = str,
                        default = "profile.csv",
                        help = 'Path to the output CSV file')
    parser.add_argument('--table',
                        action = 'store_true',
                        default = False,
                        help = 'Print a table of the profiled results.')
    args = parser.parse_args()

    profilingParser = ProfilingTraceParser()

    with open(args.trace_path, "r") as f:
        layerProfilings = profilingParser.parse(f.read())

    fieldnames = [field.name for field in dataclasses.fields(LayerInfo)]
    layerInfos = [layerInfoFromProfiling(name, profiling) for name, profiling in layerProfilings.items()]

    with open(args.output_path, 'w', newline = '') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames = fieldnames)

        writer.writeheader()
        for info in layerInfos:
            writer.writerow(dataclasses.asdict(info))

    if args.table:
        table = PrettyTable(field_names = fieldnames)
        for info in layerInfos:
            table.add_row(dataclasses.astuple(info))
        print(table)

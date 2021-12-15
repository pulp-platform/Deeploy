# ----------------------------------------------------------------------
#
# File: SliceTemplate.py
#
# Last edited: 01.06.2023
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

from typing import Dict, List, Tuple

import numpy as np

from Deeploy.DeeployTypes import NetworkContext, NodeTemplate, OperatorRepresentation
from Deeploy.Targets.PULPOpen.DataTypes import PULPStructDataTypes


class _SliceTemplate(NodeTemplate):

    def __init__(self, templateStr):
        super().__init__(templateStr)

    def alignToContext(self, ctxt: NetworkContext,
                       operatorRepresentation: OperatorRepresentation) -> Tuple[NetworkContext, Dict, List[str]]:

        assert ctxt.lookup(operatorRepresentation['data_in'])._memoryLevel in ["L2",
                                                                               "L1"], "input data needs to be on-chip!"
        assert ctxt.lookup(operatorRepresentation['data_out'])._memoryLevel in ["L2", "L1"
                                                                               ], "output data needs to be on-chip!"
        assert ctxt.lookup(operatorRepresentation['data_in'])._memoryLevel != ctxt.lookup(
            operatorRepresentation['data_out'])._memoryLevel, "Can't move on same memory level with Cluster DMA!"

        bufferList = []

        def _downSample(starts, ends, axes, steps, data_in_shape, idx) -> bool:
            return steps[idx] != 1 or starts[idx] > 0 or ends[idx] < data_in_shape[axes[idx]]

        # Immediate-ify start
        startsBuffer = ctxt.lookup(operatorRepresentation['starts'])
        axesBuffer = ctxt.lookup(operatorRepresentation['axes'])
        endsBuffer = ctxt.lookup(operatorRepresentation['ends'])
        stepsBuffer = ctxt.lookup(operatorRepresentation['steps'])

        startsBuffer._deploy = False
        axesBuffer._deploy = False
        endsBuffer._deploy = False
        stepsBuffer._deploy = False

        operatorRepresentation['starts'] = startsBuffer.values
        operatorRepresentation['ends'] = endsBuffer.values
        operatorRepresentation['axes'] = axesBuffer.values
        operatorRepresentation['steps'] = stepsBuffer.values

        operatorRepresentation['data_in_size'] = np.prod(operatorRepresentation['data_in_shape'])

        data_in_shape = operatorRepresentation['data_in_shape']
        data_in_size = operatorRepresentation['data_in_size']
        axes = operatorRepresentation['axes']
        starts = operatorRepresentation['starts']
        ends = operatorRepresentation['ends']
        steps = operatorRepresentation['steps']

        dimSteps = []
        dimSteps.append(data_in_size // data_in_shape[0])
        for dim in data_in_shape[1:]:
            dimSteps.append(dimSteps[-1] // dim)

        number_of_1d_copies = 1
        number_of_2d_copies = 1
        stride_1d = 0
        stride_2d = 0

        numCopies = []
        strides = []
        downSample = []

        switchIdx = 0

        for i in range(len(axes)):
            numCopies.append(ends[i] - starts[i])
            strides.append(dimSteps[axes[i]])
            downSample.append(_downSample(starts, ends, axes, steps, data_in_shape, i))

        for idx, switch in enumerate(downSample):
            if switch == True:
                switchIdx = idx
                break
            switchIdx = axes[idx] + 1

        operatorRepresentation["offset"] = starts[switchIdx] * dimSteps[axes[switchIdx]]

        operatorRepresentation['numberIterations'] = np.prod(data_in_shape[:axes[switchIdx]])

        inputOffset = dimSteps[axes[switchIdx]] * data_in_shape[axes[switchIdx]]
        outputOffset = int(inputOffset * ((ends[switchIdx] - starts[switchIdx]) / data_in_shape[axes[switchIdx]]))
        consecutiveCopies = outputOffset
        transferSize1D = consecutiveCopies * operatorRepresentation['data_in_type'].referencedType.typeWidth // 8

        if ctxt.lookup(operatorRepresentation['data_in'])._memoryLevel == "L2":
            # Target address:
            ext = operatorRepresentation['data_in']
            # Source address:
            loc = operatorRepresentation['data_out']
            _dir = 1
            operatorRepresentation["extOffset"] = inputOffset
            operatorRepresentation["locOffset"] = outputOffset
        else:
            # Target address:
            ext = operatorRepresentation['data_out']
            # Source address:
            loc = operatorRepresentation['data_in']
            _dir = 0
            operatorRepresentation["locOffset"] = inputOffset
            operatorRepresentation["extOffset"] = outputOffset

        operatorRepresentation["dir"] = _dir

        length_2d_copy = number_of_1d_copies * transferSize1D
        mchan_flags = _dir + 0x2 + 0x8
        if number_of_1d_copies > 1 or number_of_2d_copies > 1:
            mchan_flags += 0x4
        mchan_cmd = length_2d_copy + (mchan_flags << 17)

        bufferList += [
            ctxt.hoistStruct(
                {
                    "ext": ext,
                    "loc": loc,
                    "hwc_to_chw": 0,
                    "stride_2d": stride_2d,
                    "number_of_2d_copies": number_of_2d_copies,
                    "stride_1d": stride_1d,
                    "number_of_1d_copies": number_of_1d_copies,
                    "length_1d_copy": transferSize1D,
                    "mchan_cmd": mchan_cmd,
                    "dir": _dir,
                    "tid": 0
                }, operatorRepresentation['nodeName'] + "_stateReference", PULPStructDataTypes.DMA_copy)
        ]

        operatorRepresentation['stateReference'] = operatorRepresentation['nodeName'] + "_stateReference"

        return ctxt, operatorRepresentation, bufferList


referenceTemplate = _SliceTemplate("""
// Slice (Name: ${nodeName}, Op: ${nodeOp})
// data_in : ${data_in_shape}
// data_out : ${data_out_shape}
% if dir == 1:
${stateReference}.ext += ${offset};
% else:
${stateReference}.loc += ${offset};
% endif
for(int j=0;j<${numberIterations};j++){
dory_dma_memcpy_async(&${stateReference});
${stateReference}.ext += ${extOffset};
${stateReference}.loc += ${locOffset};
}
""")

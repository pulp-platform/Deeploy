# ----------------------------------------------------------------------
#
# File: PadTemplate.py
#
# Last edited: 27.12.2021
#
# Copyright (C) 2021, ETH Zurich and University of Bologna.
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

from Deeploy.DeeployTypes import NetworkContext, NodeTemplate, OperatorRepresentation


class _Pad2DTemplate(NodeTemplate):

    def __init__(self, templateStr):
        super().__init__(templateStr)

    def alignToContext(self, ctxt: NetworkContext,
                       operatorRepresentation: OperatorRepresentation) -> Tuple[NetworkContext, Dict, List[str]]:

        # Align padding value to input signedness

        data_in = ctxt.lookup(operatorRepresentation['data_in'])
        if hasattr(data_in, "_signed") and hasattr(data_in, "nLevels") and not data_in._signed:
            operatorRepresentation['value'] = operatorRepresentation['value'] - int(data_in.nLevels / 2)

        return ctxt, operatorRepresentation, []


reference2DTemplate = _Pad2DTemplate("""
<%
    y_offset_out = dim_im_out_ch*(pad_y*dim_im_out_y)
    x_offset_out = dim_im_out_ch*(pad_x)
    width = dim_im_in_ch*dim_im_in_y

    addoffsetOut = dim_im_out_ch * dim_im_out_y
    addoffsetIn = dim_im_in_ch * dim_im_in_y

    startPosX = y_offset_out + x_offset_out

batchOffsetOut = dim_im_out_ch * dim_im_out_x * dim_im_out_y
%>

// 2D Pad (Name: ${nodeName}, Op: ${nodeOp})
BEGIN_SINGLE_CORE
    memset(${data_out}, ${value}, ${data_out_size}*sizeof(${data_out_type.referencedType.typeName}));
    uint32_t xoffset_${data_out}_${data_in};
    uint32_t offset_in_${data_out}_${data_in} = 0;

    % if channels_first:
    // NCHW Layout
    for(uint32_t n=0; n<${batch}; n++){
        xoffset_${data_out}_${data_in} = n*${batchOffsetOut} + ${pad_y}*${dim_im_out_y}+${pad_x};
        for (uint32_t c=0; c<${dim_im_in_ch}; ++c) {
            for(uint32_t h=0; h<${dim_im_in_x}; h++){
                memcpy(${data_out} + xoffset_${data_out}_${data_in}, ${data_in}+offset_in_${data_out}_${data_in}, ${dim_im_in_y}*sizeof(${data_out_type.referencedType.typeName}));
                xoffset_${data_out}_${data_in} += ${dim_im_out_y};
                offset_in_${data_out}_${data_in} += ${dim_im_in_y};
            }
            xoffset_${data_out}_${data_in} += 2*${pad_y}*${dim_im_out_y};
        }
    }
    % else:
    // NHWC Layout
    for(uint32_t n=0; n<${batch}; n++){
        xoffset_${data_out}_${data_in} = n*${batchOffsetOut} + ${startPosX};
        for(uint32_t h=0; h<${dim_im_in_x}; h++){
            memcpy(${data_out}+xoffset_${data_out}_${data_in}, ${data_in}+offset_in_${data_out}_${data_in}, ${width}*sizeof(${data_out_type.referencedType.typeName}));
            xoffset_${data_out}_${data_in} += ${addoffsetOut};
            offset_in_${data_out}_${data_in} += ${addoffsetIn};
        }
    }
    %endif
END_SINGLE_CORE
""")


class _Pad1DTemplate(NodeTemplate):

    def __init__(self, templateStr):
        super().__init__(templateStr)

    def alignToContext(self, ctxt: NetworkContext,
                       operatorRepresentation: OperatorRepresentation) -> Tuple[NetworkContext, Dict, List[str]]:

        # Align padding value to input signedness

        data_in = ctxt.lookup(operatorRepresentation['data_in'])
        if hasattr(data_in, "_signed") and hasattr(data_in, "nLevels") and not data_in._signed:
            operatorRepresentation['value'] = operatorRepresentation['value'] - int(data_in.nLevels / 2)

        return ctxt, operatorRepresentation, []


reference1DTemplate = _Pad1DTemplate("""
<%
    x_offset_out = dim_im_out_ch*(pad_y)
    width = dim_im_in_ch*dim_im_in_y

    startPosX = x_offset_out

batchOffsetOut = dim_im_out_ch * dim_im_out_y
%>

// 1D Pad (Name: ${nodeName}, Op: ${nodeOp})
BEGIN_SINGLE_CORE
    memset(${data_out}, ${value}, ${data_out_size}*sizeof(${data_out_type.referencedType.typeName}));
    uint32_t xoffset_${data_out}_${data_in};
    uint32_t offset_in_${data_out}_${data_in} = 0;

    % if channels_first:
    // NCHW Layout
    for(uint32_t n=0; n<${batch}; n++){
        xoffset_${data_out}_${data_in} = n*${batchOffsetOut} +${pad_y};
        for (uint32_t c=0; c<${dim_im_in_ch}; ++c) {
            memcpy(${data_out} + xoffset_${data_out}_${data_in}, ${data_in}+offset_in_${data_out}_${data_in}, ${dim_im_in_y}*sizeof(${data_out_type.referencedType.typeName}));
            xoffset_${data_out}_${data_in} += ${dim_im_out_y};
            offset_in_${data_out}_${data_in} += ${dim_im_in_y};
        }
    }
    % else:
    // NHWC Layout
    for(uint32_t n=0; n<${batch}; n++){
        xoffset_${data_out}_${data_in} = n*${batchOffsetOut} + ${startPosX};
        memcpy(${data_out}+xoffset_${data_out}_${data_in}, ${data_in}+offset_in_${data_out}_${data_in}, ${width}*sizeof(${data_out_type.referencedType.typeName}));
        offset_in_${data_out}_${data_in} += ${width};
    }
    %endif
END_SINGLE_CORE
""")

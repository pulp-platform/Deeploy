# ----------------------------------------------------------------------
#
# File: MaxPoolTemplate.py
#
# Last edited: 24.01.2025
#
# Copyright (C) 2023, ETH Zurich and University of Bologna.
#
# Author: Run Wang, ETH Zurich
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

from Deeploy.DeeployTypes import NodeTemplate

referenceTemplate = NodeTemplate("""
// 2D Float MaxPool Channel Parallel (Name: ${nodeName}, Op: ${nodeOp})

${data_in_type.typeName} ref_${data_out}_${data_in} = ${data_in};
${data_out_type.typeName} ref_${data_out}_${data_out} = ${data_out};

for (uint32_t n=0; n<${batch}; ++n) {
    PULP_MaxPool2d_fp${data_in_type.referencedType.typeWidth}_fp${data_out_type.referencedType.typeWidth}_HWC(
        ref_${data_out}_${data_in}, 
        ${dim_im_in_x}, ${dim_im_in_y}, ${ch_im_in},
        ${dim_kernel_x}, ${dim_kernel_y}, 
        ${stride_x}, ${stride_y},
        ref_${data_out}_${data_out},
        ${padding_y_top}, ${padding_y_bottom}, ${padding_x_left}, ${padding_x_right}
    );
    ref_${data_out}_${data_in} += ${ch_im_in}*${dim_im_in_x}*${dim_im_in_y};
    ref_${data_out}_${data_out} += ${ch_im_out}*${dim_im_out_x}*${dim_im_out_y};
}
""")
# SPDX-FileCopyrightText: 2023 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, List, Tuple

from Deeploy.DeeployTypes import NetworkContext, NodeTemplate, OperatorRepresentation


class _Conv2D_Template(NodeTemplate):

    def __init__(self, templateStr):
        super().__init__(templateStr)

    def alignToContext(self, ctxt: NetworkContext,
                       operatorRepresentation: OperatorRepresentation) -> Tuple[NetworkContext, Dict, List[str]]:

        data_in = ctxt.lookup(operatorRepresentation['data_in'])
        data_out = ctxt.lookup(operatorRepresentation['data_out'])

        operatorRepresentation['input_offset'] = 0
        if hasattr(data_in, "_signed") and hasattr(data_in, "nLevels"):
            operatorRepresentation['input_offset'] = (data_in._signed == 0) * int(data_in.nLevels // 2)
        operatorRepresentation['output_offset'] = 0
        if hasattr(data_out, "_signed") and hasattr(data_out, "nLevels"):
            operatorRepresentation['output_offset'] = -(data_out._signed == 0) * int(data_out.nLevels // 2)

        return ctxt, operatorRepresentation, []


reference1DTemplate = _Conv2D_Template("""
<%
batchOffsetIn = ch_im_in * dim_im_in_y
batchOffsetOut = ch_im_out * dim_im_out_y
%>

// 1D Conv (Name: ${nodeName}, Op: ${nodeOp})
BEGIN_SINGLE_CORE
    ${data_in_type.typeName} ref_${nodeName}_${data_in} = ${data_in};
    ${data_out_type.typeName} ref_${nodeName}_${data_out} = ${data_out};

    for (uint32_t n=0; n<${batch}; ++n) {
        Conv2d_s${data_in_type.referencedType.typeWidth}_s${weight_type.referencedType.typeWidth}_s${data_out_type.referencedType.typeWidth}_NCHW(
            ref_${nodeName}_${data_in}, ${ch_im_in}, 1, ${dim_im_in_y},
            ${weight}, ${ch_im_out}, 1, ${dim_kernel_y},
            1, ${stride_y},
            ref_${nodeName}_${data_out}, ${input_offset}, ${output_offset}
        );
        ref_${nodeName}_${data_in} += ${batchOffsetIn};
        ref_${nodeName}_${data_out} += ${batchOffsetOut};
    }
END_SINGLE_CORE
""")

reference2DTemplate = _Conv2D_Template("""
<%
batchOffsetIn = ch_im_in * dim_im_in_x * dim_im_in_y
batchOffsetOut = ch_im_out * dim_im_out_x * dim_im_out_y
%>

// 2D Conv (Name: ${nodeName}, Op: ${nodeOp})
BEGIN_SINGLE_CORE
    ${data_in_type.typeName} ref_${nodeName}_${data_in} = ${data_in};
    ${data_out_type.typeName} ref_${nodeName}_${data_out} = ${data_out};

    for (uint32_t n=0; n<${batch}; ++n) {
        Conv2d_s${data_in_type.referencedType.typeWidth}_s${weight_type.referencedType.typeWidth}_s${data_out_type.referencedType.typeWidth}_NCHW(
            ref_${nodeName}_${data_in}, ${ch_im_in}, ${dim_im_in_x}, ${dim_im_in_y},
            ${weight}, ${ch_im_out}, ${dim_kernel_x}, ${dim_kernel_y},
            ${stride_x}, ${stride_y},
            ref_${nodeName}_${data_out}, ${input_offset}, ${output_offset}
        );
        ref_${nodeName}_${data_in} += ${batchOffsetIn};
        ref_${nodeName}_${data_out} += ${batchOffsetOut};
    }
END_SINGLE_CORE
""")

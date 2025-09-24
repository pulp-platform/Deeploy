# SPDX-FileCopyrightText: 2023 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, List, Tuple, Union

from ortools.constraint_solver.pywrapcp import IntVar

from Deeploy.DeeployTypes import NetworkContext, NodeTemplate, OperatorRepresentation


class PULP2DFloatConvIm2ColTemplate(NodeTemplate):

    def __init__(self, templateStr):
        super().__init__(templateStr)

    @staticmethod
    def computeTransientBuffersSize(
            ctxt: NetworkContext,
            operatorRepresentation: OperatorRepresentation) -> List[Tuple[str, Union[int, IntVar]]]:
        # Memory allocation for the im2col buffer can be dynamic, based on the number of cores.
        im2col_dim = (operatorRepresentation["weight_type"].typeWidth //
                      8) * operatorRepresentation["n_cores"] * operatorRepresentation[
                          'ch_im_in'] * operatorRepresentation['dim_kernel_x'] * operatorRepresentation['dim_kernel_y']

        im2col_name = operatorRepresentation['nodeName'] + "_buffer"

        return [(im2col_name, im2col_dim)]

    def hoistTransientBuffers(self, ctxt: NetworkContext,
                              operatorRepresentation: OperatorRepresentation) -> Tuple[NetworkContext, Dict, List[str]]:
        im2col_name, im2col_dim = PULP2DFloatConvIm2ColTemplate.computeTransientBuffersSize(
            ctxt, operatorRepresentation)[0]
        ctxt.hoistTransientBuffer(im2col_name, im2col_dim)

        operatorRepresentation['ctxtBuffer'] = im2col_name
        operatorRepresentation['ctxtBufferSize'] = im2col_dim
        return ctxt, operatorRepresentation, [im2col_name]


class PULP2DFloatDWConvIm2ColTemplate(NodeTemplate):

    def __init__(self, templateStr):
        super().__init__(templateStr)

    @staticmethod
    def computeTransientBuffersSize(
            ctxt: NetworkContext,
            operatorRepresentation: OperatorRepresentation) -> List[Tuple[str, str]]:

        # Memory allocation for the im2col buffer can be dynamic, based on the number of cores.
        # WARNING: This works because value is only used as string, in the allocate template.
        im2col_dim = "NUM_CORES * " + str(
            operatorRepresentation['dim_kernel_x'] * operatorRepresentation['dim_kernel_y'])
        im2col_name = operatorRepresentation['nodeName'] + "_buffer"
        return [(im2col_name, im2col_dim)]

    def hoistTransientBuffers(self, ctxt: NetworkContext,
                              operatorRepresentation: OperatorRepresentation) -> Tuple[NetworkContext, Dict, List[str]]:
        im2col_name, im2col_dim = PULP2DFloatDWConvIm2ColTemplate.computeTransientBuffersSize(
            ctxt, operatorRepresentation)[0]
        ctxt.hoistTransientBuffer(im2col_name, im2col_dim)

        # Manually set the type of the im2col buffer to match the input type, since it defaults to void for transient buffers
        ctxt.lookup(im2col_name)._type.referencedType = ctxt.lookup(operatorRepresentation['data_in'])._type.referencedType

        operatorRepresentation['ctxtBuffer'] = im2col_name
        operatorRepresentation['ctxtBufferSize'] = im2col_dim
        return ctxt, operatorRepresentation, [im2col_name]


class PULP2DFloatDWConvIm2ColTemplate(NodeTemplate):

    def __init__(self, templateStr):
        super().__init__(templateStr)

    @staticmethod
    def computeTransientBuffersSize(
            ctxt: NetworkContext,
            operatorRepresentation: OperatorRepresentation) -> List[Tuple[str, Union[int, IntVar]]]:

        # Memory allocation for the im2col buffer can be dynamic, based on the number of cores.
        im2col_dim = (operatorRepresentation["weight_type"].typeWidth // 8) * operatorRepresentation[
            "n_cores"] * operatorRepresentation['dim_kernel_x'] * operatorRepresentation['dim_kernel_y']

        im2col_name = operatorRepresentation['nodeName'] + "_buffer"

        return [(im2col_name, im2col_dim)]

    def hoistTransientBuffers(self, ctxt: NetworkContext,
                              operatorRepresentation: OperatorRepresentation) -> Tuple[NetworkContext, Dict, List[str]]:
        im2col_name, im2col_dim = PULP2DFloatDWConvIm2ColTemplate.computeTransientBuffersSize(
            ctxt, operatorRepresentation)[0]
        ctxt.hoistTransientBuffer(im2col_name, im2col_dim)

        # Manually set the type of the im2col buffer to match the input type, since it defaults to void for transient buffers
        ctxt.lookup(im2col_name)._type.referencedType = ctxt.lookup(
            operatorRepresentation['data_in'])._type.referencedType

        operatorRepresentation['ctxtBuffer'] = im2col_name
        operatorRepresentation['ctxtBufferSize'] = im2col_dim
        return ctxt, operatorRepresentation, [im2col_name]


class PULP2DFloatDWConvIm2ColTemplate(NodeTemplate):

    def __init__(self, templateStr):
        super().__init__(templateStr)

    @staticmethod
    def computeTransientBuffersSize(
            ctxt: NetworkContext,
            operatorRepresentation: OperatorRepresentation) -> List[Tuple[str, Union[int, IntVar]]]:

        # Memory allocation for the im2col buffer can be dynamic, based on the number of cores.
        im2col_dim = (operatorRepresentation["weight_type"].typeWidth // 8) * operatorRepresentation[
            "n_cores"] * operatorRepresentation['dim_kernel_x'] * operatorRepresentation['dim_kernel_y']

        im2col_name = operatorRepresentation['nodeName'] + "_buffer"

        return [(im2col_name, im2col_dim)]

    def hoistTransientBuffers(self, ctxt: NetworkContext,
                              operatorRepresentation: OperatorRepresentation) -> Tuple[NetworkContext, Dict, List[str]]:
        im2col_name, im2col_dim = PULP2DFloatDWConvIm2ColTemplate.computeTransientBuffersSize(
            ctxt, operatorRepresentation)[0]
        ctxt.hoistTransientBuffer(im2col_name, im2col_dim)

        # Manually set the type of the im2col buffer to match the input type, since it defaults to void for transient buffers
        ctxt.lookup(im2col_name)._type.referencedType = ctxt.lookup(
            operatorRepresentation['data_in'])._type.referencedType

        operatorRepresentation['ctxtBuffer'] = im2col_name
        operatorRepresentation['ctxtBufferSize'] = im2col_dim
        return ctxt, operatorRepresentation, [im2col_name]


reference2DTemplate = NodeTemplate("""
// 2D FP Conv HWC with ChannelOut parallelism (Name: ${nodeName}, Op: ${nodeOp})

${data_in_type.typeName} ref_${data_out}_${data_in} = ${data_in};
${data_out_type.typeName} ref_${data_out}_${data_out} = ${data_out};

for (uint32_t n=0; n<${batch}; ++n) {
    PULP_Conv2d_fp${data_in_type.referencedType.typeWidth}_fp${weight_type.referencedType.typeWidth}_fp${data_out_type.referencedType.typeWidth}_HWC(
        ref_${data_out}_${data_in},
        ${dim_im_in_y}, ${dim_im_in_x}, ${ch_im_in},
        ${weight}, ${ch_im_out},
        ${dim_kernel_y}, ${dim_kernel_x},
        ${stride_y}, ${stride_x},
        ${bias}, ${has_bias},
        ref_${data_out}_${data_out},
        ${padding_y_top}, ${padding_y_bottom}, ${padding_x_left}, ${padding_x_right}
    );


    ref_${data_out}_${data_in} += ${ch_im_in} * ${dim_im_in_x} * ${dim_im_in_y};
    ref_${data_out}_${data_out} += ${ch_im_out} * ${dim_im_out_x} * ${dim_im_out_y};
}
""")

reference2DIm2ColTemplate = PULP2DFloatConvIm2ColTemplate("""
// 2D FP Conv HWC with Im2Col and ChannelOout parallelism (Name: ${nodeName}, Op: ${nodeOp})

${data_in_type.typeName} ref_${data_out}_${data_in} = ${data_in};
${data_out_type.typeName} ref_${data_out}_${data_out} = ${data_out};

for (uint32_t n=0; n<${batch}; ++n) {
    PULP_Conv2d_Im2Col_fp${data_in_type.referencedType.typeWidth}_fp${weight_type.referencedType.typeWidth}_fp${data_out_type.referencedType.typeWidth}_HWC(
        ref_${data_out}_${data_in},
        ${dim_im_in_x},
        ${dim_im_in_y},
        ${ch_im_in},
        ${weight},
        ${ch_im_out},
        ${dim_kernel_x},
        ${dim_kernel_y},
        ${stride_x},
        ${stride_y},
        ${bias}, ${has_bias},
        ref_${data_out}_${data_out},
        ${padding_y_top},
        ${padding_y_bottom},
        ${padding_x_left},
        ${padding_x_right},
        ${ctxtBuffer}
    );

    ref_${data_out}_${data_in} += ${ch_im_in} * ${dim_im_in_x} * ${dim_im_in_y};
    ref_${data_out}_${data_out} += ${ch_im_out} * ${dim_im_out_x} * ${dim_im_out_y};
}
""")

referenceDW2DIm2ColTemplate = PULP2DFloatDWConvIm2ColTemplate("""
// 2D DW FP Conv HWC with Im2Col and ChannelOout parallelism (Name: ${nodeName}, Op: ${nodeOp})

${data_in_type.typeName} ref_${data_out}_${data_in} = ${data_in};
${data_out_type.typeName} ref_${data_out}_${data_out} = ${data_out};

for (uint32_t n=0; n<${batch}; ++n) {
    PULP_DW_Conv2d_Im2Col_fp${data_in_type.referencedType.typeWidth}_fp${weight_type.referencedType.typeWidth}_fp${data_out_type.referencedType.typeWidth}_HWC(
        ref_${data_out}_${data_in},
        ${dim_im_in_x},
        ${dim_im_in_y},
        ${ch_im_in},
        ${weight},
        ${ch_im_out},
        ${dim_kernel_x},
        ${dim_kernel_y},
        ${stride_x},
        ${stride_y},
        ${bias}, ${has_bias},
        ref_${data_out}_${data_out},
        ${padding_y_top},
        ${padding_y_bottom},
        ${padding_x_left},
        ${padding_x_right},
        ${ctxtBuffer}
    );

    ref_${data_out}_${data_in} += ${ch_im_in} * ${dim_im_in_x} * ${dim_im_in_y};
    ref_${data_out}_${data_out} += ${ch_im_out} * ${dim_im_out_x} * ${dim_im_out_y};
}
""")

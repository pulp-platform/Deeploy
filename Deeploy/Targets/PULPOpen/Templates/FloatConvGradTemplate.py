# SPDX-FileCopyrightText: 2023 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, List, Tuple, Union

from ortools.constraint_solver.pywrapcp import IntVar

from Deeploy.DeeployTypes import NetworkContext, NodeTemplate, OperatorRepresentation

_TILE_IDX_NULL = "NULL"


class _ConvGradWTemplate(NodeTemplate):
    """NodeTemplate subclass for ConvGradW operators.

    Injects tileIdxPtr='NULL' sentinel via alignToContext so the template
    always has a defined tileIdxPtr value, avoiding Mako strict_undefined
    eager-initialization NameError.  The tiling pass overwrites 'NULL' with
    the real buffer name when multi-tile execution is required.
    """

    def alignToContext(self, ctxt: NetworkContext,
                       operatorRepresentation: OperatorRepresentation) -> Tuple[NetworkContext, Dict, List[str]]:
        if 'tileIdxPtr' not in operatorRepresentation:
            operatorRepresentation['tileIdxPtr'] = _TILE_IDX_NULL
        return ctxt, operatorRepresentation, []


class PULP2DFloatConvGradWIm2ColTemplate(_ConvGradWTemplate):

    @staticmethod
    def computeTransientBuffersSize(
            ctxt: NetworkContext,
            operatorRepresentation: OperatorRepresentation) -> List[Tuple[str, Union[int, IntVar]]]:
        # For ConvGradW, im2col buffer stores im2row transformed input
        # Size: H_out * W_out * kernel_h * kernel_w * C_in * sizeof(float)
        im2col_dim = (operatorRepresentation["data_in_type"].typeWidth // 8) * \
                     operatorRepresentation['dim_im_out_x'] * operatorRepresentation['dim_im_out_y'] * \
                     operatorRepresentation['ch_im_in'] * \
                     operatorRepresentation['dim_kernel_x'] * operatorRepresentation['dim_kernel_y']

        im2col_name = operatorRepresentation['nodeName'] + "_buffer"

        return [(im2col_name, im2col_dim)]

    def hoistTransientBuffers(self, ctxt: NetworkContext,
                              operatorRepresentation: OperatorRepresentation) -> Tuple[NetworkContext, Dict, List[str]]:
        im2col_name, im2col_dim = PULP2DFloatConvGradWIm2ColTemplate.computeTransientBuffersSize(
            ctxt, operatorRepresentation)[0]
        ctxt.hoistTransientBuffer(im2col_name, im2col_dim)

        operatorRepresentation['ctxtBuffer'] = im2col_name
        operatorRepresentation['ctxtBufferSize'] = im2col_dim
        return ctxt, operatorRepresentation, [im2col_name]


class PULP2DFloatConvGradXIm2ColTemplate(NodeTemplate):

    def __init__(self, templateStr):
        super().__init__(templateStr)

    @staticmethod
    def computeTransientBuffersSize(
            ctxt: NetworkContext,
            operatorRepresentation: OperatorRepresentation) -> List[Tuple[str, Union[int, IntVar]]]:
        im2col_dim = (operatorRepresentation["grad_in_type"].typeWidth // 8) * \
                     operatorRepresentation['dim_im_out_x'] * operatorRepresentation['dim_im_out_y'] * \
                     operatorRepresentation['ch_im_out'] * \
                     operatorRepresentation['dim_kernel_x'] * operatorRepresentation['dim_kernel_y']

        im2col_name = operatorRepresentation['nodeName'] + "_im2col_buffer"

        bt_dim = (operatorRepresentation["weight_type"].typeWidth // 8) * \
                 operatorRepresentation['ch_im_in'] * operatorRepresentation['ch_im_out'] * \
                 operatorRepresentation['dim_kernel_x'] * operatorRepresentation['dim_kernel_y']

        bt_name = operatorRepresentation['nodeName'] + "_bt_buffer"

        return [(im2col_name, im2col_dim), (bt_name, bt_dim)]

    def hoistTransientBuffers(self, ctxt: NetworkContext,
                              operatorRepresentation: OperatorRepresentation) -> Tuple[NetworkContext, Dict, List[str]]:
        buffers = PULP2DFloatConvGradXIm2ColTemplate.computeTransientBuffersSize(ctxt, operatorRepresentation)

        im2col_name, im2col_dim = buffers[0]
        bt_name, bt_dim = buffers[1]

        ctxt.hoistTransientBuffer(im2col_name, im2col_dim)
        ctxt.hoistTransientBuffer(bt_name, bt_dim)

        operatorRepresentation['ctxtBuffer'] = im2col_name
        operatorRepresentation['ctxtBufferSize'] = im2col_dim
        operatorRepresentation['btBuffer'] = bt_name
        operatorRepresentation['btBufferSize'] = bt_dim

        return ctxt, operatorRepresentation, [im2col_name, bt_name]


# Templates for ConvGradX operations
referenceConvGradX2DTemplate = NodeTemplate("""
// 2D FP ConvGradX (dX) NCHW trainlib naive (Name: ${nodeName}, Op: ${nodeOp})
${grad_out_type.typeName}  ref_${grad_out} = ${grad_out};   // dY
${weight_type.typeName}   ref_${weight}  = ${weight};    // W
${grad_in_type.typeName} ref_${grad_in}       = ${grad_in};  // dX

for (uint32_t n=0; n<${batch}; ++n) {
    PULP_ConvGradX2d_fp${grad_out_type.referencedType.typeWidth}_fp${weight_type.referencedType.typeWidth}_fp${grad_in_type.referencedType.typeWidth}_CHW_tiled(
        ref_${grad_out},
        ${dim_im_out_x}, ${dim_im_out_y}, ${ch_im_out},
        ref_${weight},
        ${ch_im_in},
        ${dim_kernel_x}, ${dim_kernel_y},
        ${stride_x}, ${stride_y},
        ref_${grad_in},
        ${dim_im_in_x}, ${dim_im_in_y},
        ${padding_y_top}, ${padding_y_bottom}, ${padding_x_left}, ${padding_x_right},
        ${offset_grad_in_h}, ${offset_grad_in_w},
        ${offset_grad_out_h}, ${offset_grad_out_w}

    );

    ref_${grad_out} += ${ch_im_out} * ${dim_im_out_y} * ${dim_im_out_x};
    ref_${grad_in}  += ${ch_im_in}  * ${dim_im_in_y}  * ${dim_im_in_x};
}
""")


referenceConvGradX2DIm2ColTiledTemplate = PULP2DFloatConvGradXIm2ColTemplate("""
// 2D FP ConvGradX (dX) NCHW/CHW using tile-aware Im2Col (Name: ${nodeName}, Op: ${nodeOp})
${grad_out_type.typeName}  ref_${grad_out} = ${grad_out};   // dY
${weight_type.typeName}   ref_${weight}  = ${weight};    // W
${grad_in_type.typeName} ref_${grad_in}       = ${grad_in};  // dX
for (uint32_t n=0; n<${batch}; ++n) {
    PULP_ConvGradX2d_fp${grad_out_type.referencedType.typeWidth}_fp${weight_type.referencedType.typeWidth}_fp${grad_in_type.referencedType.typeWidth}_CHW_Im2Col_tiled(
        ref_${grad_out},                                   
        ${dim_im_out_x}, ${dim_im_out_y}, ${ch_im_out}, // dY tile dims
        ref_${weight},                                    
        ${ch_im_in},
        ${dim_kernel_x}, ${dim_kernel_y},
        ${stride_x}, ${stride_y},
        ref_${grad_in},                                  
        ${dim_im_in_x}, ${dim_im_in_y},              // dX tile dims
        ${padding_y_top}, ${padding_y_bottom}, ${padding_x_left}, ${padding_x_right},
        ${offset_grad_in_h}, ${offset_grad_in_w},
        ${offset_grad_out_h}, ${offset_grad_out_w},
        ${ctxtBuffer}, ${ctxtBufferSize},
        ${btBuffer}, ${btBufferSize}
    );

    ref_${grad_out} += ${ch_im_out} * ${dim_im_out_y} * ${dim_im_out_x};
    ref_${grad_in}  += ${ch_im_in}  * ${dim_im_in_y}  * ${dim_im_in_x};
}   
"""
)


# Templates for ConvGradW operations
referenceConvGradW2DTemplate = _ConvGradWTemplate("""
// 2D FP ConvGradW NCHW using pulp-trainlib naive (Name: ${nodeName}, Op: ${nodeOp})
${grad_out_type.typeName} ref_${grad_weight}_${grad_out} = ${grad_out};
${data_in_type.typeName} ref_${grad_weight}_${data_in} = ${data_in};
${grad_weight_type.typeName} ref_${grad_weight}_out = ${grad_weight};

% if tileIdxPtr != 'NULL':
{
    static uint32_t ${nodeName}_last_step = 0xFFFFFFFFu;
    if ((uint32_t)*${tileIdxPtr} != ${nodeName}_last_step) {
        memset(${grad_weight}, 0, (${ch_im_out} * ${ch_im_in} * ${dim_kernel_x} * ${dim_kernel_y}) * sizeof(${grad_weight_type.referencedType.typeName}));
        ${nodeName}_last_step = (uint32_t)*${tileIdxPtr};
    }
}
% else:
memset(${grad_weight}, 0, (${ch_im_out} * ${ch_im_in} * ${dim_kernel_x} * ${dim_kernel_y}) * sizeof(${grad_weight_type.referencedType.typeName}));
% endif

for (uint32_t n=0; n<${batch}; ++n) {
    PULP_ConvGradW2d_fp${grad_out_type.referencedType.typeWidth}_fp${data_in_type.referencedType.typeWidth}_fp${grad_weight_type.referencedType.typeWidth}_CHW(
        ref_${grad_weight}_${grad_out},
        ${dim_im_out_x}, ${dim_im_out_y}, ${ch_im_out},
        ref_${grad_weight}_${data_in},
        ${dim_im_in_x}, ${dim_im_in_y}, ${ch_im_in},
        ${dim_kernel_x}, ${dim_kernel_y},
        ${stride_x}, ${stride_y},
        ref_${grad_weight}_out,
        ${padding_y_top}, ${padding_y_bottom}, ${padding_x_left}, ${padding_x_right}
    );

    ref_${grad_weight}_${grad_out} += ${ch_im_out} * ${dim_im_out_y} * ${dim_im_out_x};
    ref_${grad_weight}_${data_in} += ${ch_im_in} * ${dim_im_in_y} * ${dim_im_in_x};
}
""")

referenceConvGradW2DIm2ColTemplate = PULP2DFloatConvGradWIm2ColTemplate("""
// 2D FP ConvGradW NCHW using pulp-trainlib Im2Col (Name: ${nodeName}, Op: ${nodeOp})
${grad_out_type.typeName} ref_${grad_weight}_${grad_out} = ${grad_out};
${data_in_type.typeName} ref_${grad_weight}_${data_in} = ${data_in};
${grad_weight_type.typeName} ref_${grad_weight}_out = ${grad_weight};

% if tileIdxPtr != 'NULL':
{
    static uint32_t ${nodeName}_last_step = 0xFFFFFFFFu;
    if ((uint32_t)*${tileIdxPtr} != ${nodeName}_last_step) {
        memset(${grad_weight}, 0, (${ch_im_out} * ${ch_im_in} * ${dim_kernel_x} * ${dim_kernel_y}) * sizeof(${grad_weight_type.referencedType.typeName}));
        ${nodeName}_last_step = (uint32_t)*${tileIdxPtr};
    }
}
% else:
memset(${grad_weight}, 0, (${ch_im_out} * ${ch_im_in} * ${dim_kernel_x} * ${dim_kernel_y}) * sizeof(${grad_weight_type.referencedType.typeName}));
% endif

for (uint32_t n=0; n<${batch}; ++n) {
    PULP_ConvGradW2d_fp${grad_out_type.referencedType.typeWidth}_fp${data_in_type.referencedType.typeWidth}_fp${grad_weight_type.referencedType.typeWidth}_CHW_Im2Col(
        ref_${grad_weight}_${grad_out},
        ${dim_im_out_x}, ${dim_im_out_y}, ${ch_im_out},
        ref_${grad_weight}_${data_in},
        ${dim_im_in_x}, ${dim_im_in_y}, ${ch_im_in},
        ${dim_kernel_x}, ${dim_kernel_y},
        ${stride_x}, ${stride_y},
        ref_${grad_weight}_out,
        ${padding_y_top}, ${padding_y_bottom}, ${padding_x_left}, ${padding_x_right},
        ${ctxtBuffer}, ${ctxtBufferSize}
    );

    ref_${grad_weight}_${grad_out} += ${ch_im_out} * ${dim_im_out_y} * ${dim_im_out_x};
    ref_${grad_weight}_${data_in} += ${ch_im_in} * ${dim_im_in_y} * ${dim_im_in_x};
}
""")

#  ============================================================================
#  Depthwise Convolution Gradient Templates
#  ============================================================================


referenceDWConvGradW2DTemplate = _ConvGradWTemplate("""
// 2D FP DW ConvGradW NCHW (Name: ${nodeName}, Op: ${nodeOp})
${grad_out_type.typeName} ref_${grad_weight}_${grad_out} = ${grad_out};
${data_in_type.typeName} ref_${grad_weight}_${data_in} = ${data_in};
${grad_weight_type.typeName} ref_${grad_weight}_out = ${grad_weight};

% if tileIdxPtr != 'NULL':
{
    static uint32_t ${nodeName}_last_step = 0xFFFFFFFFu;
    if ((uint32_t)*${tileIdxPtr} != ${nodeName}_last_step) {
        memset(${grad_weight}, 0, ${ch_im_out} * ${dim_kernel_x} * ${dim_kernel_y} * sizeof(${grad_weight_type.referencedType.typeName}));
        ${nodeName}_last_step = (uint32_t)*${tileIdxPtr};
    }
}
% else:
memset(${grad_weight}, 0, ${ch_im_out} * ${dim_kernel_x} * ${dim_kernel_y} * sizeof(${grad_weight_type.referencedType.typeName}));
% endif

for (uint32_t n=0; n<${batch}; ++n) {
    PULP_DWConvGradW2d_fp${grad_out_type.referencedType.typeWidth}_fp${data_in_type.referencedType.typeWidth}_fp${grad_weight_type.referencedType.typeWidth}_CHW(
        ref_${grad_weight}_${grad_out},
        ${dim_im_out_x}, ${dim_im_out_y}, ${ch_im_out},
        ref_${grad_weight}_${data_in},
        ${dim_im_in_x}, ${dim_im_in_y}, ${ch_im_in},
        ${dim_kernel_x}, ${dim_kernel_y},
        ${stride_x}, ${stride_y},
        ref_${grad_weight}_out,
        ${padding_y_top}, ${padding_y_bottom}, ${padding_x_left}, ${padding_x_right}
    );

    ref_${grad_weight}_${grad_out} += ${ch_im_out} * ${dim_im_out_x} * ${dim_im_out_y};
    ref_${grad_weight}_${data_in} += ${ch_im_in} * ${dim_im_in_x} * ${dim_im_in_y};
}

""")


referenceDWConvGradX2DTiledTemplate = NodeTemplate("""
// 2D FP DW ConvGradX (dX) CHW tiled (Name: ${nodeName}, Op: ${nodeOp})
${grad_out_type.typeName}  ref_${grad_out} = ${grad_out};   // dY
${weight_type.typeName}   ref_${weight}  = ${weight};    // W 
${grad_in_type.typeName}  ref_${grad_in}_out = ${grad_in};  // dX

for (uint32_t n=0; n<${batch}; ++n) {
    PULP_DWConvGradX2d_fp${grad_out_type.referencedType.typeWidth}_fp${weight_type.referencedType.typeWidth}_fp${grad_in_type.referencedType.typeWidth}_CHW_tiled(
        ref_${grad_out},
        ${dim_im_out_x}, ${dim_im_out_y}, ${ch_im_out},
        ref_${weight},
        ${ch_im_in},
        ${dim_kernel_x}, ${dim_kernel_y},
        ${stride_x}, ${stride_y},
        ref_${grad_in}_out,
        ${dim_im_in_x}, ${dim_im_in_y},
        ${padding_y_top}, ${padding_y_bottom}, ${padding_x_left}, ${padding_x_right},
        ${offset_grad_in_h}, ${offset_grad_in_w},
        ${offset_grad_out_h}, ${offset_grad_out_w}
    );

    ref_${grad_out} += ${ch_im_out} * ${dim_im_out_y} * ${dim_im_out_x};
    ref_${grad_in}_out += ${ch_im_in}  * ${dim_im_in_y}  * ${dim_im_in_x};
}
""")

#  ============================================================================
#  Pointwise Convolution Gradient Templates
#  ============================================================================

class PULP2DFloatPWConvGradXTemplate(NodeTemplate):

    def __init__(self, templateStr):
        super().__init__(templateStr)

    @staticmethod
    def computeTransientBuffersSize(
            ctxt: NetworkContext,
            operatorRepresentation: OperatorRepresentation) -> List[Tuple[str, Union[int, IntVar]]]:
        # Transpose buffer for weight matrix transpose (C_out x C_in)
        # For pointwise convolution, kernel size is 1x1
        bt_dim = (operatorRepresentation["weight_type"].typeWidth // 8) * \
                 operatorRepresentation['ch_im_in'] * operatorRepresentation['ch_im_out']

        bt_name = operatorRepresentation['nodeName'] + "_transpose_buffer"

        return [(bt_name, bt_dim)]

    def hoistTransientBuffers(self, ctxt: NetworkContext,
                              operatorRepresentation: OperatorRepresentation) -> Tuple[NetworkContext, Dict, List[str]]:
        bt_name, bt_dim = PULP2DFloatPWConvGradXTemplate.computeTransientBuffersSize(
            ctxt, operatorRepresentation)[0]

        ctxt.hoistTransientBuffer(bt_name, bt_dim)

        operatorRepresentation['transposeBuffer'] = bt_name
        operatorRepresentation['transposeBufferSize'] = bt_dim

        return ctxt, operatorRepresentation, [bt_name]


referencePWConvGradW2DTemplate = _ConvGradWTemplate("""
// 2D FP Pointwise ConvGradW (1x1) NCHW using pulp-trainlib pw interface (Name: ${nodeName}, Op: ${nodeOp})
${grad_out_type.typeName} ref_${grad_weight}_${grad_out} = ${grad_out};
${data_in_type.typeName} ref_${grad_weight}_${data_in} = ${data_in};
${grad_weight_type.typeName} ref_${grad_weight}_out = ${grad_weight};

% if tileIdxPtr != 'NULL':
{
    static uint32_t ${nodeName}_last_step = 0xFFFFFFFFu;
    if ((uint32_t)*${tileIdxPtr} != ${nodeName}_last_step) {
        memset(${grad_weight}, 0, ${ch_im_out} * ${ch_im_in} * sizeof(${grad_weight_type.referencedType.typeName}));
        ${nodeName}_last_step = (uint32_t)*${tileIdxPtr};
    }
}
% else:
memset(${grad_weight}, 0, ${ch_im_out} * ${ch_im_in} * sizeof(${grad_weight_type.referencedType.typeName}));
% endif

for (uint32_t n=0; n<${batch}; ++n) {
    PULP_PWConvGradW2d_fp${grad_out_type.referencedType.typeWidth}_fp${data_in_type.referencedType.typeWidth}_fp${grad_weight_type.referencedType.typeWidth}_CHW(
        ref_${grad_weight}_${grad_out},
        ${dim_im_out_x}, ${dim_im_out_y}, ${ch_im_out},
        ref_${grad_weight}_${data_in},
        ${dim_im_in_x}, ${dim_im_in_y}, ${ch_im_in},
        ref_${grad_weight}_out
    );
}   

    ref_${grad_weight}_${grad_out} += ${ch_im_out} * ${dim_im_out_y} * ${dim_im_out_x};
    ref_${grad_weight}_${data_in} += ${ch_im_in} * ${dim_im_in_y} * ${dim_im_in_x};

""")

referencePWConvGradX2DTemplate = PULP2DFloatPWConvGradXTemplate("""
// 2D FP Pointwise ConvGradX (1x1) CHW using pulp-trainlib pw interface (Name: ${nodeName}, Op: ${nodeOp})
${grad_out_type.typeName}  ref_${grad_in}_${grad_out} = ${grad_out};   // dY
${weight_type.typeName}   ref_${grad_in}_${weight}  = ${weight};    // W
${grad_in_type.typeName} ref_${grad_in}_out        = ${grad_in};  // dX

for (uint32_t n=0; n<${batch}; ++n) {
    PULP_PWConvGradX2d_fp${grad_out_type.referencedType.typeWidth}_fp${weight_type.referencedType.typeWidth}_fp${grad_in_type.referencedType.typeWidth}_CHW(
        ref_${grad_in}_${grad_out},
        ${dim_im_out_x}, ${dim_im_out_y}, ${ch_im_out},
        ref_${grad_in}_${weight},
        ${ch_im_in},
        ref_${grad_in}_out,
        ${dim_im_in_x}, ${dim_im_in_y},
        ${transposeBuffer}, ${transposeBufferSize}
    );

    ref_${grad_in}_${grad_out} += ${ch_im_out} * ${dim_im_out_y} * ${dim_im_out_x};
    ref_${grad_in}_out        += ${ch_im_in}  * ${dim_im_in_y}  * ${dim_im_in_x};
}

""")


# Template for ConvGradB: dB[c] = sum_{n,h,w} dY[n,c,h,w]
referenceConvGradB2DTemplate = NodeTemplate("""
// 2D FP ConvGradB: bias gradient = sum dY over N,H,W (Name: ${nodeName}, Op: ${nodeOp})
${grad_out_type.typeName} ref_dB_dy = ${grad_out};
${grad_bias_type.typeName} ref_dB_db = ${grad_bias};
for (uint32_t c = 0; c < ${ch_im_out}; ++c) {
    ref_dB_db[c] = 0.0f;
    for (uint32_t n = 0; n < ${batch}; ++n) {
        for (uint32_t h = 0; h < ${dim_im_out_y}; ++h) {
            for (uint32_t w = 0; w < ${dim_im_out_x}; ++w) {
                ref_dB_db[c] += ref_dB_dy[n * ${ch_im_out} * ${dim_im_out_y} * ${dim_im_out_x} + c * ${dim_im_out_y} * ${dim_im_out_x} + h * ${dim_im_out_x} + w];
            }
        }
    }
}
""")
# SPDX-FileCopyrightText: 2023 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Dict, List, Tuple, Union

from ortools.constraint_solver.pywrapcp import IntVar

from Deeploy.DeeployTypes import NetworkContext, NodeTemplate, OperatorRepresentation

# Predictable L1 tileIdxPtr symbol convention, matching
# TilingHoistingMixIn._DEFAULT_HOIST_PREFIX ("TILING_CODEGEN_") +
# f"{memory}_{nodeName}_" + "tileIdxPtr". Pre-hoisted in alignToContext so
# the template always sees the real name even when a Closure pass renders
# the kernel body before TilingCodeGeneration runs.
_TILE_IDX_SYMBOL_FMT = "TILING_CODEGEN_L1_{node}_tileIdxPtr"


def _is_tiled_expr(val: Any) -> bool:
    """Tiled template vars have been rewritten by TilingVariableReplacement
    into C reference-buffer name strings (e.g.
    "DeeployNetwork_..._ch_im_out_ref"); untiled vars stay as literal ints
    (or in rare cases, plain identifier strings that are still not tiled).

    For ConvGradW's tile-axis detection all the dim template keys (ch_im_*,
    dim_im_*) start life as integers; when the tiler elects to tile one of
    them, the replacement swaps in a buffer-name string. So 'is str' is a
    sufficient oracle for "this dim was tiled by the tiler".
    """
    return isinstance(val, str)


class _ConvGradWTemplate(NodeTemplate):
    """NodeTemplate for ConvGradW operators.

    Pre-hoists a predictable L1 ``tileIdxPtr`` buffer under
    ``TILING_CODEGEN_L1_{nodeName}_tileIdxPtr``. The tiling pass's
    ``_hoistTileNumAndIdxPtr`` is idempotent and will reuse this exact
    symbol. Pre-hoisting here matters because Closure code-transformation
    passes render the kernel body via ``executionBlock.generate(ctxt)``
    BEFORE ``TilingCodeGeneration`` runs; without pre-hoisting the
    template's tileIdxPtr reference would render to the 'NULL' sentinel
    and the first-tile memset guard would never fire.

    Also stashes Mako-accessible ``_cout_tiled`` / ``_hw_tiled`` flags so
    each per-op template body can emit the correct dW-zeroing strategy
    for whichever tile dimension the tiler ended up picking.
    """

    def alignToContext(self, ctxt: NetworkContext,
                       operatorRepresentation: OperatorRepresentation) -> Tuple[NetworkContext, Dict, List[str]]:
        node = operatorRepresentation['nodeName']
        symbol = _TILE_IDX_SYMBOL_FMT.format(node = node)

        if not ctxt.is_buffer(symbol):
            # Declare a 1-element local uint32_t counter that's stack-scoped
            # in the outermost generated function (re-initialised to 0 on
            # every entry = every backward pass). Matches the shape the
            # tiling pass would otherwise create in _hoistTileNumAndIdxPtr.
            from Deeploy.AbstractDataTypes import PointerClass
            from Deeploy.CommonExtensions.DataTypes import uint32_t
            tileIdxPtr = ctxt.VariableBuffer(symbol, shape = [1])
            ctxt.add(tileIdxPtr, "local")
            tileIdxPtr._type = PointerClass(uint32_t)
            tileIdxPtr._instance = tileIdxPtr._type(tileIdxPtr.name, ctxt)
            tileIdxPtr.allocTemplate = NodeTemplate("")
            tileIdxPtr.deallocTemplate = NodeTemplate("")
            tileIdxPtr.initTemplate = NodeTemplate("${type.referencedType.typeName} bu_${name} = 0;\n"
                                                   "${type.referencedType.typeName}* ${name} = &bu_${name};")

        operatorRepresentation['tileIdxPtr'] = symbol
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
""")

# Templates for ConvGradW operations
referenceConvGradW2DTemplate = _ConvGradWTemplate("""
// 2D FP ConvGradW NCHW using pulp-trainlib naive (Name: ${nodeName}, Op: ${nodeOp})
${grad_out_type.typeName} ref_${grad_weight}_${grad_out} = ${grad_out};
${data_in_type.typeName} ref_${grad_weight}_${data_in} = ${data_in};
${grad_weight_type.typeName} ref_${grad_weight}_out = ${grad_weight};

## Emit a different memset strategy depending on what the tiler chose:
##   H/W tiled (but not C_out): memset once at first tile (preserves mm_add
##                              accumulation of HW partials across tiles).
##   otherwise (C_out tiled, or untiled): memset every call (each tile
##                              computes a fresh dW slice in L1 which is
##                              reused across tiles).
## Tiled template vars render as '*..._ref' pointer-deref strings; untiled
## vars render as literal ints/identifiers — see _is_tiled_expr.
% if (isinstance(dim_im_out_x, str) or isinstance(dim_im_out_y, str) or isinstance(dim_im_in_x, str) or isinstance(dim_im_in_y, str)) and not isinstance(ch_im_out, str):
if ((uint32_t)*${tileIdxPtr} == 0u) {
    memset(${grad_weight}, 0, (${ch_im_out} * ${ch_im_in} * ${dim_kernel_x} * ${dim_kernel_y}) * sizeof(${grad_weight_type.referencedType.typeName}));
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

## Emit a different memset strategy depending on what the tiler chose:
##   H/W tiled (but not C_out): memset once at first tile (preserves mm_add
##                              accumulation of HW partials across tiles).
##   otherwise (C_out tiled, or untiled): memset every call (each tile
##                              computes a fresh dW slice in L1 which is
##                              reused across tiles).
## Tiled template vars render as '*..._ref' pointer-deref strings; untiled
## vars render as literal ints/identifiers — see _is_tiled_expr.
% if (isinstance(dim_im_out_x, str) or isinstance(dim_im_out_y, str) or isinstance(dim_im_in_x, str) or isinstance(dim_im_in_y, str)) and not isinstance(ch_im_out, str):
if ((uint32_t)*${tileIdxPtr} == 0u) {
    memset(${grad_weight}, 0, (${ch_im_out} * ${ch_im_in} * ${dim_kernel_x} * ${dim_kernel_y}) * sizeof(${grad_weight_type.referencedType.typeName}));
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

## Emit a different memset strategy depending on what the tiler chose:
##   H/W tiled (but not C_out): memset once at first tile (preserves mm_add
##                              accumulation of HW partials across tiles).
##   otherwise (C_out tiled, or untiled): memset every call (each tile
##                              computes a fresh dW slice in L1 which is
##                              reused across tiles).
## Tiled template vars render as '*..._ref' pointer-deref strings; untiled
## vars render as literal ints/identifiers — see _is_tiled_expr.
% if (isinstance(dim_im_out_x, str) or isinstance(dim_im_out_y, str) or isinstance(dim_im_in_x, str) or isinstance(dim_im_in_y, str)) and not isinstance(ch_im_out, str):
if ((uint32_t)*${tileIdxPtr} == 0u) {
    memset(${grad_weight}, 0, ${ch_im_out} * ${dim_kernel_x} * ${dim_kernel_y} * sizeof(${grad_weight_type.referencedType.typeName}));
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
    """PW (1x1) ConvGradX template.

    The direct PULP_PWConvGradX2d_fp32_fp32_fp32_CHW kernel parallelises over
    Cin and streams W rows / dY rows contiguously, so no weight-transpose
    scratch is required. Allocating a Cin*Cout transient (the old
    transpose buffer) used to eat ~64 KB of L1 for the MobileNetV1
    block 6-10 PW layers and forced the tiler to fragment Cin/H/W into
    ~36 tiles per layer; removing it lets the tiler pick coarse tiles.
    """

    def __init__(self, templateStr):
        super().__init__(templateStr)


referencePWConvGradW2DTemplate = _ConvGradWTemplate("""
// 2D FP Pointwise ConvGradW (1x1) NCHW using pulp-trainlib pw interface (Name: ${nodeName}, Op: ${nodeOp})
${grad_out_type.typeName} ref_${grad_weight}_${grad_out} = ${grad_out};
${data_in_type.typeName} ref_${grad_weight}_${data_in} = ${data_in};
${grad_weight_type.typeName} ref_${grad_weight}_out = ${grad_weight};

% if (isinstance(dim_im_out_x, str) or isinstance(dim_im_out_y, str) or isinstance(dim_im_in_x, str) or isinstance(dim_im_in_y, str)) and not isinstance(ch_im_out, str):
if ((uint32_t)*${tileIdxPtr} == 0u) {
    memset(${grad_weight}, 0, ${ch_im_out} * ${ch_im_in} * sizeof(${grad_weight_type.referencedType.typeName}));
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
        ${dim_im_in_x}, ${dim_im_in_y}
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

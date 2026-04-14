# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, List, Tuple

import numpy as np

from Deeploy.DeeployTypes import NetworkContext, NodeTemplate, OperatorRepresentation

def _ne16_conv_1x1_weight_layout(W, w_bits=8):
    """Pack int8 weights for NE16 1x1 conv mode.
    W: int8 [Ko, Ki] -> uint8 [Ko, Nb_KI, Qw, 2] (bitplane packed)
    Weights stored as uint8 = int8 + 128.
    """
    tp_in = 16
    Ko_, Ki_ = W.shape
    W_uint8 = (W.astype(np.int32) + 128).astype(np.uint8)
    nb_ki = (Ki_ + tp_in - 1) // tp_in
    w_binary = np.zeros((Ko_ * nb_ki, w_bits, 8, tp_in // 8), dtype=np.uint8)
    for ko in range(Ko_):
        for ki_maj in range(nb_ki):
            for ki_min in range(tp_in):
                idx = ko * nb_ki + ki_maj
                ki = ki_maj * tp_in + ki_min
                val = int(W_uint8[ko, ki]) if ki < Ki_ else 0
                for q in range(w_bits):
                    w_binary[idx, q, ki_min % 8, ki_min // 8] = (val >> q) & 1
    space = np.logspace(0, 7, num=8, base=2, dtype=np.int32).reshape((8, 1))
    w_layout = np.sum(w_binary * space, axis=2, dtype=np.uint8)
    return w_layout.reshape((Ko_, nb_ki, w_bits, tp_in // 8))


class NE16GEMMTemplate(NodeTemplate):

    def __init__(self, templateStr):
        super().__init__(templateStr)

    def alignToContext(self, ctxt: NetworkContext,
                       operatorRepresentation: OperatorRepresentation) -> Tuple[NetworkContext, Dict, List[str]]:

        A = ctxt.lookup(operatorRepresentation['A'])
        B = ctxt.lookup(operatorRepresentation['B'])
        C = ctxt.lookup(operatorRepresentation['C'])
        data_out = ctxt.lookup(operatorRepresentation['data_out'])

        # Determine signedness from type system (reliable, post-type-inference)
        input_signed = A._type.referencedType.typeMin < 0
        output_bits = data_out._type.referencedType.typeWidth
        output_signed = data_out._type.referencedType.typeMin < 0

        operatorRepresentation['input_signed'] = input_signed
        operatorRepresentation['output_bits'] = output_bits
        operatorRepresentation['quant_bits'] = 2 if output_bits == 32 else 0
        operatorRepresentation['quant_norect'] = 1 if (output_bits == 32 or output_signed) else 0

        # Weight packing and signed bias compensation
        w_int8 = B.values.astype(np.int8)  # [Ko, Ki], still int8 at this point
        Ko, Ki = w_int8.shape

        # Truncate broadcast bias [M, O] → per-channel [Ko] for NE16
        bias_flat = C.values.flatten()
        if bias_flat.size > Ko:
            C.values = bias_flat[:Ko].copy()

        # Signed input bias compensation
        if input_signed:
            # Compute w_sum BEFORE packing (needed for signed bias compensation)
            w_sum = w_int8.astype(np.int64).sum(axis=1)  # [Ko]
            bias_values = C.values.flatten().astype(np.int64)
            if 'mul' in operatorRepresentation:
                # RequantizedGemm: bias -= 128 * w_sum * scale
                scale_buf = ctxt.lookup(operatorRepresentation['mul'])
                scale_values = scale_buf.values.flatten().astype(np.int64)
                bias_values -= 128 * w_sum * scale_values
            else:
                # Gemm int32: bias -= 128 * w_sum (no scale)
                bias_values -= 128 * w_sum
            C.values = bias_values.astype(np.int32)

        # Pack weights to NE16 bitplane format
        ne16_weights = _ne16_conv_1x1_weight_layout(w_int8)
        B.values = ne16_weights.reshape(Ko, -1)

        return ctxt, operatorRepresentation, []


# 8-bit output template (RequantizedGemm) — uses tiled ${mul} and ${scale_n}
referenceTemplate = NE16GEMMTemplate("""
// NE16 Linear 8-bit (Name: ${nodeName}, Op: ${nodeOp})

% if input_signed:
// Signed input: add 128 offset to convert int8 -> uint8 (multi-core SIMD)
{
    ne16_int8_to_uint8_T _offset_arg = {
        .In = (int8_t *)${A},
        .Out = (uint8_t *)${A},
        .size = ${batch} * ${M} * ${N}
    };
    pi_cl_team_fork(NUM_CORES, (void *)ne16_int8_to_uint8, &_offset_arg);
}
% endif

{
    unsigned int _ne16_cfg = 0;
    _ne16_cfg |= ((8 - 1)              & NE16_MASK_WBITS_M1)         << NE16_SHIFT_WBITS_M1;
    _ne16_cfg |= (0                     & NE16_MASK_MODE16)            << NE16_SHIFT_MODE16;
    _ne16_cfg |= (1                     & NE16_MASK_OUTQUANT)          << NE16_SHIFT_OUTQUANT;
    _ne16_cfg |= (NE16_FILTER_MODE_1x1  & NE16_MASK_FILTER_MODE)       << NE16_SHIFT_FILTER_MODE;
    _ne16_cfg |= (0                     & NE16_MASK_LINEAR_MODE)       << NE16_SHIFT_LINEAR_MODE;
    _ne16_cfg |= (0                     & NE16_MASK_STRIDED_MODE)      << NE16_SHIFT_STRIDED_MODE;
    _ne16_cfg |= (NE16_BITS_8BIT        & NE16_MASK_NORM_BITS)         << NE16_SHIFT_NORM_BITS;
    _ne16_cfg |= (0                     & NE16_MASK_STREAMIN)          << NE16_SHIFT_STREAMIN;
    _ne16_cfg |= (1                     & NE16_MASK_WEIGHT_OFFSET_CFG) << NE16_SHIFT_WEIGHT_OFFSET_CFG;
    _ne16_cfg |= (0                     & NE16_MASK_QUANT_RIGHT_SHIFT) << NE16_SHIFT_QUANT_RIGHT_SHIFT;
    _ne16_cfg |= (${quant_bits}         & NE16_MASK_QUANT_BITS)        << NE16_SHIFT_QUANT_BITS;
    _ne16_cfg |= (${quant_norect}       & NE16_MASK_QUANT_NORECT)      << NE16_SHIFT_QUANT_NORECT;
    _ne16_cfg |= (1                     & NE16_MASK_NORM_SHIFT)        << NE16_SHIFT_NORM_SHIFT;
    _ne16_cfg |= (1                     & NE16_MASK_NORM_BIAS)         << NE16_SHIFT_NORM_BIAS;

    NE16_Enable();
    NE16_SoftReset();

    KerConv_NE16_T _ne16_arg = {
        .In                   = (void *)${A},
        .Filter               = (unsigned short *)${B},
        .Bias                 = (int *)${C},
        .Out                  = (void *)${data_out},
        .Scale                = (unsigned char *)${mul},
        .ScaleN               = (unsigned char *)${scale_n},
        .Tile_InFeat          = ${N},
        .TotalInFeatures      = ${N},
        .Tile_InH             = 1,
        .Tile_InW             = ${batch} * ${M},
        .Tile_OutFeat         = ${O},
        .Tile_OutH            = 1,
        .Tile_OutW            = ${batch} * ${M},
        .FilterSize           = 1,
        .Pad_Val              = 0,
        .Pad                  = (v4s){0, 0, 0, 0},
        .W_Offset             = -128,
        .Qw                   = 8,
        .Mode16               = 0,
        .FirstD0              = 1,
        .LastD0               = 1,
        .Default_NE16_Job_Cfg = _ne16_cfg,
        .Fx                   = 1,
        .Fy                   = 1,
        .Sx                   = 1,
        .Sy                   = 1,
        .Dx                   = 1,
        .Dy                   = 1,
        .BuffOut              = NULL,
        .Infos                = NULL,
        .Extra                = NULL,
    };
    KerConv1x1_SmallHW_Stride1_NE16(&_ne16_arg);

    NE16_Disable();
}
""")

# Int32 output template (plain Gemm) — hardcoded scale=1, scale_n=0
int32OutputTemplate = NE16GEMMTemplate("""
// NE16 Linear Int32 (Name: ${nodeName}, Op: ${nodeOp})

% if input_signed:
// Signed input: add 128 offset to convert int8 -> uint8 (multi-core SIMD)
{
    ne16_int8_to_uint8_T _offset_arg = {
        .In = (int8_t *)${A},
        .Out = (uint8_t *)${A},
        .size = ${batch} * ${M} * ${N}
    };
    pi_cl_team_fork(NUM_CORES, (void *)ne16_int8_to_uint8, &_offset_arg);
}
% endif

{
    unsigned char _ne16_ones[${O}];
    unsigned char _ne16_zeros[${O}];
    memset(_ne16_ones, 1, ${O});
    memset(_ne16_zeros, 0, ${O});

    unsigned int _ne16_cfg = 0;
    _ne16_cfg |= ((8 - 1)              & NE16_MASK_WBITS_M1)         << NE16_SHIFT_WBITS_M1;
    _ne16_cfg |= (0                     & NE16_MASK_MODE16)            << NE16_SHIFT_MODE16;
    _ne16_cfg |= (1                     & NE16_MASK_OUTQUANT)          << NE16_SHIFT_OUTQUANT;
    _ne16_cfg |= (NE16_FILTER_MODE_1x1  & NE16_MASK_FILTER_MODE)       << NE16_SHIFT_FILTER_MODE;
    _ne16_cfg |= (0                     & NE16_MASK_LINEAR_MODE)       << NE16_SHIFT_LINEAR_MODE;
    _ne16_cfg |= (0                     & NE16_MASK_STRIDED_MODE)      << NE16_SHIFT_STRIDED_MODE;
    _ne16_cfg |= (NE16_BITS_8BIT        & NE16_MASK_NORM_BITS)         << NE16_SHIFT_NORM_BITS;
    _ne16_cfg |= (0                     & NE16_MASK_STREAMIN)          << NE16_SHIFT_STREAMIN;
    _ne16_cfg |= (1                     & NE16_MASK_WEIGHT_OFFSET_CFG) << NE16_SHIFT_WEIGHT_OFFSET_CFG;
    _ne16_cfg |= (0                     & NE16_MASK_QUANT_RIGHT_SHIFT) << NE16_SHIFT_QUANT_RIGHT_SHIFT;
    _ne16_cfg |= (${quant_bits}         & NE16_MASK_QUANT_BITS)        << NE16_SHIFT_QUANT_BITS;
    _ne16_cfg |= (${quant_norect}       & NE16_MASK_QUANT_NORECT)      << NE16_SHIFT_QUANT_NORECT;
    _ne16_cfg |= (1                     & NE16_MASK_NORM_SHIFT)        << NE16_SHIFT_NORM_SHIFT;
    _ne16_cfg |= (1                     & NE16_MASK_NORM_BIAS)         << NE16_SHIFT_NORM_BIAS;

    NE16_Enable();
    NE16_SoftReset();

    KerConv_NE16_T _ne16_arg = {
        .In                   = (void *)${A},
        .Filter               = (unsigned short *)${B},
        .Bias                 = (int *)${C},
        .Out                  = (void *)${data_out},
        .Scale                = _ne16_ones,
        .ScaleN               = _ne16_zeros,
        .Tile_InFeat          = ${N},
        .TotalInFeatures      = ${N},
        .Tile_InH             = 1,
        .Tile_InW             = ${batch} * ${M},
        .Tile_OutFeat         = ${O},
        .Tile_OutH            = 1,
        .Tile_OutW            = ${batch} * ${M},
        .FilterSize           = 1,
        .Pad_Val              = 0,
        .Pad                  = (v4s){0, 0, 0, 0},
        .W_Offset             = -128,
        .Qw                   = 8,
        .Mode16               = 0,
        .FirstD0              = 1,
        .LastD0               = 1,
        .Default_NE16_Job_Cfg = _ne16_cfg,
        .Fx                   = 1,
        .Fy                   = 1,
        .Sx                   = 1,
        .Sy                   = 1,
        .Dx                   = 1,
        .Dy                   = 1,
        .BuffOut              = NULL,
        .Infos                = NULL,
        .Extra                = NULL,
    };
    KerConv1x1_SmallHW_Stride1_NE16(&_ne16_arg);

    NE16_Disable();
}
""")

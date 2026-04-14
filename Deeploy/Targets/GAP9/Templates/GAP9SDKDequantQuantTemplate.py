# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0
#
# Quant/Dequant templates using GAP9 SDK kernels (CNN_Copy.c).
# All called via GAP9Transformer which handles pi_cl_team_fork.

from Deeploy.DeeployTypes import NodeTemplate

# ============================================================
# Dequant templates: int → fp16 (SDK kernels from CNN_Copy.c)
# ============================================================

# int8 → fp16: SDK kernel CNN_FpsIEEE16
fp16DequantI8Template = NodeTemplate("""
// FP16 Dequant int8→fp16 (Name: ${nodeName}, Op: ${nodeOp})
{
    signed char _dq_infos[8];
    *((float *)(_dq_infos + 0)) = (float)(-(${zero_point}));
    *((float *)(_dq_infos + 4)) = (float)(${scale});
    CNN_Quantize_T _dq_arg = {
        .In = (void *)${data_in},
        .Out = (void *)${data_out},
        .W = ${size},
        .H = 1,
        .Infos = _dq_infos,
    };
    CNN_FpsIEEE16(&_dq_arg);
}
""")

# uint8 → fp16: SDK kernel CNN_UFpsIEEE16
fp16DequantU8Template = NodeTemplate("""
// FP16 Dequant uint8→fp16 (Name: ${nodeName}, Op: ${nodeOp})
{
    signed char _dq_infos[8];
    *((float *)(_dq_infos + 0)) = (float)(-(${zero_point}));
    *((float *)(_dq_infos + 4)) = (float)(${scale});
    CNN_Quantize_T _dq_arg = {
        .In = (void *)${data_in},
        .Out = (void *)${data_out},
        .W = ${size},
        .H = 1,
        .Infos = _dq_infos,
    };
    CNN_UFpsIEEE16(&_dq_arg);
}
""")

# ============================================================
# Dequant templates: int → fp32 (SDK kernels from CNN_Copy.c)
# ============================================================

# int8 → fp32: SDK kernel CNN_FpsFloat32
fp32DequantI8Template = NodeTemplate("""
// FP32 Dequant int8→fp32 (Name: ${nodeName}, Op: ${nodeOp})
{
    signed char _dq_infos[8];
    *((float *)(_dq_infos + 0)) = (float)(-(${zero_point}));
    *((float *)(_dq_infos + 4)) = (float)(${scale});
    CNN_FpsFloat32_T _dq_arg = {
        .In = (signed char *)${data_in},
        .Out = (float *)${data_out},
        .W = ${size},
        .H = 1,
        .Infos = _dq_infos,
    };
    CNN_FpsFloat32(&_dq_arg);
}
""")

# uint8 → fp32: SDK kernel CNN_UFpsFloat32
fp32DequantU8Template = NodeTemplate("""
// FP32 Dequant uint8→fp32 (Name: ${nodeName}, Op: ${nodeOp})
{
    signed char _dq_infos[8];
    *((float *)(_dq_infos + 0)) = (float)(-(${zero_point}));
    *((float *)(_dq_infos + 4)) = (float)(${scale});
    CNN_UFpsFloat32_T _dq_arg = {
        .In = (unsigned char *)${data_in},
        .Out = (float *)${data_out},
        .W = ${size},
        .H = 1,
        .Infos = _dq_infos,
    };
    CNN_UFpsFloat32(&_dq_arg);
}
""")

# ============================================================
# Quant templates: fp16 → int (SDK kernels from CNN_Copy.c)
# ============================================================

# fp16 → int8: SDK kernel CNN_IEEE16Fps
fp16QuantI8Template = NodeTemplate("""
// FP16 Quant fp16→int8 (Name: ${nodeName}, Op: ${nodeOp})
{
    signed char _q_infos[8];
    *((float *)(_q_infos + 0)) = (float)(${zero_point});
    *((float *)(_q_infos + 4)) = (float)(${scale});
    CNN_Quantize_T _q_arg = {
        .In = (void *)${data_in},
        .Out = (void *)${data_out},
        .W = ${size},
        .H = 1,
        .Infos = _q_infos,
    };
    CNN_IEEE16Fps(&_q_arg);
}
""")

# fp16 → uint8: SDK kernel CNN_IEEE16UFps
fp16QuantU8Template = NodeTemplate("""
// FP16 Quant fp16→uint8 (Name: ${nodeName}, Op: ${nodeOp})
{
    signed char _q_infos[8];
    *((float *)(_q_infos + 0)) = (float)(${zero_point});
    *((float *)(_q_infos + 4)) = (float)(${scale});
    CNN_Quantize_T _q_arg = {
        .In = (void *)${data_in},
        .Out = (void *)${data_out},
        .W = ${size},
        .H = 1,
        .Infos = _q_infos,
    };
    CNN_IEEE16UFps(&_q_arg);
}
""")

# ============================================================
# Quant templates: fp32 → int (SDK kernels from CNN_Copy.c)
# ============================================================

# fp32 → int8: SDK kernel CNN_Float32Fps
fp32QuantI8Template = NodeTemplate("""
// FP32 Quant fp32→int8 (Name: ${nodeName}, Op: ${nodeOp})
{
    signed char _q_infos[8];
    *((float *)(_q_infos + 0)) = (float)(${zero_point});
    *((float *)(_q_infos + 4)) = (float)(${scale});
    CNN_Float32Fps_T _q_arg = {
        .In = (float *)${data_in},
        .Out = (signed char *)${data_out},
        .W = ${size},
        .H = 1,
        .Infos = _q_infos,
    };
    CNN_Float32Fps(&_q_arg);
}
""")

# fp32 → uint8: SDK kernel CNN_Float32UFps
fp32QuantU8Template = NodeTemplate("""
// FP32 Quant fp32→uint8 (Name: ${nodeName}, Op: ${nodeOp})
{
    signed char _q_infos[8];
    *((float *)(_q_infos + 0)) = (float)(${zero_point});
    *((float *)(_q_infos + 4)) = (float)(${scale});
    CNN_Float32UFps_T _q_arg = {
        .In = (float *)${data_in},
        .Out = (unsigned char *)${data_out},
        .W = ${size},
        .H = 1,
        .Infos = _q_infos,
    };
    CNN_Float32UFps(&_q_arg);
}
""")

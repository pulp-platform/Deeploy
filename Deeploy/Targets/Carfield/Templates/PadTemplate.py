
from Deeploy.DeeployTypes import  NodeTemplate

# pad_y = pads[2] is for vertical padding (on rows)
# pad_x = pads[3] is for horizontal padding (on columns)
# dim_im_in_x = shape[2] (height) is the number of rows in the input image
# dim_im_in_y = shape[3] (width) is the number of columns in the input image
reference2DTemplate = NodeTemplate("""
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
    for (uint32_t i = 0; i < ${data_out_size}; i++) {
        ${data_out}[i] = ${value};
    }
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

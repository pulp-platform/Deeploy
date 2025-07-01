from typing import Dict, List, Tuple

from Deeploy.DeeployTypes import NetworkContext, NodeTemplate, OperatorRepresentation


class _ConvTranspose1D_Template(NodeTemplate):

    def __init__(self, templateStr):
        super().__init__(templateStr)

    def alignToContext(self, ctxt: NetworkContext,
                       operatorRepresentation: OperatorRepresentation) -> Tuple[NetworkContext, Dict, List[str]]:

        # input/output tensors
        # input/output tensors
        data_in = ctxt.lookup(operatorRepresentation["data_in"])
        data_out = ctxt.lookup(operatorRepresentation["data_out"])

        # quantized tensor offset computation 
        operatorRepresentation["input_offset"] = 0
        if hasattr(data_in, "_signed") and hasattr(data_in, "nLevels"):
            operatorRepresentation["input_offset"] = (data_in._signed == 0) * int(data_in.nLevels // 2)

        operatorRepresentation["output_offset"] = 0
        if hasattr(data_out, "_signed") and hasattr(data_out, "nLevels"):
            operatorRepresentation["output_offset"] = -(data_out._signed == 0) * int(data_out.nLevels // 2)

        # Batch size
        operatorRepresentation["batch"] = data_in.shape[0]

        # input/output shape (format NCW)
        operatorRepresentation["ch_im_in"] = data_in.shape[1]
        operatorRepresentation["dim_im_in_y"] = data_in.shape[2]

        operatorRepresentation["ch_im_out"] = data_out.shape[1]
        operatorRepresentation["dim_im_out_y"] = data_out.shape[2]

        # weights and kernel
        weight = ctxt.lookup(operatorRepresentation["weight"])
        operatorRepresentation["dim_kernel_y"] = weight.shape[2]  # Shape: [C_out, C_in, K]

        # Stride
        operatorRepresentation["stride_y"] = operatorRepresentation.get("stride_y", 1)

        # Bias (optional)
        operatorRepresentation["has_bias"] = "true" if "bias" in operatorRepresentation else "false"
        operatorRepresentation["bias"] = operatorRepresentation.get("bias", "NULL")

        operatorRepresentation[
            "batchOffsetIn"] = operatorRepresentation["ch_im_in"] * operatorRepresentation["dim_im_in_y"]
        operatorRepresentation[
            "batchOffsetOut"] = operatorRepresentation["ch_im_out"] * operatorRepresentation["dim_im_out_y"]

        operatorRepresentation[
            "batchOffsetIn"] = operatorRepresentation["ch_im_in"] * operatorRepresentation["dim_im_in_y"]
        operatorRepresentation[
            "batchOffsetOut"] = operatorRepresentation["ch_im_out"] * operatorRepresentation["dim_im_out_y"]

        return ctxt, operatorRepresentation, []

referenceTemplate = _ConvTranspose1D_Template("""
<%
batchOffsetIn = ch_im_in * dim_im_in_y
batchOffsetOut = ch_im_out * dim_im_out_y
%>

// 1D Transposed Conv (Name: ${nodeName}, Op: ${nodeOp})
BEGIN_SINGLE_CORE
    ${data_in_type.typeName} ref_${data_out}_${data_in} = ${data_in};
    ${data_out_type.typeName} ref_${data_out}_${data_out} = ${data_out};

    for (uint32_t n=0; n<${batch}; ++n) {
        ConvTranspose1d_fp32(
            ref_${data_out}_${data_in}, ${ch_im_in}, ${dim_im_in_y},
            ${weight}, ${ch_im_out}, ${dim_kernel_y},
            ${stride_y},
            ${bias}, ${has_bias},
            ref_${data_out}_${data_out}, ${dim_im_out_y}
        );
        
    }
END_SINGLE_CORE
""")

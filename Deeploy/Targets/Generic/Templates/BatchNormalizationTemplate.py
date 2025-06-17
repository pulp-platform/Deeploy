from Deeploy.DeeployTypes import NodeTemplate

referenceTemplate = NodeTemplate("""
// BatchNorm (Name: ${nodeName}, Op: ${nodeOp})

BEGIN_SINGLE_CORE
    BatchNorm_fp32(
        ${data_in}, ${scale}, ${bias}, ${mean}, ${variance},
        ${data_out}, ${size}
    );
END_SINGLE_CORE
""")

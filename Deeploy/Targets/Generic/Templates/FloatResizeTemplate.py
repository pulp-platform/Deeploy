
from Deeploy.DeeployTypes import NodeTemplate

referenceTemplate = NodeTemplate("""
// Float Resize (Name: ${nodeName}, Op: ${nodeOp})
BEGIN_SINGLE_CORE

${data_in_type.typeName} ref_${data_out}_${data_in} = ${data_in};
${scale_type.typeName} ref_${data_out}_${scale} = ${scale};
${data_out_type.typeName} ref_${data_out}_${data_out} = ${data_out};

int32_t inputA_${data_out}[${shapeA_len}]=${shapeA};
//printf(\" Not you Resize...\\n\");
Resize_fp${data_in_type.referencedType.typeWidth}_fp${data_out_type.referencedType.typeWidth}(
                ref_${data_out}_${data_in}, inputA_${data_out}, ${shapeA_len},
                ref_${data_out}_${scale}, ${shapeB_len},
                ref_${data_out}_${data_out},
                ${coordinate_transformation_mode},
                ${mode}, ${nearest_mode});
//printf(\" Ouf\\n\");
END_SINGLE_CORE
""")

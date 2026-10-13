

from Deeploy.DeeployTypes import NodeTemplate

referenceTemplate = NodeTemplate("""
// Log Template (Name: ${nodeName}, Op: ${nodeOp})
BEGIN_SINGLE_CORE
${data_in_type.typeName} ref_${data_out}_${data_in} = ${data_in};
${data_out_type.typeName} ref_${data_out}_${data_out} = ${data_out};

Log_fp${data_in_type.referencedType.typeWidth}(
ref_${data_out}_${data_in}, ref_${data_out}_${data_out}, ${size});

END_SINGLE_CORE
""")

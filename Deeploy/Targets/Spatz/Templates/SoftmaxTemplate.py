from Deeploy.DeeployTypes import NodeTemplate

# integerTilingTemplate

floatTilingTemplate = NodeTemplate("""
// Softmax (Name: ${nodeName}, Op: ${nodeOp})
Spatz_Softmax_fp${data_in_type.referencedType.typeWidth}_fp${data_out_type.referencedType.typeWidth}(${data_in}, ${data_out}, ${size}, ${lastDimLength});
""")

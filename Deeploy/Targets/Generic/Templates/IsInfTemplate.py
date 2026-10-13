
from Deeploy.DeeployTypes import NodeTemplate

referenceTemplate = NodeTemplate("""
// Add (Name: ${nodeName}, Op: ${nodeOp})
BEGIN_SINGLE_CORE
    for (uint32_t i=0;i<${size};i++){
        ${Y}[i] = isinf(${X}[i]);
    }
END_SINGLE_CORE
""")

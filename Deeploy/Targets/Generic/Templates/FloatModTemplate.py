
from Deeploy.DeeployTypes import NodeTemplate

referenceTemplate = NodeTemplate("""
// Float Mod (Name: ${nodeName}, Op: ${nodeOp})
BEGIN_SINGLE_CORE

${A_type.typeName} ref_${C}_${A} = ${A};
${B_type.typeName} ref_${C}_${B} = ${B};
${C_type.typeName} ref_${C}_${C} = ${C};

int32_t inputA_${C}[${shapeA_len}]=${shapeA};
int32_t inputB_${C}[${shapeB_len}]=${shapeB};

Mod_fp${A_type.referencedType.typeWidth}_fp${B_type.referencedType.typeWidth}(
                ref_${C}_${A},
                ref_${C}_${B},
                1,
                ref_${C}_${C},
                inputA_${C}, ${shapeA_len},
                inputB_${C}, ${shapeB_len});

END_SINGLE_CORE
""")

# SPDX-FileCopyrightText: 2024 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from Deeploy.DeeployTypes import NodeTemplate

SoftHierInitTemplate = NodeTemplate("${type.typeName} ${name} __attribute__((section(\".l1\")));\n")
SoftHierAllocateTemplate = NodeTemplate("""
if (core_id ==0) {
    % if _memoryLevel == "L1":
    ${name} = (${type.typeName}) flex_l1_malloc(sizeof(${type.referencedType.typeName}) * ${size});\n
    % else:
    ${name} = (${type.typeName}) flex_hbm_malloc(sizeof(${type.referencedType.typeName}) * ${size});\n
    % endif
}
""")

SoftHierGlobalInitTemplate = NodeTemplate(
    "static ${type.referencedType.typeName} ${name}[${size}] __attribute__((section(\".l2\"))) = {${values}};\n")
SoftHierGlobalAllocateTemplate = NodeTemplate("")

SoftHierStructInitTemplate = NodeTemplate("""
static ${type.typeName} ${name} __attribute__((section(\".l1\")));
""")

SoftHierStructAllocateTemplate = NodeTemplate("""
if (core_id == 0) {
    ${name} = (${structDict.typeName}) ${str(structDict)};
}
""")

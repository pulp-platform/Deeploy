# SPDX-FileCopyrightText: 2021 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from Deeploy.DeeployTypes import NodeTemplate

magiaInitTemplate = NodeTemplate("${type.typeName} ${name};\n")

magiaAllocateTemplate = NodeTemplate("""
% if _memoryLevel == "L1":
${name} = (${type.typeName}) magia_l1_malloc(sizeof(${type.referencedType.typeName}) * ${size});\n
% elif _memoryLevel == "L2" or _memoryLevel is None:
${name} = (${type.typeName}) magia_l2_malloc(sizeof(${type.referencedType.typeName}) * ${size});\n
% endif
""")

magiaGlobalInitTemplate = NodeTemplate("""
% if _memoryLevel == "L1":
static ${type.referencedType.typeName} ${name}[${size}] = {${values}};\n
% elif _memoryLevel == "L2" or _memoryLevel is None:
extern ${type.referencedType.typeName} ${name}[${size}] = {${values}};\n
% endif
""")

magiaGlobalAllocateTemplate = NodeTemplate("")
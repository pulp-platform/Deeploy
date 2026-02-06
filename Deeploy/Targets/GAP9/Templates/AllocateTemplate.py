# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from Deeploy.DeeployTypes import NodeTemplate

gap9L2InitTemplate = NodeTemplate("${type.typeName} ${name};\n")

gap9L1InitTemplate = NodeTemplate("${type.typeName} ${name};\n")

gap9L2AllocateTemplate = NodeTemplate(
    "${name} = (${type.typeName}) pi_l2_malloc(sizeof(${type.referencedType.typeName}) * ${size});\n")

gap9L1AllocateTemplate = NodeTemplate(
    "${name} = (${type.typeName}) pi_l1_malloc((void *) 0, sizeof(${type.referencedType.typeName}) * ${size});\n")

gap9L2GlobalInitTemplate = NodeTemplate(
    "static PI_L2 ${type.referencedType.typeName} ${name}[${size}] = {${values}};\n")

gap9L1GlobalInitTemplate = NodeTemplate(
    "static PI_L1 ${type.referencedType.typeName} ${name}[${size}] = {${values}};\n")

gap9L2GlobalAllocateTemplate = NodeTemplate("")

gap9L1GlobalAllocateTemplate = NodeTemplate("")

gap9L2StructInitTemplate = NodeTemplate("""static PI_L2 ${type.typeName} ${name};
""")

gap9L2StructAllocateTemplate = NodeTemplate(""" % for key, value in structDict.items():
    ${name}.${key} = ${value};
% endfor """)

gap9GenericStructInitTemplate = NodeTemplate("""
% if _memoryLevel == "L1":
static PI_L1 ${type.typeName} ${name};\n
% elif _memoryLevel == "L2" or _memoryLevel is None:
static PI_L2 ${type.typeName} ${name};\n
% elif _memoryLevel == "L3":
// ${name} is allocated in L3 \n
% endif
""")

gap9GenericGlobalInitTemplate = NodeTemplate("""
% if _memoryLevel == "L1":
static PI_L1 ${type.referencedType.typeName} ${name}[${size}] = {${values}};\n
% elif _memoryLevel == "L2" or _memoryLevel is None:
static PI_L2 ${type.referencedType.typeName} ${name}[${size}] = {${values}};\n
% elif _memoryLevel == "L3":
// ${name} is allocated in L3 \n
static PI_L2 ${type.referencedType.typeName}* ${name};
% endif
""")

gap9GenericAllocate = NodeTemplate("""
% if _memoryLevel == "L1":
${name} = (${type.typeName}) pi_l1_malloc((void *) 0, sizeof(${type.referencedType.typeName}) * ${size});\n
% elif _memoryLevel == "L2" or _memoryLevel is None:
${name} = (${type.typeName}) pi_l2_malloc(sizeof(${type.referencedType.typeName}) * ${size});\n
% elif _memoryLevel == "L3":
${name} = (${type.typeName}) cl_ram_malloc(sizeof(${type.referencedType.typeName}) * ${size});\n
% else:
//COMPILER BLOCK - MEMORYLEVEL ${_memoryLevel} NOT FOUND \n
${name} = (${type.typeName}) pi_l2_malloc(sizeof(${type.referencedType.typeName}) * ${size});\n
// ${name} with size ${size} allocated in L2!
% endif
""")

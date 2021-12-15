# ----------------------------------------------------------------------
#
# File: AllocateTemplate.py
#
# Last edited: 09.03.2023
#
# Copyright (C) 2023, ETH Zurich and University of Bologna.
#
# Author: Moritz Scherer, ETH Zurich
#
# ----------------------------------------------------------------------
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the License); you may
# not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an AS IS BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from Deeploy.DeeployTypes import NodeTemplate

pulpL2InitTemplate = NodeTemplate("${type.typeName} ${name};\n")

pulpL1InitTemplate = NodeTemplate("${type.typeName} ${name};\n")
#pulpL2AllocateTemplate = NodeTemplate("${name} = (${type.typeName}) pi_l2_malloc(${type.referencedType.typeWidth//8} * ${size});\n")
pulpL2AllocateTemplate = NodeTemplate(
    "${name} = (${type.typeName}) pi_l2_malloc(sizeof(${type.referencedType.typeName}) * ${size});\n")

pulpL1AllocateTemplate = NodeTemplate(
    "${name} = (${type.typeName}) pmsis_l1_malloc(sizeof(${type.referencedType.typeName}) * ${size});\n")

pulpL2GlobalInitTemplate = NodeTemplate(
    "static PI_L2 ${type.referencedType.typeName} ${name}[${size}] = {${values}};\n")

pulpL1GlobalInitTemplate = NodeTemplate(
    "static PI_L1 ${type.referencedType.typeName} ${name}[${size}] = {${values}};\n")

#pulpL2GlobalInitTemplate = NodeTemplate("static const ${type} ${name}[${size}];\n")
pulpL2GlobalAllocateTemplate = NodeTemplate("")

pulpL1GlobalAllocateTemplate = NodeTemplate("")

pulpL2StructInitTemplate = NodeTemplate("""static PI_L2 ${type.typeName} ${name};
""")
#static const ${type}* ${name} = &${name}_UL;

pulpL2StructAllocateTemplate = NodeTemplate(""" % for key, value in structDict.items():
    ${name}.${key} = ${value};
% endfor """)

pulpGenericStructInitTemplate = NodeTemplate("""
% if _memoryLevel == "L1":
static PI_L1 ${type.typeName} ${name};\n
% elif _memoryLevel == "L2" or _memoryLevel is None:
static PI_L2 ${type.typeName} ${name};\n
% elif _memoryLevel == "L3":
// ${name} is allocated in L3 \n
% endif
""")

pulpGenericGlobalInitTemplate = NodeTemplate("""
% if _memoryLevel == "L1":
static PI_L1 ${type.referencedType.typeName} ${name}[${size}] = {${values}};\n
% elif _memoryLevel == "L2" or _memoryLevel is None:
static PI_L2 ${type.referencedType.typeName} ${name}[${size}] = {${values}};\n
% elif _memoryLevel == "L3":
// ${name} is allocated in L3 \n
static PI_L2 ${type.referencedType.typeName}* ${name};
% endif
""")

pulpGenericAllocate = NodeTemplate("""
% if _memoryLevel == "L1":
${name} = (${type.typeName}) pmsis_l1_malloc(sizeof(${type.referencedType.typeName}) * ${size});\n
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

# ----------------------------------------------------------------------
#
# File: AllocateTemplate.py
#
# Last edited: 23.04.2024
#
# Copyright (C) 2024, ETH Zurich and University of Bologna.
#
# Authors:
# - Philip Wiese (wiesep@iis.ee.ethz.ch), ETH Zurich
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

snitchL2InitTemplate = NodeTemplate("${type.typeName} ${name};\n")

snitchL1InitTemplate = NodeTemplate("${type.typeName} ${name};\n")

snitchL2AllocateTemplate = NodeTemplate(
    "${name} = (${type.typeName}) snrt_l3alloc(sizeof(${type.referencedType.typeName}) * ${size});\n")

snitchL1AllocateTemplate = NodeTemplate(
    "${name} = (${type.typeName}) snrt_l1alloc(sizeof(${type.referencedType.typeName}) * ${size});\n")

snitchL2GlobalInitTemplate = NodeTemplate("static ${type.referencedType.typeName} ${name}[${size}] = {${values}};\n")

snitchL1GlobalInitTemplate = NodeTemplate("static ${type.referencedType.typeName} ${name}[${size}] = {${values}};\n")

snitchL2GlobalAllocateTemplate = NodeTemplate("")

snitchL1GlobalAllocateTemplate = NodeTemplate("")

snitchL2StructInitTemplate = NodeTemplate("""static ${type.typeName} ${name};
""")

snitchL2StructAllocateTemplate = NodeTemplate(""" % for key, value in structDict.items():
    ${name}.${key} = ${value};
% endfor """)

snitchGenericStructInitTemplate = NodeTemplate("""
% if _memoryLevel == "L1":
static ${type.typeName} ${name};\n
% elif _memoryLevel == "L2" or _memoryLevel is None:
static ${type.typeName} ${name};\n
% endif
""")

snitchGenericGlobalInitTemplate = NodeTemplate("""
% if _memoryLevel == "L1":
static ${type.referencedType.typeName} ${name}[${size}] = {${values}};\n
% elif _memoryLevel == "L2" or _memoryLevel is None:
static ${type.referencedType.typeName} ${name}[${size}] = {${values}};\n
% endif
""")

snitchGenericAllocate = NodeTemplate("""
% if _memoryLevel == "L1":
${name} = (${type.typeName}) snrt_l1alloc(sizeof(${type.referencedType.typeName}) * ${size});\n
% elif _memoryLevel == "L2" or _memoryLevel is None:
${name} = (${type.typeName}) snrt_l3alloc(sizeof(${type.referencedType.typeName}) * ${size});\n% else:
//COMPILER BLOCK - MEMORYLEVEL ${_memoryLevel} NOT FOUND \n
${name} = (${type.typeName}) snrt_l3alloc(sizeof(${type.referencedType.typeName}) * ${size});\n
// ${name} with size ${size} allocated in L2!
% endif
""")

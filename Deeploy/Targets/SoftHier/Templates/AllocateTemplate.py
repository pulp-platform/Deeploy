# ----------------------------------------------------------------------
#
# File: AllocateTemplate.py
#
# Last edited: 07.06.2025
#
# Copyright (C) 2025, ETH Zurich and University of Bologna.
#
# Author: Moritz Scherer, ETH Zurich
#         Bowen Wang, ETH Zurich
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

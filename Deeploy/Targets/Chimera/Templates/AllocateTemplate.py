# ----------------------------------------------------------------------
#
# File: AllocateTemplate.py
#
# Last edited: 16.06.2025
#
# Copyright (C) 2025, ETH Zurich and University of Bologna.
#
# Author:
# - Victor Jung, jungvi@iis.ee.ethz.ch, ETH Zurich
#
# ----------------------------------------------------------------------
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the License); you may
# not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an AS IS BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from Deeploy.DeeployTypes import NodeTemplate

memoryIslandAllocateTemplate = NodeTemplate(
    "${name} = (${type.typeName}) memory_island_malloc(${type.referencedType.typeWidth//8} * ${size});\n")

memoryIslandFreeTemplate = NodeTemplate(
    "memory_island_free(${name})")

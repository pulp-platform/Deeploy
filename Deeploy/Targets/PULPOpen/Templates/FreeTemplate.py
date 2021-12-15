# ----------------------------------------------------------------------
#
# File: FreeTemplate.py
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

pulpL2LocalTemplate = NodeTemplate("pi_l2_free(${name}, sizeof(${type.referencedType.typeName}) * ${size});")
pulpL2GlobalTemplate = NodeTemplate("pi_l2_free(${name}, sizeof(${type.referencedType.typeName}) * ${size});")
pulpL1FreeTemplate = NodeTemplate("pmsis_l1_malloc_free(${name}, sizeof(${type.referencedType.typeName}) * ${size});\n")
pulpL1GlobalFreeTemplate = NodeTemplate("")

pulpGenericFree = NodeTemplate("""
% if _memoryLevel == "L1":
pmsis_l1_malloc_free(${name}, sizeof(${type.referencedType.typeName}) * ${size});
% elif _memoryLevel == "L2" or _memoryLevel is None:
pi_l2_free(${name}, sizeof(${type.referencedType.typeName}) * ${size});
% elif _memoryLevel == "L3":
cl_ram_free(${name}, sizeof(${type.referencedType.typeName}) * ${size});
% else:
//COMPILER BLOCK - MEMORYLEVEL ${_memoryLevel} NOT FOUND \n
% endif
""")

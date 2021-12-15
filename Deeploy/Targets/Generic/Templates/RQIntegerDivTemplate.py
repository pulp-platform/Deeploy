# ----------------------------------------------------------------------
#
# File: RQIntegerDivTemplate.py
#
# Last edited: 02.09.2022
#
# Copyright (C) 2022, ETH Zurich and University of Bologna.
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

referenceTemplate = NodeTemplate("""
// RQIntegerDiv (Name: ${nodeName}, Op: ${nodeOp})
SINGLE_CORE RQDiv_s${A_type.referencedType.typeWidth}_s${C_type.referencedType.typeWidth}(${A}, ${B}, ${sizeA}, ${sizeB}, ${nomStep}, ${denomStep}, ${C}, ${Delta}, ${eps}, ${eta}, *${requant_mul}, *${requant_add}, *${requant_div});
""")

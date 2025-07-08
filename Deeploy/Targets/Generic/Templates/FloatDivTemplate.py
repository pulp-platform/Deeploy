# ----------------------------------------------------------------------
#
# File: FloatDivTemplate.py
#
# Last edited: 23.01.2025
#
# Copyright (C) 2023, ETH Zurich and University of Bologna.
#
# Author: Run Wang, ETH Zurich
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
// Division (Name: ${nodeName}, Op: ${nodeOp})
SINGLE_CORE Div_fp${input1_type.referencedType.typeWidth}_fp${input2_type.referencedType.typeWidth}_fp${output_type.referencedType.typeWidth}(${input1}, ${input2}, ${output}, ${size});
""")

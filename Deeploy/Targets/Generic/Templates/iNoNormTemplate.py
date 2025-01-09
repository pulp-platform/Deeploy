# ----------------------------------------------------------------------
#
# File: iNoNormTemplate.py
#
# Last edited: 22.02.2024
#
# Copyright (C) 2024, ETH Zurich and University of Bologna.
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


class _iNoNormTemplate(NodeTemplate):

    def __init__(self, templateStr):
        super().__init__(templateStr)


referenceTemplate = _iNoNormTemplate("""
// iNoNorm (Name: ${nodeName}, Op: ${nodeOp})
SnitchiNoNorm_s${data_in_type.referencedType.typeWidth}_s${data_out_type.referencedType.typeWidth}(${data_in}, ${data_out}, ${weights}, ${bias}, ${size}, ${mul}, ${log2D});
""")

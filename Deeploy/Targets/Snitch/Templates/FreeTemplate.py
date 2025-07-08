# ----------------------------------------------------------------------
#
# File: FreeTemplate.py
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

snitchL2LocalTemplate = NodeTemplate("")
snitchL2GlobalTemplate = NodeTemplate("")
snitchL1FreeTemplate = NodeTemplate("")
snitchL1GlobalFreeTemplate = NodeTemplate("")

snitchGenericFree = NodeTemplate("""
% if _memoryLevel == "L1":
//COMPILER BLOCK - L2 FREE not yet implemented \n
% elif _memoryLevel == "L2" or _memoryLevel is None:
//COMPILER BLOCK - L2 FREE not yet implemented \n
% else:
//COMPILER BLOCK - MEMORYLEVEL ${_memoryLevel} NOT FOUND \n
% endif
""")

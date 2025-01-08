# ----------------------------------------------------------------------
#
# File: RQAddTemplate.py
#
# Last edited: 11.11.2023
#
# Copyright (C) 2023, ETH Zurich and University of Bologna.
#
# Author:
# - Moritz Scherer, ETH Zurich
# - Victor Jung, ETH Zurich
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

from Deeploy.Targets.Generic.Templates.RQAddTemplate import RQAddTemplate

referenceTemplate = RQAddTemplate("""

<%
signatureString = ''
if input_signed:
    signatureString += '_i8'
else:
    signatureString += '_u8'
if input_2_signed:
    signatureString += '_i8'
else:
    signatureString += '_u8'
if output_signed:
    signatureString += '_i8'
else:
    signatureString += '_u8'
%>

// PULP NN RQADD
snitch_nn_add${signatureString}(${data_in_1}, ${data_in_2}, ${data_out}, ${rqs1_mul}, ${rqs1_add}, ${rqs1_log2D}, ${rqs2_mul}, ${rqs2_add}, ${rqs2_log2D}, ${rqsOut_mul}, ${rqsOut_add}, ${rqsOut_log2D}, 1, ${size}, 1, 1);
""")

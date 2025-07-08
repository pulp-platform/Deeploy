# ----------------------------------------------------------------------
#
# File: iSoftmaxTemplate.py
#
# Last edited: 30.05.2024
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

from Deeploy.Targets.Generic.Templates.iSoftmaxPreAllocatedBuffTemplate import iSoftmaxPreAllocatedBuffTemplate

referenceTemplate = iSoftmaxPreAllocatedBuffTemplate("""
<%
signatureString = ''
if input_signed:
    signatureString += '_i8'
else:
    signatureString += '_u8'
if output_signed:
    signatureString += '_i8'
else:
    signatureString += '_u8'
%>
SnitchSoftmax${signatureString}(${data_in}, ${data_out}, ${lastDimBuffer}, ${size}, ${lastDimLength}, ${coeffB}, ${coeffC}, ${log2});
""")

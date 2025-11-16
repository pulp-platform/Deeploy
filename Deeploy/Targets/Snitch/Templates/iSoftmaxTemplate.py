# SPDX-FileCopyrightText: 2024 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

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

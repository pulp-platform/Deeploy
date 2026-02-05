# SPDX-FileCopyrightText: 2023 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

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
pulp_nn_add${signatureString}(${data_in_1}, ${data_in_2}, ${data_out}, ${rqs1_mul}, ${rqs1_add}, ${rqs1_log2D}, ${rqs2_mul}, ${rqs2_add}, ${rqs2_log2D}, ${rqsOut_mul}, ${rqsOut_add}, ${rqsOut_log2D}, 1, ${size}, 1, 1);
""")

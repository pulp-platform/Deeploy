# Copyright (C) 2025, ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, List, Tuple

from Deeploy.DeeployTypes import NetworkContext, NodeTemplate, OperatorRepresentation


class PULPMaxPoolTemplate(NodeTemplate):

    def __init__(self, templateStr):
        super().__init__(templateStr)

    def alignToContext(self, ctxt: NetworkContext,
                       operatorRepresentation: OperatorRepresentation) -> Tuple[NetworkContext, Dict, List[str]]:
        signedI = ctxt.lookup(operatorRepresentation['data_in'])._type.referencedType.typeMin < 0
        operatorRepresentation['input_signed'] = signedI
        return ctxt, operatorRepresentation, []


PULPMaxPool2D_8_Template = PULPMaxPoolTemplate("""
// PULP NN MaxPool 2D
<%
signatureString = ''
if input_signed:
    signatureString += '_i8'
else:
    signatureString += '_u8'
%>
pulp_nn_maxpool${signatureString}(${data_in}, ${data_out}, ${dim_im_in_y}, ${dim_im_in_x}, ${ch_im_in}, ${dim_im_out_y}, ${dim_im_out_x}, ${dim_kernel_y}, ${dim_kernel_x}, ${padding_y_top}, ${padding_y_bottom}, ${padding_x_left}, ${padding_x_right}, ${stride_y}, ${stride_x});
""")

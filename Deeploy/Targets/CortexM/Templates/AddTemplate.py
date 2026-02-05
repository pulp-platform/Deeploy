# SPDX-FileCopyrightText: 2021 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from Deeploy.DeeployTypes import NodeTemplate

AddInt8Template = NodeTemplate("\
arm_add_q7(${data_in_1}, ${data_in_2}, ${data_out}, ${size});\
")
AddInt16Template = NodeTemplate("\
arm_add_q15(${data_in_1}, ${data_in_2}, ${data_out}, ${size});\
")
AddInt32Template = NodeTemplate("\
arm_add_q31(${data_in_1}, ${data_in_2}, ${data_out}, ${size});\
")

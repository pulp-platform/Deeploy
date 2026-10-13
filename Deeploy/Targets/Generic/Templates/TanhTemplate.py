# SPDX-FileCopyrightText: 2026 EPFL
#
# SPDX-License-Identifier: Apache-2.0
#
# Author: Mohammad Hossein Nikkhah

from Deeploy.DeeployTypes import NodeTemplate

referenceTemplate = NodeTemplate("""
// Tan (Name: ${nodeName}, Op: ${nodeOp})
BEGIN_SINGLE_CORE
    tanh_fp32(${data_in}, ${size}, ${data_out});
END_SINGLE_CORE
""")

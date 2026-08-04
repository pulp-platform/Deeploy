# SPDX-FileCopyrightText: 2024 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from Deeploy.DeeployTypes import NodeTemplate

# Multi-core Softmax: all compute cores enter, kernel parallelizes across batch dimension.
# Framework adds snrt_is_compute_core() guard and barriers via SnitchCoreFilterPass/SnitchSynchCoresPass.
FloatSoftmaxTemplateStr = r"""
Softmax_fp32(${data_in}, ${data_out}, ${size}, ${lastDimLength});
"""

FloatSoftmax_Template = NodeTemplate(FloatSoftmaxTemplateStr)

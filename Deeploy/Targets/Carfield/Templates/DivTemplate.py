# SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from Deeploy.DeeployTypes import NodeTemplate

referenceTemplate = NodeTemplate("""
Carfield_Div(${data_in_1}, ${data_in_2}, ${data_out}, ${size}, ${channels}, ${spatialSize}, core_id, numThreads);
""")

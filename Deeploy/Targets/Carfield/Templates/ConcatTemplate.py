# SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from Deeploy.DeeployTypes import NodeTemplate

reference2InputTemplate = NodeTemplate("""
Carfield_Concat2(${data_in_1}, ${data_in_2}, ${data_out}, ${size_1}, ${size_2}, core_id, numThreads);
""")

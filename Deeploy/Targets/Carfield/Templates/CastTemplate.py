# SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from Deeploy.DeeployTypes import NodeTemplate

referenceInt64ToInt32Template = NodeTemplate("""
Carfield_CastInt64ToInt32(${data_in}, ${data_out}, ${size}, core_id, numThreads);
""")

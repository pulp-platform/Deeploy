# SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from Deeploy.DeeployTypes import NodeTemplate

referenceNearest2DTemplate = NodeTemplate("""
Carfield_ResizeNearest2D(${data_in}, ${data_out}, ${channels}, ${H_in}, ${W_in}, ${H_out}, ${W_out}, ${scale_h}, ${scale_w}, core_id, numThreads);
""")

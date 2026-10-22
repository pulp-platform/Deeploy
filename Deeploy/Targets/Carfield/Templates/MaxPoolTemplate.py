# SPDX-License-Identifier: Apache-2.0

from Deeploy.DeeployTypes import NodeTemplate

reference2DTemplate = NodeTemplate("""
Carfield_MaxPool2D(${data_in}, ${data_out}, ${ch_im_in}, ${dim_im_in_x}, ${dim_im_in_y}, ${dim_im_out_x}, ${dim_im_out_y}, ${dim_kernel_x}, ${dim_kernel_y}, ${stride_x}, ${stride_y}, core_id, numThreads);
""")

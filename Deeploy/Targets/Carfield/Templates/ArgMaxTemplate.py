# SPDX-License-Identifier: Apache-2.0

from Deeploy.DeeployTypes import NodeTemplate

referenceChannelTemplate = NodeTemplate("""
Carfield_ArgMaxChannel(${data_in}, ${data_out}, ${channels}, ${spatialSize}, core_id, numThreads);
""")

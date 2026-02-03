# SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from Deeploy.Targets.Generic.Templates.ReshapeTemplate import _ReshapeTemplate

# Use snrt_cluster_core_idx() == 0 instead of SINGLE_CORE macro to avoid core_id dependency
referenceTemplate = _ReshapeTemplate("""
// Reshape (Name: ${nodeName}, Op: ${nodeOp})
if (snrt_cluster_core_idx() == 0) { ${data_out} = ${data_in}; }
""")

# SPDX-FileCopyrightText: 2024 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from Deeploy.DeeployTypes import NodeTemplate

SoftHierLocalTemplate = NodeTemplate("""
if (core_id ==0) {
    simple_free(${name});
}
""")

SoftHierGlobalTemplate = NodeTemplate("""
if (core_id ==0) {
    simple_free(${name});
}
""")

# Copyright (C) 2025, ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
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

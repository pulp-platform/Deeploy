# SPDX-FileCopyrightText: 2021 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from Deeploy.DeeployTypes import NodeTemplate

referenceLocalTemplate = NodeTemplate("""
SINGLE_CORE deeploy_free(${name});
""")

referenceGlobalTemplate = NodeTemplate("""
SINGLE_CORE deeploy_free(${name});
""")

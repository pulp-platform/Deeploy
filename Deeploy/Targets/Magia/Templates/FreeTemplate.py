# SPDX-FileCopyrightText: 2023 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from Deeploy.DeeployTypes import NodeTemplate

magiaFreeTemplate = NodeTemplate("""
% if _memoryLevel == "L1":
magia_l1_free(${name}, sizeof(${type.referencedType.typeName}) * ${size});
% elif _memoryLevel == "L2" or _memoryLevel is None:
magia_l2_free(${name}, sizeof(${type.referencedType.typeName}) * ${size});
% endif
""")

magiaGlobalTemplate = NodeTemplate("magia_l2_free(${name}, sizeof(${type.referencedType.typeName}) * ${size});")
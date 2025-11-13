# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from Deeploy.DeeployTypes import NodeTemplate

gap9L2LocalTemplate = NodeTemplate("pi_l2_free(${name}, sizeof(${type.referencedType.typeName}) * ${size});")
gap9L2GlobalTemplate = NodeTemplate("pi_l2_free(${name}, sizeof(${type.referencedType.typeName}) * ${size});")
gap9L1FreeTemplate = NodeTemplate("pi_l1_malloc_free(${name}, sizeof(${type.referencedType.typeName}) * ${size});\n")
gap9L1GlobalFreeTemplate = NodeTemplate("")

gap9GenericFree = NodeTemplate("""
% if _memoryLevel == "L1":
pi_l1_malloc_free(${name}, sizeof(${type.referencedType.typeName}) * ${size});
% elif _memoryLevel == "L2" or _memoryLevel is None:
pi_l2_free(${name}, sizeof(${type.referencedType.typeName}) * ${size});
% elif _memoryLevel == "L3":
cl_ram_free(${name}, sizeof(${type.referencedType.typeName}) * ${size});
% else:
//COMPILER BLOCK - MEMORYLEVEL ${_memoryLevel} NOT FOUND \n
% endif
""")

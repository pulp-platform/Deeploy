# SPDX-FileCopyrightText: 2023 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from Deeploy.DeeployTypes import NodeTemplate

pulpL2LocalTemplate = NodeTemplate("pi_l2_free(${name}, sizeof(${type.referencedType.typeName}) * ${size});")
pulpL2GlobalTemplate = NodeTemplate("pi_l2_free(${name}, sizeof(${type.referencedType.typeName}) * ${size});")
pulpL1FreeTemplate = NodeTemplate("pmsis_l1_malloc_free(${name}, sizeof(${type.referencedType.typeName}) * ${size});\n")
pulpL1GlobalFreeTemplate = NodeTemplate("")

pulpGenericFree = NodeTemplate("""
% if _memoryLevel == "L1":
pmsis_l1_malloc_free(${name}, sizeof(${type.referencedType.typeName}) * ${size});
% elif _memoryLevel == "L2" or _memoryLevel is None:
pi_l2_free(${name}, sizeof(${type.referencedType.typeName}) * ${size});
% elif _memoryLevel == "L3":
cl_ram_free(${name}, sizeof(${type.referencedType.typeName}) * ${size});
% else:
//COMPILER BLOCK - MEMORYLEVEL ${_memoryLevel} NOT FOUND \n
% endif
""")

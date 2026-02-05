# SPDX-FileCopyrightText: 2024 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from Deeploy.DeeployTypes import NodeTemplate

snitchL2LocalTemplate = NodeTemplate("")
snitchL2GlobalTemplate = NodeTemplate("")
snitchL1FreeTemplate = NodeTemplate("")
snitchL1GlobalFreeTemplate = NodeTemplate("")

snitchGenericFree = NodeTemplate("""
% if _memoryLevel == "L1":
//COMPILER BLOCK - L2 FREE not yet implemented \n
% elif _memoryLevel == "L2" or _memoryLevel is None:
//COMPILER BLOCK - L2 FREE not yet implemented \n
% else:
//COMPILER BLOCK - MEMORYLEVEL ${_memoryLevel} NOT FOUND \n
% endif
""")

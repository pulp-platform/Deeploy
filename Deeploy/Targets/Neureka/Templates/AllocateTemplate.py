# Copyright (C) 2025, ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

from Deeploy.DeeployTypes import NodeTemplate

neurekaGenericGlobalInitTemplate = NodeTemplate("""
% if _memoryLevel == "L1":
static PI_L1 ${type.referencedType.typeName} ${name}[${size}] = {${values}};\n
% elif _memoryLevel == "L2" or _memoryLevel is None:
static PI_L2 ${type.referencedType.typeName} ${name}[${size}] = {${values}};\n
% elif _memoryLevel == "L3":
// ${name} is allocated in L3 \n
static PI_L2 ${type.referencedType.typeName}* ${name};
% elif _memoryLevel == "WeightMemory_SRAM":
static __attribute__((section(".weightmem_sram"))) ${type.referencedType.typeName} ${name}[${size}] = {${values}};\n
% endif
""")

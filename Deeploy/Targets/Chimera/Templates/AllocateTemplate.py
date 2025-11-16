# SPDX-FileCopyrightText: 2024 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from Deeploy.DeeployTypes import NodeTemplate

memoryIslandAllocateTemplate = NodeTemplate(
    "${name} = (${type.typeName}) memory_island_malloc(${type.referencedType.typeWidth//8} * ${size});\n")

memoryIslandFreeTemplate = NodeTemplate("memory_island_free(${name})")

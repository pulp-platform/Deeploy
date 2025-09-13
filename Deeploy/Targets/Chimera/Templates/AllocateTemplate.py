# Copyright (C) 2025, ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

from Deeploy.DeeployTypes import NodeTemplate

memoryIslandAllocateTemplate = NodeTemplate(
    "${name} = (${type.typeName}) memory_island_malloc(${type.referencedType.typeWidth//8} * ${size});\n")

memoryIslandFreeTemplate = NodeTemplate("memory_island_free(${name})")

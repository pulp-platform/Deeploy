# SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from Deeploy.DeeployTypes import NodeTemplate

# Declaration of a runtime-allocated buffer (just a pointer; the memory is
# obtained at runtime by the allocate template below).
spatzInitTemplate = NodeTemplate("${type.typeName} ${name}; // variable buffer of size ${size}\n")

# Runtime allocation: L1 -> TCDM (snrt_l1alloc), L3/None -> DRAM (snrt_l3alloc).
spatzGenericAllocate = NodeTemplate("""
% if _memoryLevel == "L1":
${name} = (${type.typeName}) snrt_l1alloc(sizeof(${type.referencedType.typeName}) * ${size});\n
% elif _memoryLevel == "L3" or _memoryLevel is None:
${name} = (${type.typeName}) snrt_l3alloc(sizeof(${type.referencedType.typeName}) * ${size});\n
% else:
// COMPILER WARNING — unsupported memory level ${_memoryLevel}, defaulting to L3
${name} = (${type.typeName}) snrt_l3alloc(${type.referencedType.typeWidth//8} * ${size});
% endif
""")

# Constant buffers: emitted as static initialized arrays.
spatzGlobalInitTemplate = NodeTemplate("static ${type.referencedType.typeName} ${name}[${size}] = {${values}};\n")

# Struct buffers.
spatzStructInitTemplate = NodeTemplate("""
static ${type.typeName} ${name};
""")

spatzStructAllocateTemplate = NodeTemplate("""
    ${name} = (${structDict.typeName}) ${str(structDict)};
""")

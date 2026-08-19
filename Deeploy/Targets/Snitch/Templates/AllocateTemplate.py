# SPDX-FileCopyrightText: 2024 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from Deeploy.DeeployTypes import NodeTemplate

snitchL2InitTemplate = NodeTemplate("${type.typeName} ${name};\n")

snitchL1InitTemplate = NodeTemplate("${type.typeName} ${name};\n")

snitchL2AllocateTemplate = NodeTemplate(
    "${name} = (${type.typeName}) snrt_l3_alloc(sizeof(${type.referencedType.typeName}) * ${size});\n")

snitchL1AllocateTemplate = NodeTemplate(
    "${name} = (${type.typeName}) snrt_l1_alloc(sizeof(${type.referencedType.typeName}) * ${size});\n")

snitchL2GlobalInitTemplate = NodeTemplate("static ${type.referencedType.typeName} ${name}[${size}] = {${values}};\n")

snitchL1GlobalInitTemplate = NodeTemplate("static ${type.referencedType.typeName} ${name}[${size}] = {${values}};\n")

snitchL2GlobalAllocateTemplate = NodeTemplate("")

snitchL1GlobalAllocateTemplate = NodeTemplate("")

snitchL2StructInitTemplate = NodeTemplate("""static ${type.typeName} ${name};
""")

snitchL2StructAllocateTemplate = NodeTemplate(""" % for key, value in structDict.items():
    ${name}.${key} = ${value};
% endfor """)

snitchGenericStructInitTemplate = NodeTemplate("""
% if _memoryLevel == "L1":
static ${type.typeName} ${name};\n
% elif _memoryLevel == "L2" or _memoryLevel is None:
static ${type.typeName} ${name};\n
% endif
""")

snitchGenericGlobalInitTemplate = NodeTemplate("""
% if _memoryLevel == "L1":
static ${type.referencedType.typeName} ${name}[${size}] = {${values}};\n
% elif _memoryLevel == "L2" or _memoryLevel is None:
static ${type.referencedType.typeName} ${name}[${size}] = {${values}};\n
% endif
""")

snitchGenericGuardedAllocate = NodeTemplate("""
<%
# TransientBuffers are void pointers whose size is already expressed in bytes.
# sizeof(void) is a GNU C extension that evaluates to 1; it is invalid in C++,
# which the Snitch platform is compiled as.
if type.referencedType.typeName == "void":
    allocSize = f"{size}"
else:
    allocSize = f"sizeof({type.referencedType.typeName}) * {size}"
%>
% if _memoryLevel == "L1":
if (snrt_is_dm_core()) { ${name} = (${type.typeName}) snrt_l1_alloc(${allocSize}); }
snrt_cluster_hw_barrier();\n
% else:
if (snrt_is_dm_core()) { ${name} = (${type.typeName}) snrt_l3_alloc(${allocSize}); }
snrt_cluster_hw_barrier();\n
% endif
""")

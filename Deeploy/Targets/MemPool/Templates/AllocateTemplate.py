# SPDX-FileCopyrightText: 2021 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from Deeploy.DeeployTypes import NodeTemplate

MemPoolInitTemplate = NodeTemplate("${type.typeName} ${name} __attribute__((section(\".l1\")));\n")
MemPoolAllocateTemplate = NodeTemplate("""
if (core_id ==0) {
    ## #if DEEPLOY_TRACE_MALLOC
    ## deeploy_log("[Deeploy] Alloc ${name} (${type.referencedType.typeName} * ${size})\\r\\n");
    ## alloc_dump(get_alloc_l1());
    ## #endif

    ${name} = (${type.typeName}) deeploy_malloc(sizeof(${type.referencedType.typeName}) * ${size});

    ## #if DEEPLOY_TRACE_MALLOC
    ## deeploy_log("  -> @ %p\\r\\n", ${name});
    ## alloc_dump(get_alloc_l1());
    ## #endif
}
""")

MemPoolGlobalInitTemplate = NodeTemplate(
    "static ${type.referencedType.typeName} ${name}[${size}] __attribute__((section(\".l2\"))) = {${values}};\n")
MemPoolGlobalAllocateTemplate = NodeTemplate("")

MemPoolStructInitTemplate = NodeTemplate("""
static ${type.typeName} ${name} __attribute__((section(\".l1\")));
""")
#static const ${type}* ${name} = &${name}_UL;

MemPoolStructAllocateTemplate = NodeTemplate("""
if (core_id == 0) {
    ${name} = (${structDict.typeName}) ${str(structDict)};
}
""")

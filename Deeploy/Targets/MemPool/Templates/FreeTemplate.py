# SPDX-FileCopyrightText: 2021 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from Deeploy.DeeployTypes import NodeTemplate

MemPoolLocalTemplate = NodeTemplate("""
if (core_id ==0) {
    ## #if DEEPLOY_TRACE_MALLOC
    ## deeploy_log("[Deeploy] Free ${name} @ %p\\r\\n", ${name});
    ## alloc_dump(get_alloc_l1());
    ## #endif

    simple_free(${name});

    ## #if DEEPLOY_TRACE_MALLOC
    ## alloc_dump(get_alloc_l1());
    ## #endif
}
""")

MemPoolGlobalTemplate = NodeTemplate("""
if (core_id ==0) {
    ## #if DEEPLOY_TRACE_MALLOC
    ## deeploy_log("[Deeploy] Free ${name} @ %p\\r\\n", ${name});
    ## alloc_dump(get_alloc_l1());
    ## #endif

    simple_free(${name});

    ## #if DEEPLOY_TRACE_MALLOC
    ## alloc_dump(get_alloc_l1());
    ## #endif
}
""")
